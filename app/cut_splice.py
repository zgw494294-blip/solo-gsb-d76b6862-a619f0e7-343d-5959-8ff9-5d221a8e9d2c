"""Cut & splice: delete half-open sample intervals and crossfade the seams.

The regions of the input outside ``remove_intervals`` are concatenated in
their original order. At every seam the last F frames of the left segment
and the first F frames of the right segment are overlapped and mixed on
every channel with linear weights: the right segment fades in from 0 to 1
(endpoints included) while the left segment fades out from 1 to 0, where
F = 48 * crossfade_ms frames. F = 0 is a plain hard splice. With seams,
the first/last kept segments must hold at least F frames and every middle
segment at least 2F frames, so the two fades of a middle segment never
share a sample. The output keeps the input format and may be shorter
than one second.

Optional ``cue_intervals`` are user annotations on the input sample axis.
Each cue is split at the deletion boundaries and the surviving pieces are
mapped onto the output axis. A frame at local offset t of kept segment j
(0-based) lands at output frame ``sum(len(segments[:j])) - j*F + t``, so
the two pieces of a cue straddling a seam may map to overlapping output
intervals inside the crossfade region; both pieces are reported as
separate fragments.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .loudness import (
    SAMPLE_RATE,
    AudioValidationError,
    _quantize,
    _read_wav,
    _write_wav,
)

# 48 kHz -> 48 frames per millisecond of crossfade.
FRAMES_PER_MS = SAMPLE_RATE // 1000

# At most this many cue annotations per request.
MAX_CUE_INTERVALS = 2000


@dataclass
class CueFragment:
    """One surviving piece of a cue: a half-open source piece and the
    half-open output interval its frames land on (the two may overlap with
    other fragments inside a crossfade region)."""

    source_interval: tuple[int, int]
    output_interval: tuple[int, int]


@dataclass
class MappedCue:
    id: str
    fragments: list[CueFragment]


@dataclass
class CutSpliceResult:
    wav_bytes: bytes
    output_frames: int
    mapped_cues: list[MappedCue]


def _validate_intervals(
    remove_intervals: list[tuple[int, int]], n_frames: int
) -> list[tuple[int, int]]:
    """Check the half-open deletion intervals and return them as plain ints.

    Every interval must be non-empty and inside [0, n_frames]; the list
    must be strictly ascending with non-touching neighbours (a gap of at
    least one kept frame between consecutive intervals).
    """
    intervals: list[tuple[int, int]] = []
    for start, end in remove_intervals:
        if not 0 <= start < end <= n_frames:
            raise AudioValidationError(
                f"Interval [{start}, {end}) is empty or outside "
                f"[0, {n_frames})"
            )
        if intervals and start <= intervals[-1][1]:
            raise AudioValidationError(
                "remove_intervals must be strictly ascending and non-touching"
            )
        intervals.append((start, end))
    return intervals


def _segment_bounds(
    n_frames: int, intervals: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Half-open [start, end) bounds of the kept regions, in order."""
    bounds: list[tuple[int, int]] = []
    cursor = 0
    for start, end in intervals:
        if start > cursor:
            bounds.append((cursor, start))
        cursor = end
    if cursor < n_frames:
        bounds.append((cursor, n_frames))
    return bounds


def _kept_segments(
    samples: np.ndarray, intervals: list[tuple[int, int]]
) -> list[np.ndarray]:
    """Non-empty kept regions between the deletion intervals, in order."""
    return [samples[start:end] for start, end in _segment_bounds(
        samples.shape[0], intervals
    )]


def _validate_cues(
    cue_intervals: list[tuple[str, int, int]], n_frames: int
) -> list[tuple[str, int, int]]:
    """Check cue annotations: non-empty unique ids and non-empty in-bounds
    half-open intervals. Order is arbitrary and intervals may overlap."""
    if len(cue_intervals) > MAX_CUE_INTERVALS:
        raise AudioValidationError(
            f"cue_intervals may hold at most {MAX_CUE_INTERVALS} entries"
        )
    cues: list[tuple[str, int, int]] = []
    seen_ids: set[str] = set()
    for cue_id, start, end in cue_intervals:
        if not isinstance(cue_id, str) or not cue_id:
            raise AudioValidationError("cue_intervals ids must be non-empty strings")
        if cue_id in seen_ids:
            raise AudioValidationError(f"Duplicate cue id: {cue_id!r}")
        if not 0 <= start < end <= n_frames:
            raise AudioValidationError(
                f"Cue {cue_id!r} interval [{start}, {end}) is empty or outside "
                f"[0, {n_frames})"
            )
        seen_ids.add(cue_id)
        cues.append((cue_id, start, end))
    return cues


def _map_cues(
    cues: list[tuple[str, int, int]],
    seg_bounds: list[tuple[int, int]],
    fade_frames: int,
) -> list[MappedCue]:
    """Split every cue against the kept segments and place the pieces on the
    output axis, preserving request order.

    Segment j starts at output frame ``prefix - j*F`` where ``prefix`` is
    the total length of segments 0..j-1, so a frame at local offset t of
    segment j lands at ``prefix - j*F + t``. Pieces straddling a seam may
    therefore map onto overlapping output intervals inside the crossfade
    region; each piece is reported separately.

    A sorted sweep over segment and cue boundaries avoids iterating every
    cue over every segment. At one coordinate the visit order is segment
    end -> cue end -> segment start -> cue start: ends before starts for
    half-open semantics, and a segment starts before a cue sharing its
    edge (the cue-start handler then finds the segment active).
    """
    if not cues:
        return []

    # Output frame at each segment start: total frames of earlier segments
    # minus one F per earlier seam.
    seg_out_start: list[int] = []
    prefix = 0
    for j, (start, end) in enumerate(seg_bounds):
        seg_out_start.append(prefix - j * fade_frames)
        prefix += end - start

    # (coordinate, end(0)|start(1), segment(0)|cue(1), index)
    events: list[tuple[int, int, int, int]] = []
    for j, (start, end) in enumerate(seg_bounds):
        events.append((start, 1, 0, j))
        events.append((end, 0, 0, j))
    for c, (_id, start, end) in enumerate(cues):
        events.append((start, 1, 1, c))
        events.append((end, 0, 1, c))

    active_segments: set[int] = set()
    active_cues: set[int] = set()
    # cue index -> (source start, output start) of its open piece; at most
    # one per cue since kept segments never overlap.
    open_pieces: dict[int, tuple[int, int]] = {}
    # Reverse lookup: which segment holds each open piece.
    piece_segment: dict[int, int] = {}
    # segment index -> cues with an open piece in that segment, so a segment
    # end closes only its own pieces (no cue x segment scan).
    open_by_segment: dict[int, set[int]] = {}
    pieces: dict[int, list[CueFragment]] = {c: [] for c in range(len(cues))}

    def close_piece(c: int, x: int) -> None:
        s_start, o_start = open_pieces.pop(c)
        piece_segment.pop(c, None)
        pieces[c].append(
            CueFragment(
                source_interval=(s_start, x),
                output_interval=(o_start, o_start + x - s_start),
            )
        )

    for x, edge, kind, index in sorted(events):
        if edge == 0 and kind == 0:
            # Segment end: close every piece open in this segment; a cue
            # that continues into the deletion stays active.
            for c in open_by_segment.pop(index, ()):
                close_piece(c, x)
            active_segments.remove(index)
        elif edge == 0:
            # Cue end inside an active segment: flush its open piece. If the
            # cue ends exactly at a segment edge that segment-end event ran
            # first and already closed the piece.
            active_cues.discard(index)
            if index in open_pieces:
                open_by_segment[piece_segment[index]].discard(index)
                close_piece(index, x)
        elif kind == 0:
            # Segment start: resume every cue spanning the preceding cut.
            # Such cues have no open piece (it was closed at the cut edge).
            active_segments.add(index)
            seg_start = seg_bounds[index][0]
            members: set[int] = set()
            for c in active_cues:
                open_pieces[c] = (x, seg_out_start[index] + x - seg_start)
                piece_segment[c] = index
                members.add(c)
            open_by_segment[index] = members
        else:
            # Cue start: the non-empty gaps between kept segments guarantee
            # at most one segment is active at any coordinate.
            active_cues.add(index)
            for j in active_segments:
                seg_start = seg_bounds[j][0]
                open_pieces[index] = (
                    x,
                    seg_out_start[j] + x - seg_start,
                )
                piece_segment[index] = j
                open_by_segment.setdefault(j, set()).add(index)

    return [
        MappedCue(id=cue_id, fragments=pieces[c])
        for c, (cue_id, _start, _end) in enumerate(cues)
    ]


def _splice(segments: list[np.ndarray], fade_frames: int) -> np.ndarray:
    """Concatenate the kept segments, crossfading F frames at every seam."""
    if fade_frames == 0 or len(segments) == 1:
        return np.concatenate(segments, axis=0)
    # Right-segment weight per overlap frame: 0 -> 1, endpoints included.
    weight = np.linspace(0.0, 1.0, fade_frames).reshape(-1, 1)
    pieces: list[np.ndarray] = []
    prev = segments[0]
    for seg in segments[1:]:
        pieces.append(prev[:-fade_frames])
        pieces.append(
            prev[-fade_frames:] * (1.0 - weight) + seg[:fade_frames] * weight
        )
        prev = seg[fade_frames:]
    pieces.append(prev)
    return np.concatenate(pieces, axis=0)


def cut_splice_wav(
    wav_bytes: bytes,
    remove_intervals: list[tuple[int, int]],
    crossfade_ms: int,
    cue_intervals: list[tuple[str, int, int]] | None = None,
) -> CutSpliceResult:
    samples = _read_wav(wav_bytes)
    n_frames, channels = samples.shape
    fade_frames = FRAMES_PER_MS * crossfade_ms

    intervals = _validate_intervals(remove_intervals, n_frames)
    cues = _validate_cues(cue_intervals or [], n_frames)
    segments = _kept_segments(samples, intervals)
    if not segments:
        raise AudioValidationError("remove_intervals delete the entire file")

    if fade_frames > 0 and len(segments) > 1:
        # Every seam borrows F tail frames from its left segment and F head
        # frames from its right one; a middle segment feeds two seams, so it
        # needs 2F frames to keep the two fades from sharing samples.
        for index, seg in enumerate(segments):
            edge = index == 0 or index == len(segments) - 1
            required = fade_frames if edge else 2 * fade_frames
            if seg.shape[0] < required:
                raise AudioValidationError(
                    f"Kept segment {index} has {seg.shape[0]} frames but the "
                    f"crossfade needs at least {required}"
                )

    spliced = _splice(segments, fade_frames)
    pcm = _quantize(spliced)
    mapped_cues = _map_cues(
        cues, _segment_bounds(n_frames, intervals), fade_frames
    )
    return CutSpliceResult(
        wav_bytes=_write_wav(pcm, channels),
        output_frames=int(pcm.shape[0]),
        mapped_cues=mapped_cues,
    )
