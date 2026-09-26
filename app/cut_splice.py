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


@dataclass
class CutSpliceResult:
    wav_bytes: bytes
    output_frames: int


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


def _kept_segments(
    samples: np.ndarray, intervals: list[tuple[int, int]]
) -> list[np.ndarray]:
    """Non-empty kept regions between the deletion intervals, in order."""
    segments: list[np.ndarray] = []
    cursor = 0
    for start, end in intervals:
        if start > cursor:
            segments.append(samples[cursor:start])
        cursor = end
    if cursor < samples.shape[0]:
        segments.append(samples[cursor:])
    return segments


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
) -> CutSpliceResult:
    samples = _read_wav(wav_bytes)
    n_frames, channels = samples.shape
    fade_frames = FRAMES_PER_MS * crossfade_ms

    intervals = _validate_intervals(remove_intervals, n_frames)
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
    return CutSpliceResult(
        wav_bytes=_write_wav(pcm, channels), output_frames=int(pcm.shape[0])
    )
