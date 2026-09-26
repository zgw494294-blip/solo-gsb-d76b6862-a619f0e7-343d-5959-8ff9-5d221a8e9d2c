import base64
import io
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

RATE = 48000
N = 2 * RATE  # 2 s of audio


def wav_b64(pcm: np.ndarray, channels: int = 1, rate: int = RATE) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.astype("<i2").tobytes())
    return base64.b64encode(buf.getvalue()).decode()


def decode_wav(b64: str):
    raw = base64.b64decode(b64)
    with wave.open(io.BytesIO(raw), "rb") as wf:
        params = (wf.getnchannels(), wf.getsampwidth(), wf.getframerate())
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2")
    return params, pcm


def make_ramp(n=N, seed=7):
    # Deterministic non-constant content so splices are easy to verify.
    rng = np.random.default_rng(seed)
    return rng.integers(-20000, 20000, size=n).astype("<i2")


def post_cut(pcm, intervals, crossfade_ms=0, channels=1, rate=RATE, cues=None):
    payload = {
        "wav_base64": wav_b64(pcm, channels, rate),
        "remove_intervals": intervals,
        "crossfade_ms": crossfade_ms,
    }
    if cues is not None:
        payload["cue_intervals"] = cues
    return client.post("/v1/cut-splice", json=payload)


def test_hard_splice_removes_middle():
    pcm = make_ramp()
    resp = post_cut(pcm, [[RATE, RATE + 5000]], crossfade_ms=0)
    assert resp.status_code == 200
    data = resp.json()
    assert data["output_frames"] == N - 5000
    params, out = decode_wav(data["wav_base64"])
    assert params == (1, 2, RATE)
    assert np.array_equal(out, np.concatenate([pcm[:RATE], pcm[RATE + 5000 :]]))


def test_empty_intervals_returns_input_unchanged():
    pcm = make_ramp()
    resp = post_cut(pcm, [], crossfade_ms=5)
    assert resp.status_code == 200
    data = resp.json()
    assert data["output_frames"] == N
    _, out = decode_wav(data["wav_base64"])
    assert np.array_equal(out, pcm)


def test_delete_prefix_and_suffix():
    pcm = make_ramp()
    resp = post_cut(pcm, [[0, 1000], [N - 2000, N]], crossfade_ms=0)
    assert resp.status_code == 200
    data = resp.json()
    assert data["output_frames"] == N - 3000
    _, out = decode_wav(data["wav_base64"])
    assert np.array_equal(out, pcm[1000 : N - 2000])


def test_multiple_intervals_hard_splice():
    pcm = make_ramp()
    cuts = [[1000, 2000], [5000, 6000], [9000, 9500]]
    resp = post_cut(pcm, cuts, crossfade_ms=0)
    assert resp.status_code == 200
    data = resp.json()
    kept = np.concatenate(
        [pcm[:1000], pcm[2000:5000], pcm[6000:9000], pcm[9500:]]
    )
    assert data["output_frames"] == kept.size
    _, out = decode_wav(data["wav_base64"])
    assert np.array_equal(out, kept)


def test_crossfade_blends_overlap_linearly():
    pcm = np.empty(N, dtype="<i2")
    pcm[: RATE] = 1000
    pcm[RATE :] = 3000
    fade = 48  # crossfade_ms = 1
    resp = post_cut(pcm, [[RATE - 100, RATE + 100]], crossfade_ms=1)
    assert resp.status_code == 200
    data = resp.json()
    kept = (RATE - 100) + (RATE - 100)
    assert data["output_frames"] == kept - fade
    params, out = decode_wav(data["wav_base64"])
    assert params == (1, 2, RATE)
    head = RATE - 100 - fade
    assert np.all(out[:head] == 1000)
    assert np.all(out[head + fade :] == 3000)
    mixed = out[head : head + fade].astype(np.float64)
    weight = np.linspace(0.0, 1.0, fade)
    expected = 1000.0 * (1.0 - weight) + 3000.0 * weight
    # endpoints included: first frame is fully left, last fully right
    assert mixed[0] == 1000
    assert mixed[-1] == 3000
    assert np.allclose(mixed, expected, atol=1.0)


def test_crossfade_frame_count_multiple_seams():
    pcm = make_ramp()
    cuts = [[10000, 20000], [40000, 50000]]
    fade = 48 * 20
    resp = post_cut(pcm, cuts, crossfade_ms=20)
    assert resp.status_code == 200
    data = resp.json()
    kept = N - 10000 - 10000
    assert data["output_frames"] == kept - 2 * fade
    _, out = decode_wav(data["wav_base64"])
    assert out.size == kept - 2 * fade


def test_crossfade_stereo_fades_every_channel():
    left = np.empty(N, dtype="<i2")
    left[: RATE] = 1000
    left[RATE :] = 3000
    right = np.empty(N, dtype="<i2")
    right[: RATE] = -500
    right[RATE :] = -2500
    pcm = np.stack([left, right], axis=1)
    fade = 48 * 2
    resp = post_cut(pcm, [[RATE - 100, RATE + 100]], crossfade_ms=2, channels=2)
    assert resp.status_code == 200
    data = resp.json()
    params, out = decode_wav(data["wav_base64"])
    assert params == (2, 2, RATE)
    out = out.reshape(-1, 2)
    head = RATE - 100 - fade
    weight = np.linspace(0.0, 1.0, fade)
    for col, (a, b) in enumerate([(1000.0, 3000.0), (-500.0, -2500.0)]):
        assert np.all(out[:head, col] == int(a))
        assert np.all(out[head + fade :, col] == int(b))
        mixed = out[head : head + fade, col].astype(np.float64)
        expected = a * (1.0 - weight) + b * weight
        assert mixed[0] == a
        assert mixed[-1] == b
        assert np.allclose(mixed, expected, atol=1.0)


def test_output_may_be_shorter_than_one_second():
    pcm = make_ramp()
    resp = post_cut(pcm, [[1000, N - 1000]], crossfade_ms=0)
    assert resp.status_code == 200
    data = resp.json()
    assert data["output_frames"] == 2000  # ~42 ms
    _, out = decode_wav(data["wav_base64"])
    assert np.array_equal(out, np.concatenate([pcm[:1000], pcm[N - 1000 :]]))


@pytest.mark.parametrize(
    "intervals",
    [
        [[100, 500], [400, 900]],      # overlapping
        [[100, 500], [500, 900]],      # touching (no kept frame between)
        [[500, 900], [100, 200]],      # not ascending
        [[0, N + 1]],                  # end out of bounds
        [[-1, 100]],                   # negative start
        [[N, N + 10]],                 # start out of bounds
        [[100, 100]],                  # empty interval
        [[500, 100]],                  # start after end
    ],
)
def test_invalid_intervals_return_422(intervals):
    assert post_cut(make_ramp(), intervals).status_code == 422


def test_delete_everything_returns_422():
    assert post_cut(make_ramp(), [[0, N]]).status_code == 422


def test_first_segment_too_short_for_fade_returns_422():
    # Kept: [0, 30) and [5000, N); 30 < F = 48.
    assert post_cut(make_ramp(), [[30, 5000]], crossfade_ms=1).status_code == 422


def test_last_segment_too_short_for_fade_returns_422():
    assert (
        post_cut(make_ramp(), [[5000, N - 30]], crossfade_ms=1).status_code == 422
    )


def test_middle_segment_too_short_for_two_fades_returns_422():
    # Kept middle segment [2000, 2060) has 60 frames < 2F = 96.
    cuts = [[1000, 2000], [2060, 3000]]
    assert post_cut(make_ramp(), cuts, crossfade_ms=1).status_code == 422


def test_minimal_fade_capacity_accepted():
    pcm = make_ramp()
    fade = 48
    # First/last segments exactly F, middle exactly 2F.
    cuts = [[fade, 5000], [5000 + 2 * fade, N - fade]]
    resp = post_cut(pcm, cuts, crossfade_ms=1)
    assert resp.status_code == 200
    kept = fade + 2 * fade + fade
    assert resp.json()["output_frames"] == kept - 2 * fade


@pytest.mark.parametrize("crossfade_ms", [-1, 21])
def test_crossfade_ms_out_of_range_returns_422(crossfade_ms):
    resp = post_cut(make_ramp(), [[100, 200]], crossfade_ms=crossfade_ms)
    assert resp.status_code == 422


def test_invalid_base64_returns_422():
    resp = client.post(
        "/v1/cut-splice",
        json={
            "wav_base64": "!!!not-base64!!!",
            "remove_intervals": [[0, 100]],
            "crossfade_ms": 0,
        },
    )
    assert resp.status_code == 422


def test_garbage_wav_returns_422():
    resp = client.post(
        "/v1/cut-splice",
        json={
            "wav_base64": base64.b64encode(b"not a wav").decode(),
            "remove_intervals": [[0, 100]],
            "crossfade_ms": 0,
        },
    )
    assert resp.status_code == 422


def test_non_compliant_wav_returns_422():
    assert post_cut(make_ramp(), [[0, 100]], rate=44100).status_code == 422


@pytest.mark.parametrize("field", ["wav_base64", "remove_intervals", "crossfade_ms"])
def test_missing_field_returns_422(field):
    payload = {
        "wav_base64": wav_b64(make_ramp()),
        "remove_intervals": [[0, 100]],
        "crossfade_ms": 0,
    }
    del payload[field]
    assert client.post("/v1/cut-splice", json=payload).status_code == 422


# --- cue_intervals / mapped_cues ---------------------------------------

def cue(cid, start, end):
    return {"id": cid, "start": start, "end": end}


def test_mapped_cues_empty_when_absent_or_empty():
    pcm = make_ramp()
    resp = post_cut(pcm, [[100, 200]])
    assert resp.status_code == 200
    assert resp.json()["mapped_cues"] == []

    resp = post_cut(pcm, [[100, 200]], cues=[])
    assert resp.status_code == 200
    assert resp.json()["mapped_cues"] == []


def test_mapped_cues_identity_without_cuts():
    pcm = make_ramp()
    resp = post_cut(
        pcm, [], cues=[cue("a", 100, 200), cue("b", 0, N)]
    )
    assert resp.status_code == 200
    assert resp.json()["mapped_cues"] == [
        {"id": "a", "fragments": [
            {"source_interval": [100, 200], "output_interval": [100, 200]}]},
        {"id": "b", "fragments": [
            {"source_interval": [0, N], "output_interval": [0, N]}]},
    ]


def test_mapped_cues_hard_splice_multiple_cuts():
    pcm = make_ramp()
    cuts = [[1000, 2000], [5000, 6000], [9000, 9500]]
    # kept: [0,1000) -> out 0; [2000,5000) -> out 1000;
    #       [6000,9000) -> out 4000; [9500,N) -> out 7000
    cues_req = [
        cue("whole", 0, N),          # one piece per kept segment
        cue("prefix", 0, 500),       # entirely in first segment
        cue("across-one", 800, 2500),   # [800,1000)+[2000,2500)
        cue("across-two", 4500, 7000),  # [4500,5000)+[6000,7000)
        cue("deleted", 1000, 2000),     # exactly a deletion
        cue("inside-cut", 5200, 5800),  # fully inside a deletion
        cue("edges", 2000, 5000),       # exactly a kept segment
        cue("tail", N - 100, N),
    ]
    resp = post_cut(pcm, cuts, cues=cues_req)
    assert resp.status_code == 200
    mapped = {m["id"]: m["fragments"] for m in resp.json()["mapped_cues"]}
    assert list(mapped) == [
        "whole", "prefix", "across-one", "across-two",
        "deleted", "inside-cut", "edges", "tail",
    ]
    assert mapped["whole"] == [
        {"source_interval": [0, 1000], "output_interval": [0, 1000]},
        {"source_interval": [2000, 5000], "output_interval": [1000, 4000]},
        {"source_interval": [6000, 9000], "output_interval": [4000, 7000]},
        {"source_interval": [9500, N], "output_interval": [7000, N - 2500]},
    ]
    assert mapped["prefix"] == [
        {"source_interval": [0, 500], "output_interval": [0, 500]}]
    assert mapped["across-one"] == [
        {"source_interval": [800, 1000], "output_interval": [800, 1000]},
        {"source_interval": [2000, 2500], "output_interval": [1000, 1500]},
    ]
    assert mapped["across-two"] == [
        {"source_interval": [4500, 5000], "output_interval": [3500, 4000]},
        {"source_interval": [6000, 7000], "output_interval": [4000, 5000]},
    ]
    assert mapped["deleted"] == []
    assert mapped["inside-cut"] == []
    assert mapped["edges"] == [
        {"source_interval": [2000, 5000], "output_interval": [1000, 4000]}]
    assert mapped["tail"] == [
        {"source_interval": [N - 100, N],
         "output_interval": [N - 2600, N - 2500]},
    ]


def test_mapped_cues_piece_frames_match_output_pcm_hard_splice():
    pcm = make_ramp()
    cuts = [[1000, 2000], [5000, 6000]]
    resp = post_cut(pcm, cuts, cues=[cue("c", 500, 8000)])
    assert resp.status_code == 200
    _, out = decode_wav(resp.json()["wav_base64"])
    fragments = resp.json()["mapped_cues"][0]["fragments"]
    for frag in fragments:
        s0, s1 = frag["source_interval"]
        o0, o1 = frag["output_interval"]
        assert s1 - s0 == o1 - o0
        assert np.array_equal(out[o0:o1], pcm[s0:s1])


def test_cue_order_preserved_and_overlaps_allowed():
    pcm = make_ramp()
    resp = post_cut(
        pcm,
        [[3000, 4000]],
        cues=[cue("z", 0, 100), cue("a", 0, N), cue("m", 0, 100)],
    )
    assert resp.status_code == 200
    assert [m["id"] for m in resp.json()["mapped_cues"]] == ["z", "a", "m"]


def test_mapped_cues_crossfade_overlap_pieces_kept_separately():
    pcm = make_ramp()
    cuts = [[10000, 20000], [40000, 50000]]
    fade = 48 * 20  # 960
    resp = post_cut(pcm, cuts, crossfade_ms=20, cues=[cue("x", 0, N)])
    assert resp.status_code == 200
    data = resp.json()
    # kept 76000 - 2*960 = 74080 output frames (unchanged behavior)
    assert data["output_frames"] == 76000 - 2 * fade
    fragments = data["mapped_cues"][0]["fragments"]
    assert fragments == [
        {"source_interval": [0, 10000], "output_interval": [0, 10000]},
        {"source_interval": [20000, 40000],
         "output_interval": [10000 - fade, 30000 - fade]},
        {"source_interval": [50000, N],
         "output_interval": [30000 - 2 * fade, 76000 - 2 * fade]},
    ]
    # Adjacent pieces overlap by exactly F frames in the crossfade region;
    # F is subtracted once per seam, never twice per segment.
    first, second, third = (f["output_interval"] for f in fragments)
    assert second[0] == first[1] - fade
    assert third[0] == second[1] - fade
    assert third[1] == data["output_frames"]


def test_mapped_cues_small_cues_inside_crossfade_zones():
    pcm = make_ramp()
    fade = 48  # crossfade_ms = 1
    resp = post_cut(
        pcm,
        [[10000, 20000]],
        crossfade_ms=1,
        cues=[
            cue("left-tail", 9900, 10000),
            cue("right-head", 20000, 20100),
            cue("bridge", 9900, 20100),
        ],
    )
    assert resp.status_code == 200
    mapped = {m["id"]: m["fragments"]
              for m in resp.json()["mapped_cues"]}
    assert mapped["left-tail"] == [
        {"source_interval": [9900, 10000], "output_interval": [9900, 10000]}]
    assert mapped["right-head"] == [
        {"source_interval": [20000, 20100],
         "output_interval": [10000 - fade, 10100 - fade]}]
    # Both sides of the seam map onto overlapping output ranges [9952,10000)
    # and are still reported as two separate fragments.
    assert mapped["bridge"] == [
        {"source_interval": [9900, 10000], "output_interval": [9900, 10000]},
        {"source_interval": [20000, 20100],
         "output_interval": [9952, 10052]},
    ]


def test_mapped_cues_boundary_touching_deletions():
    pcm = make_ramp()
    resp = post_cut(
        pcm,
        [[1000, 2000]],
        cues=[
            cue("ends-at-cut-start", 0, 1000),
            cue("starts-at-cut-end", 2000, 3000),
        ],
    )
    assert resp.status_code == 200
    mapped = {m["id"]: m["fragments"]
              for m in resp.json()["mapped_cues"]}
    assert mapped["ends-at-cut-start"] == [
        {"source_interval": [0, 1000], "output_interval": [0, 1000]}]
    assert mapped["starts-at-cut-end"] == [
        {"source_interval": [2000, 3000], "output_interval": [1000, 2000]}]


@pytest.mark.parametrize(
    "bad_cues",
    [
        [cue("a", 0, 10), cue("a", 20, 30)],      # duplicate id
        [cue("", 0, 10)],                          # empty id
        [cue("a", 100, 100)],                      # empty interval
        [cue("a", -1, 100)],                       # negative start
        [cue("a", N, N + 1)],                      # start out of bounds
        [cue("a", 0, N + 1)],                      # end out of bounds
        [cue("a", 500, 100)],                      # start after end
    ],
)
def test_invalid_cue_intervals_return_422(bad_cues):
    assert post_cut(make_ramp(), [[100, 200]], cues=bad_cues).status_code == 422


def test_too_many_cue_intervals_returns_422():
    pcm = make_ramp()
    too_many = [cue(f"c{i}", i, i + 1) for i in range(2001)]
    assert post_cut(pcm, [], cues=too_many).status_code == 422


def test_exactly_2000_cue_intervals_accepted():
    pcm = make_ramp()
    cues_req = [cue(f"c{i}", i, i + 1) for i in range(2000)]
    resp = post_cut(pcm, [], cues=cues_req)
    assert resp.status_code == 200
    mapped = resp.json()["mapped_cues"]
    assert len(mapped) == 2000
    assert mapped[1234] == {
        "id": "c1234",
        "fragments": [
            {"source_interval": [1234, 1235],
             "output_interval": [1234, 1235]}],
    }


def test_invalid_cues_422_even_when_audio_otherwise_valid():
    # Duplicate ids must surface as 422 regardless of other fields.
    pcm = make_ramp()
    resp = post_cut(
        pcm, [], cues=[cue("dup", 0, 10), cue("dup", 20, 30)]
    )
    assert resp.status_code == 422


@pytest.mark.parametrize(
    "bad_payload",
    [
        {"id": "a", "start": 0},               # missing end
        {"id": "a", "start": 0, "end": 1.5},   # non-integer end
        {"id": 7, "start": 0, "end": 10},      # non-string id
    ],
)
def test_malformed_cue_objects_return_422(bad_payload):
    pcm = make_ramp()
    resp = post_cut(pcm, [], cues=[bad_payload])
    assert resp.status_code == 422


def test_mapped_cues_randomized_matches_bruteforce():
    rng = np.random.default_rng(2026)
    n = 48000 * 2

    def brute_force(cues_req, remove, fade):
        segs, cur = [], 0
        for a, b in remove:
            if a > cur:
                segs.append((cur, a))
            cur = b
        if cur < n:
            segs.append((cur, n))
        out_starts, prefix = [], 0
        for j, (a, b) in enumerate(segs):
            out_starts.append(prefix - j * fade)
            prefix += b - a
        result = []
        for c in cues_req:
            frags = []
            for j, (a, b) in enumerate(segs):
                lo, hi = max(c["start"], a), min(c["end"], b)
                if lo < hi:
                    frags.append({
                        "source_interval": [lo, hi],
                        "output_interval": [
                            out_starts[j] + lo - a,
                            out_starts[j] + hi - a,
                        ],
                    })
            result.append({"id": c["id"], "fragments": frags})
        return result

    for trial in range(40):
        fade_ms = int(rng.integers(0, 3))
        fade = 48 * fade_ms
        # Build well-spaced segments (>= 3000 frames) and deletions.
        n_cuts = int(rng.integers(0, 5))
        x = int(rng.integers(0, 1000))
        remove = []
        for _ in range(n_cuts):
            seg_len = int(rng.integers(3000, 8000))
            cut_len = int(rng.integers(100, 1000))
            a = x + seg_len
            remove.append([a, a + cut_len])
            x = a + cut_len
        if x >= n - 3000:
            remove = remove[:-1] if remove else []
        n_cues = int(rng.integers(0, 12))
        cues_req = []
        for k in range(n_cues):
            s = int(rng.integers(0, n))
            e = s + 1 + int(rng.integers(0, min(6000, n - s)))
            cues_req.append(cue(f"cue{k}", s, e))
        resp = post_cut(
            make_ramp(seed=trial), remove,
            crossfade_ms=fade_ms, cues=cues_req,
        )
        assert resp.status_code == 200, trial
        assert resp.json()["mapped_cues"] == brute_force(
            cues_req, remove, fade
        )
