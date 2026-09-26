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


def post_cut(pcm, intervals, crossfade_ms=0, channels=1, rate=RATE):
    return client.post(
        "/v1/cut-splice",
        json={
            "wav_base64": wav_b64(pcm, channels, rate),
            "remove_intervals": intervals,
            "crossfade_ms": crossfade_ms,
        },
    )


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
