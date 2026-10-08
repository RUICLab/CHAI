import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("av")
pytest.importorskip("torchaudio")

from biomedical.coughvid_pilot import TinyCough, audio_spectrogram, make_transform
from biomedical.pcam_pilot import TinyPcam, estimate_macs, resize_batch


def test_pcam_resolution_levels_preserve_output_shape_and_reduce_work():
    model = TinyPcam().eval()
    images = torch.rand(2, 3, 96, 96)
    macs = []
    for resolution in (64, 48, 32, 24, 16):
        resized = resize_batch(images, resolution)
        assert resized.shape == (2, 3, resolution, resolution)
        assert model(resized).shape == (2,)
        macs.append(estimate_macs(resolution))
    assert all(left > right for left, right in zip(macs, macs[1:]))


def test_coughvid_hop_knob_preserves_feature_width_and_reduces_frames():
    waveform = torch.randn(16_000 * 3)
    frame_counts = []
    for multiplier in (1, 2, 4, 8):
        transform = make_transform(multiplier)
        features = audio_spectrogram(waveform, transform)
        assert features.shape[0] == 40
        assert TinyCough()(features[None, None, ...]).shape == (1,)
        frame_counts.append(features.shape[-1])
    assert all(left > right for left, right in zip(frame_counts, frame_counts[1:]))
