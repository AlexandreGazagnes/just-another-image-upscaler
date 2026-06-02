"""
Tests for upscale.py (local inference CLI).
Torch and inference module calls are mocked — no GPU or model weights required.
"""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import upscale


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def jpg(tmp_path) -> Path:
    p = tmp_path / "photo.jpg"
    p.write_bytes(b"\xff\xd8\xff")
    return p


@pytest.fixture()
def png(tmp_path) -> Path:
    p = tmp_path / "photo.png"
    p.write_bytes(b"\x89PNG\r\n")
    return p


@pytest.fixture()
def mock_torch():
    mod = types.ModuleType("torch")
    mod.device = lambda x: x
    cuda = types.ModuleType("torch.cuda")
    cuda.is_available = MagicMock(return_value=False)
    mod.cuda = cuda
    return mod


@pytest.fixture()
def mock_inference():
    mod = types.ModuleType("inference")
    mod.load_model = MagicMock(return_value=MagicMock())
    mod.upscale_image = MagicMock()
    return mod


# ---------------------------------------------------------------------------
# Output path logic
# ---------------------------------------------------------------------------

class TestOutputPathDefault:
    def test_defaults_to_upscaled_suffix(self, jpg):
        expected = jpg.parent / "photo_upscaled.jpg"
        with patch("upscale.run_upscale") as mock_run:
            sys.argv = ["upscale.py", str(jpg)]
            upscale.main()
        mock_run.assert_called_once()
        _, out, *_ = mock_run.call_args.args
        assert out == expected

    def test_explicit_output_is_used_verbatim(self, jpg, tmp_path):
        custom = tmp_path / "out" / "result.jpg"
        with patch("upscale.run_upscale") as mock_run:
            sys.argv = ["upscale.py", str(jpg), "-o", str(custom)]
            upscale.main()
        _, out, *_ = mock_run.call_args.args
        assert out == custom.resolve()


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_exits_on_missing_file(self, tmp_path):
        with pytest.raises(SystemExit):
            sys.argv = ["upscale.py", str(tmp_path / "ghost.jpg")]
            upscale.main()

    def test_exits_on_unsupported_extension(self, tmp_path):
        bad = tmp_path / "image.gif"
        bad.write_bytes(b"GIF89a")
        with patch("upscale.run_upscale"), pytest.raises(SystemExit):
            sys.argv = ["upscale.py", str(bad)]
            upscale.main()

    @pytest.mark.parametrize("ext", [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"])
    def test_accepted_formats(self, tmp_path, ext):
        img = tmp_path / f"photo{ext}"
        img.write_bytes(b"\x00" * 4)
        with patch("upscale.run_upscale"):
            sys.argv = ["upscale.py", str(img)]
            upscale.main()  # must not raise


# ---------------------------------------------------------------------------
# run_upscale — inference call construction
# ---------------------------------------------------------------------------

class TestRunUpscale:
    def _run(self, jpg, tmp_path, mock_torch, mock_inference, tile=None, tile_overlap=32):
        out = tmp_path / "out.jpg"
        with patch.dict("sys.modules", {"torch": mock_torch, "inference": mock_inference}), \
             patch("upscale.MODELS_CACHE", tmp_path / "models"):
            upscale.run_upscale(jpg, out, 4, "fp32", tile, tile_overlap)
        return mock_inference

    def test_load_model_is_called(self, jpg, tmp_path, mock_torch, mock_inference):
        self._run(jpg, tmp_path, mock_torch, mock_inference)
        mock_inference.load_model.assert_called_once()

    def test_upscale_image_is_called(self, jpg, tmp_path, mock_torch, mock_inference):
        self._run(jpg, tmp_path, mock_torch, mock_inference)
        mock_inference.upscale_image.assert_called_once()

    def test_input_path_passed_to_upscale_image(self, jpg, tmp_path, mock_torch, mock_inference):
        inf = self._run(jpg, tmp_path, mock_torch, mock_inference)
        assert str(jpg) in inf.upscale_image.call_args.args

    def test_tile_args_passed_when_specified(self, jpg, tmp_path, mock_torch, mock_inference):
        inf = self._run(jpg, tmp_path, mock_torch, mock_inference, tile=256)
        assert 256 in inf.upscale_image.call_args.args

    def test_tile_none_when_not_specified(self, jpg, tmp_path, mock_torch, mock_inference):
        inf = self._run(jpg, tmp_path, mock_torch, mock_inference)
        assert None in inf.upscale_image.call_args.args

    def test_scale_passed_to_load_model(self, jpg, tmp_path, mock_torch, mock_inference):
        inf = self._run(jpg, tmp_path, mock_torch, mock_inference)
        assert 4 in inf.load_model.call_args.args

    def test_scale_passed_to_upscale_image(self, jpg, tmp_path, mock_torch, mock_inference):
        inf = self._run(jpg, tmp_path, mock_torch, mock_inference)
        assert 4 in inf.upscale_image.call_args.args


# ---------------------------------------------------------------------------
# Real JPEG fixture — tests/fixtures/test_image.jpg
# ---------------------------------------------------------------------------

FIXTURE_JPG = Path(__file__).parent / "fixtures" / "test_image.jpg"


class TestRealJpegFixture:
    """Pipeline tests using the committed JPEG fixture (64×64, real pixels)."""

    def test_fixture_file_exists_and_is_valid_jpeg(self):
        import cv2
        assert FIXTURE_JPG.exists(), "fixtures/test_image.jpg is missing from the repo"
        img = cv2.imread(str(FIXTURE_JPG))
        assert img is not None, "cv2 could not decode the fixture JPEG"
        assert img.shape == (64, 64, 3)

    def test_fixture_accepted_as_valid_format(self):
        with patch("upscale.run_upscale"):
            sys.argv = ["upscale.py", str(FIXTURE_JPG)]
            upscale.main()

    def test_fixture_default_output_path(self):
        with patch("upscale.run_upscale") as mock_run:
            sys.argv = ["upscale.py", str(FIXTURE_JPG)]
            upscale.main()
        _, out, *_ = mock_run.call_args.args
        assert out.name == "test_image_upscaled.jpg"
        assert out.parent == FIXTURE_JPG.parent

    def test_fixture_through_run_upscale(self, tmp_path, mock_torch, mock_inference):
        out = tmp_path / "test_image_upscaled.jpg"
        with patch.dict("sys.modules", {"torch": mock_torch, "inference": mock_inference}), \
             patch("upscale.MODELS_CACHE", tmp_path / "models"):
            upscale.run_upscale(FIXTURE_JPG, out, 4, "fp32", None, 32)
        assert str(FIXTURE_JPG) in mock_inference.upscale_image.call_args.args

    def test_fixture_through_run_upscale_with_tile(self, tmp_path, mock_torch, mock_inference):
        out = tmp_path / "test_image_upscaled.jpg"
        with patch.dict("sys.modules", {"torch": mock_torch, "inference": mock_inference}), \
             patch("upscale.MODELS_CACHE", tmp_path / "models"):
            upscale.run_upscale(FIXTURE_JPG, out, 4, "fp32", tile=32, tile_overlap=16)
        assert 32 in mock_inference.upscale_image.call_args.args
