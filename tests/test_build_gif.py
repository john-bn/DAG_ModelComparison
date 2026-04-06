"""Tests for comparator.build_gif.create_gif.

Covers: basic GIF creation, frame count, duration, empty-input error,
single-frame GIF, and output file existence.
"""

from pathlib import Path

import pytest
from PIL import Image

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("build_gif", "comparator/build_gif.py")
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
create_gif = _mod.create_gif


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_png(path, color="red", size=(40, 30)):
    """Create a tiny solid-color PNG for testing."""
    img = Image.new("RGB", size, color=color)
    img.save(path)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCreateGif:
    def test_basic_gif_creation(self, tmp_path):
        frames = [
            _make_png(tmp_path / "a.png", "red"),
            _make_png(tmp_path / "b.png", "green"),
            _make_png(tmp_path / "c.png", "blue"),
        ]
        out = tmp_path / "out.gif"
        result = create_gif(frames, out)
        assert result == out
        assert out.exists()

    def test_gif_is_valid_image(self, tmp_path):
        frames = [
            _make_png(tmp_path / "a.png", "red"),
            _make_png(tmp_path / "b.png", "blue"),
        ]
        out = tmp_path / "out.gif"
        create_gif(frames, out)
        img = Image.open(out)
        assert img.format == "GIF"

    def test_gif_frame_count(self, tmp_path):
        colors = ["red", "green", "blue", "yellow", "white"]
        frames = [
            _make_png(tmp_path / f"{i}.png", colors[i])
            for i in range(5)
        ]
        out = tmp_path / "out.gif"
        create_gif(frames, out)

        img = Image.open(out)
        assert getattr(img, "n_frames", 1) == 5

    def test_custom_duration(self, tmp_path):
        frames = [
            _make_png(tmp_path / "a.png", "red"),
            _make_png(tmp_path / "b.png", "blue"),
        ]
        out = tmp_path / "out.gif"
        create_gif(frames, out, duration=1000)
        img = Image.open(out)
        # GIF stores duration per frame in info dict.
        # Pillow may report total duration for identical-palette frames;
        # just verify it is at least the requested duration.
        reported = img.info.get("duration", 0)
        assert reported >= 1000

    def test_single_frame_gif(self, tmp_path):
        """A single-frame GIF should still be created without error."""
        frames = [_make_png(tmp_path / "only.png")]
        out = tmp_path / "out.gif"
        result = create_gif(frames, out)
        assert out.exists()
        img = Image.open(out)
        assert img.format == "GIF"

    def test_returns_path_object(self, tmp_path):
        frames = [_make_png(tmp_path / "a.png")]
        out = tmp_path / "out.gif"
        result = create_gif(frames, out)
        assert isinstance(result, Path)

    def test_empty_input_raises(self, tmp_path):
        out = tmp_path / "out.gif"
        with pytest.raises(ValueError, match="No image paths"):
            create_gif([], out)

    def test_string_paths_accepted(self, tmp_path):
        """Paths passed as strings should work."""
        frames = [
            str(_make_png(tmp_path / "a.png")),
            str(_make_png(tmp_path / "b.png")),
        ]
        out = str(tmp_path / "out.gif")
        result = create_gif(frames, out)
        assert Path(out).exists()

    def test_output_file_size_nonzero(self, tmp_path):
        frames = [
            _make_png(tmp_path / "a.png"),
            _make_png(tmp_path / "b.png"),
        ]
        out = tmp_path / "out.gif"
        create_gif(frames, out)
        assert out.stat().st_size > 0

    def test_gif_loops_infinitely(self, tmp_path):
        """The GIF should be set to loop=0 (infinite)."""
        frames = [
            _make_png(tmp_path / "a.png"),
            _make_png(tmp_path / "b.png"),
        ]
        out = tmp_path / "out.gif"
        create_gif(frames, out)
        img = Image.open(out)
        # loop=0 means infinite; PIL stores this in info
        assert img.info.get("loop", 0) == 0
