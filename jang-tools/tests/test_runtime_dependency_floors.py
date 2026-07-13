from __future__ import annotations

from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_mlx_runtime_dependency_floors_match_latest_published_stack():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    extras = pyproject["project"]["optional-dependencies"]

    assert "mlx>=0.31.2" in extras["mlx"]
    assert "mlx-lm>=0.31.3" in extras["mlx"]
    assert "mlx>=0.31.2" in extras["vlm"]
    assert "mlx-lm>=0.31.3" in extras["vlm"]
    assert "mlx-vlm>=0.6.3" in extras["vlm"]


def test_jang_studio_bundle_installs_latest_vlm_runtime_floor():
    script = (ROOT.parent / "JANGStudio/Scripts/build-python-bundle.sh").read_text()

    assert '"mlx-vlm>=0.6.3"' in script
    assert "mlx-vlm>=0.1" not in script
