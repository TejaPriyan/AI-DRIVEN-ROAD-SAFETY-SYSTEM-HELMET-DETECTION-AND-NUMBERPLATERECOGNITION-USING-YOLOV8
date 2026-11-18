import ast
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_data_yaml_schema_matches_readme():
    data_yaml = REPO_ROOT / "data.yaml"
    assert data_yaml.is_file()
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    assert cfg["nc"] == 4
    assert cfg["train"] == "data/images/train"
    assert cfg["val"] == "data/images/val"
    assert cfg["names"] == {
        0: "Helmet",
        1: "No-Helmet",
        2: "Rider",
        3: "Number-Plate",
    }


def test_license_and_packaging_metadata_consistent():
    license_text = (REPO_ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "GNU AFFERO GENERAL PUBLIC LICENSE" in license_text

    readme_text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "AGPL-3.0" in readme_text or "AGPL_v3" in readme_text
    assert "License-MIT" not in readme_text
    assert "MIT License" not in readme_text

    pyproject_text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "YOLOv5" not in pyproject_text
    assert "Glenn Jocher" not in pyproject_text
    assert "easyocr" in pyproject_text
    assert "AGPL-3.0" in pyproject_text

    if sys.version_info >= (3, 11):
        import tomllib

        parsed = tomllib.loads(pyproject_text)
        assert parsed["project"]["name"] == "road-safety-yolov8"
        assert parsed["project"]["license"]["text"] == "AGPL-3.0"
        assert any(dep.startswith("easyocr") for dep in parsed["project"]["dependencies"])

    citation_cfg = yaml.safe_load((REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert citation_cfg["license"] == "AGPL-3.0"
    assert "Jocher" not in str(citation_cfg)

    assert not (REPO_ROOT / "tutorial.ipynb").exists()
    assert not (REPO_ROOT / "google87bb3bc53ec346d2.html").exists()


def test_notebooks_are_stripped_and_syntactically_clean():
    notebooks = [
        REPO_ROOT / "Object Detection with COCO Dataset.ipynb",
        REPO_ROOT / "Object Detection with Custom Dataset.ipynb",
    ]
    for nb_path in notebooks:
        assert nb_path.is_file()
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        for cell in nb["cells"]:
            if cell["cell_type"] != "code":
                continue
            assert cell.get("outputs", []) == []
            assert cell.get("execution_count") is None
            src = "".join(cell.get("source", []))
            assert "yolo5nu.pt" not in src
            assert "opencv-python-headless" not in src
            assert "C:\\Program Files\\Tesseract-OCR" not in src
            py_lines = [
                line
                for line in src.splitlines()
                if not line.lstrip().startswith(("!", "%"))
            ]
            ast.parse("\n".join(py_lines))


def test_detect_has_no_unused_legacy_imports():
    tree = ast.parse((REPO_ROOT / "detect.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert "sys" not in imported
    assert "pathlib" not in imported
    assert "numpy" not in imported
