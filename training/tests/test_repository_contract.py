from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_public_runtime_has_no_legacy_workspace_dependency():
    forbidden = ("temporal_zero_shot_fault_diagnosis", "/home/smai/")
    preflight = (ROOT / "scripts/preflight.py").resolve()
    files = list((ROOT / "src").rglob("*.py")) + [
        path for path in (ROOT / "scripts").rglob("*.py") if path.resolve() != preflight
    ]
    assert files
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert not any(token in text for token in forbidden), path


def test_results_are_not_imported_by_runtime():
    for path in list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").rglob("*.py")):
        assert "results/hust" not in path.read_text(encoding="utf-8")
