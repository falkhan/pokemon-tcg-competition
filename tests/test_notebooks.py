"""The pipeline notebook is generated (notebooks/build_pipeline.py) — pin the
committed .ipynb to the generator's output and enforce the repo laws the
notebook must not break: no auto-submit, no --workers above the ceiling of 8,
and committed-clean cells (no outputs). Hand-editing .ipynb JSON is how cells
silently break; the generator is the source of truth."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GENERATOR = ROOT / "notebooks" / "build_pipeline.py"
NOTEBOOK = ROOT / "notebooks" / "pipeline.ipynb"


def _load_generator():
    spec = importlib.util.spec_from_file_location("build_pipeline", GENERATOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cells():
    return json.loads(NOTEBOOK.read_text())["cells"]


def test_committed_notebook_matches_generator():
    assert _load_generator().build() == json.loads(NOTEBOOK.read_text()), (
        "notebooks/pipeline.ipynb drifted from its generator — edit "
        "notebooks/build_pipeline.py and re-run it, never the .ipynb")


def test_cells_are_committed_clean():
    for cell in _cells():
        assert cell["id"].startswith("pipe-")
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None


def test_notebook_never_submits():
    # The ship path is human (CLAUDE.md mandatory QC stop) — the notebook may
    # describe it but must never contain the submit command.
    for cell in _cells():
        assert "competitions submit" not in "".join(cell["source"])


def test_workers_ceiling_is_respected():
    # PARALLELISM CAP (CLAUDE.md): 12 workers deadlocks libcg's mp.Pool.
    src = "".join("".join(c["source"]) for c in _cells())
    assert "min(8, os.cpu_count()" in src          # the clamp exists
    for frag in src.split("workers")[1:]:          # no literal above the cap
        head = frag.lstrip("= \"',")[:2].rstrip("\"'], )")
        if head.isdigit():
            assert int(head) <= 8, f"workers literal {head} exceeds the cap"
