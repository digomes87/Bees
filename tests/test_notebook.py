import json
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).parent.parent / "notebooks" / "bee_migration_tutorial.ipynb"


@pytest.fixture(scope="module")
def cells() -> list[dict]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def test_tutorial_runs_from_top_to_bottom(cells):
    namespace: dict = {}
    sources = ["".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"]

    for source in sources:
        exec(compile(source, NOTEBOOK.name, "exec"), namespace)  # noqa: S102

    assert namespace["best"].name in namespace["leaderboard"].metrics().index
    assert len(namespace["observations"]) == 120


def test_tutorial_is_committed_without_outputs(cells):
    outputs = [cell["outputs"] for cell in cells if cell["cell_type"] == "code"]

    assert all(output == [] for output in outputs)
