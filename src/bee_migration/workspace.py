from pathlib import Path


class Workspace:
    """Where a project keeps its data, results and images."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def data(self, name: str) -> Path:
        return self._writable(self._root / "data" / name)

    def result(self, name: str) -> Path:
        return self._writable(self._root / "results" / name)

    def images(self) -> Path:
        return self._root / "images"

    def _writable(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
