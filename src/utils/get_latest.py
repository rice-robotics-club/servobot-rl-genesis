import os
from pathlib import Path
from typing import Literal


def get_latest(
    path: str | Path,
    mode: Literal["children", "exact"] = "children",
) -> Path | None:
    path = Path(path)

    if not path.exists():
        prefix = path.name
        path = path.parent
    elif mode == "exact":
        return path
    else:
        prefix = ""

    folders = [f for f in path.iterdir() if f.exists() and f.name.find(prefix) == 0]

    if not folders:
        return None

    return max(folders, key=os.path.getmtime)
