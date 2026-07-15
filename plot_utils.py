"""Small plotting helpers shared by run scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def savefig_overwrite(fig: Any, path: str | Path, **kwargs: Any) -> Path:
    """Save a Matplotlib figure, explicitly replacing an existing file."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    fig.savefig(output, **kwargs)
    print(f"Saved {output.resolve()}")
    return output
