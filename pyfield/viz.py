"""Notebook-friendly visualisation of scan structures.

`animate_xyz_dir(path)` reads every `.xyz` in a directory, builds a
matplotlib `FuncAnimation` of the atoms (3D scatter, coloured by
element), and returns an HTML-embed object that displays inline in
Jupyter. Use it after `pyfield make-scan` to confirm perturbations look
the way you intended before paying for QM:

    from pyfield.viz import animate_xyz_dir
    animate_xyz_dir('runs/scan_structures/Cl2_d')      # all frames matching Cl2_d_*.xyz

Falls back gracefully when ffmpeg/HTML is not the right output target —
returns the `FuncAnimation` object so callers can call `.save()` etc.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np


# Minimal element → CPK colour table. Anything not listed falls back to grey.
_CPK = {
    "H":  "#FFFFFF", "He": "#D9FFFF",
    "Li": "#CC80FF", "Be": "#C2FF00", "B": "#FFB5B5",
    "C":  "#909090", "N":  "#3050F8", "O": "#FF0D0D",
    "F":  "#90E050", "Ne": "#B3E3F5",
    "Na": "#AB5CF2", "Mg": "#8AFF00", "Al": "#BFA6A6",
    "Si": "#F0C8A0", "P":  "#FF8000", "S":  "#FFFF30",
    "Cl": "#1FF01F", "Ar": "#80D1E3",
    "K":  "#8F40D4", "Ca": "#3DFF00",
    "Br": "#A62929", "I":  "#940094",
    "Fe": "#E06633", "Cu": "#C88033", "Zn": "#7D80B0",
}
_DEFAULT_COLOR = "#888888"

# Crude radii in Å (just for scatter sizing — not chemical truth).
_RADII = {
    "H": 0.31, "He": 0.28, "Li": 1.28, "Be": 0.96, "B": 0.84,
    "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "Ne": 0.58,
    "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "Ar": 1.06,
}


def _parse_xyz(path: Path) -> Tuple[List[str], np.ndarray, str]:
    text = path.read_text().splitlines()
    if len(text) < 2:
        raise ValueError(f"{path}: too short to be xyz")
    n = int(text[0].strip())
    comment = text[1]
    elements: List[str] = []
    coords: List[List[float]] = []
    for line in text[2:2 + n]:
        parts = line.split()
        elements.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return elements, np.asarray(coords, dtype=float), comment


def _index_in_name(p: Path) -> int:
    """Sort xyz files by trailing _N if present, else lexicographic."""
    m = re.search(r"_(\d+)\.xyz$", p.name)
    return int(m.group(1)) if m else 1_000_000


def _gather_frames(directory: Path, pattern: str) -> List[Path]:
    files = sorted(directory.glob(pattern), key=_index_in_name)
    if not files:
        raise FileNotFoundError(
            f"{directory}: no files matching {pattern!r} found"
        )
    return files


def animate_xyz_dir(
    directory,
    *,
    pattern: str = "*.xyz",
    title: Optional[str] = None,
    interval_ms: int = 250,
    figsize: Tuple[float, float] = (5.5, 5.0),
    size_scale: float = 600.0,
):
    """Build a 3D scatter animation over every xyz frame in `directory`.

    Returns an `IPython.display.HTML` object when running in a notebook
    (so the bare expression renders inline) plus the underlying
    `matplotlib.animation.FuncAnimation` reachable via `.anim` for
    callers that want to `.save(...)` it.
    """
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

    directory = Path(directory)
    paths = _gather_frames(directory, pattern)

    frames = [_parse_xyz(p) for p in paths]
    all_coords = np.concatenate([f[1] for f in frames], axis=0)
    lo = all_coords.min(axis=0)
    hi = all_coords.max(axis=0)
    pad = max(0.5, 0.1 * float(np.linalg.norm(hi - lo)))
    lo -= pad
    hi += pad

    elements_first = frames[0][0]
    colors = [_CPK.get(e, _DEFAULT_COLOR) for e in elements_first]
    sizes = [size_scale * _RADII.get(e, 0.7) for e in elements_first]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        frames[0][1][:, 0], frames[0][1][:, 1], frames[0][1][:, 2],
        c=colors, s=sizes, edgecolors="black", linewidths=0.5, depthshade=True,
    )
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_xlabel("x [Å]"); ax.set_ylabel("y [Å]"); ax.set_zlabel("z [Å]")
    title_artist = ax.set_title(title or _frame_title(paths[0], frames[0][2]))

    def _update(idx):
        elements, coords, comment = frames[idx]
        scatter._offsets3d = (coords[:, 0], coords[:, 1], coords[:, 2])
        title_artist.set_text(title or _frame_title(paths[idx], comment))
        return scatter, title_artist

    anim = animation.FuncAnimation(
        fig, _update, frames=len(frames), interval=interval_ms, blit=False,
    )

    # Suppress the static figure that Jupyter would otherwise render
    # alongside the animation HTML.
    plt.close(fig)

    try:
        from IPython.display import HTML
        html = HTML(anim.to_jshtml())
        html.anim = anim       # keep a ref so the animation is not garbage-collected
        return html
    except Exception:
        return anim


def _frame_title(path: Path, comment: str) -> str:
    return f"{path.name}   ({comment})" if comment else path.name
