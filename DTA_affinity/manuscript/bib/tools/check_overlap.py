#!/usr/bin/env python3
"""Figure layout hygiene — overlap / occupancy gate (v1).

Two modes:

1. Matplotlib artists (preferred when you have a live Figure):
   collect Text / Annotation / Legend window extents and report intersections.

2. PNG occupancy (post-export gate):
   build an ink mask, find label-sized connected components, report overlapping
   component bboxes. Optional ``--overlay`` draws collision boxes.

Exit codes: 0 = clean (or only warnings under --soft), 1 = collisions found,
2 = usage / I/O error.

Examples
--------
  python check_overlap.py fig.png
  python check_overlap.py fig.png --overlay fig_overlap.png
  python check_overlap.py --self-test

In a plotting script (before savefig)::

  from pathlib import Path
  import sys
  sys.path.insert(0, str(Path(__file__).resolve().parents[N] / ".cursor/skills/figure-layout-hygiene/scripts"))
  from check_overlap import assert_figure_clean
  assert_figure_clean(fig)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BBox:
    """Inclusive-exclusive pixel / display box: x0, y0, x1, y1."""

    x0: float
    y0: float
    x1: float
    y1: float
    label: str = ""

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        return self.width * self.height

    def intersects(self, other: "BBox", pad: float = 0.0) -> bool:
        return not (
            self.x1 + pad <= other.x0
            or other.x1 + pad <= self.x0
            or self.y1 + pad <= other.y0
            or other.y1 + pad <= self.y0
        )

    def intersection_area(self, other: "BBox") -> float:
        ix0 = max(self.x0, other.x0)
        iy0 = max(self.y0, other.y0)
        ix1 = min(self.x1, other.x1)
        iy1 = min(self.y1, other.y1)
        return max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)

    def iou(self, other: "BBox") -> float:
        inter = self.intersection_area(other)
        if inter <= 0:
            return 0.0
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


@dataclass
class Collision:
    a: str
    b: str
    area: float
    iou: float
    kind: str  # "artist" | "png-component"


@dataclass
class Report:
    source: str
    n_boxes: int
    collisions: List[Collision]
    notes: List[str]

    @property
    def ok(self) -> bool:
        return len(self.collisions) == 0

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "n_boxes": self.n_boxes,
            "ok": self.ok,
            "collisions": [asdict(c) for c in self.collisions],
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def pairwise_collisions(
    boxes: Sequence[BBox],
    *,
    min_area: float = 4.0,
    min_iou: float = 0.0,
    kind: str = "artist",
    skip_containment: bool = False,
    containment_frac: float = 0.85,
) -> List[Collision]:
    hits: List[Collision] = []
    n = len(boxes)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = boxes[i], boxes[j]
            if a.area < 1 or b.area < 1:
                continue
            inter = a.intersection_area(b)
            if inter < min_area:
                continue
            # Nested boxes (small glyph inside a large panel blob) ≠ label fight
            if skip_containment:
                smaller = min(a.area, b.area)
                if smaller > 0 and inter / smaller >= containment_frac:
                    continue
            iou = a.iou(b)
            if iou < min_iou:
                continue
            hits.append(
                Collision(
                    a=a.label or f"box[{i}]",
                    b=b.label or f"box[{j}]",
                    area=float(inter),
                    iou=float(iou),
                    kind=kind,
                )
            )
    return hits


# ---------------------------------------------------------------------------
# Matplotlib artist mode
# ---------------------------------------------------------------------------


def _mpl_renderer(fig):
    fig.canvas.draw()
    return fig.canvas.get_renderer()


def collect_figure_text_bboxes(
    fig,
    *,
    min_chars: int = 1,
    include_ticks: bool = False,
) -> List[BBox]:
    """Collect display-coordinate bboxes for visible Text-like artists."""
    from matplotlib.legend import Legend
    from matplotlib.text import Annotation, Text

    renderer = _mpl_renderer(fig)
    boxes: List[BBox] = []

    def _add_text(t: Text, prefix: str) -> None:
        raw = (t.get_text() or "").strip()
        if len(raw) < min_chars:
            return
        if not t.get_visible():
            return
        try:
            ext = t.get_window_extent(renderer=renderer)
        except Exception:
            return
        if ext.width < 1 or ext.height < 1:
            return
        label = f"{prefix}:{raw[:40]!r}"
        boxes.append(BBox(ext.x0, ext.y0, ext.x1, ext.y1, label=label))

    for i, ax in enumerate(fig.axes):
        for j, t in enumerate(ax.texts):
            _add_text(t, f"ax{i}.text{j}")
        title = ax.title
        if title is not None:
            _add_text(title, f"ax{i}.title")
        _add_text(ax.xaxis.label, f"ax{i}.xlabel")
        _add_text(ax.yaxis.label, f"ax{i}.ylabel")
        if include_ticks:
            for t in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
                _add_text(t, f"ax{i}.tick")
        leg = ax.get_legend()
        if isinstance(leg, Legend) and leg.get_visible():
            try:
                ext = leg.get_window_extent(renderer)
                boxes.append(
                    BBox(
                        ext.x0,
                        ext.y0,
                        ext.x1,
                        ext.y1,
                        label=f"ax{i}.legend",
                    )
                )
            except Exception:
                pass
            for t in leg.get_texts():
                _add_text(t, f"ax{i}.legtxt")

    for j, t in enumerate(getattr(fig, "texts", [])):
        _add_text(t, f"fig.text{j}")

    _ = Annotation
    return boxes


def check_figure(
    fig,
    *,
    min_area: float = 8.0,
    min_iou: float = 0.02,
    include_ticks: bool = False,
) -> Report:
    boxes = collect_figure_text_bboxes(fig, include_ticks=include_ticks)
    hits = pairwise_collisions(
        boxes, min_area=min_area, min_iou=min_iou, kind="artist"
    )
    notes = [
        "Artist mode checks Text / title / axis labels / legend extents "
        f"(include_ticks={include_ticks}).",
        "Heatmap cell ink vs labels is not covered here — use PNG mode after export.",
    ]
    return Report(
        source="matplotlib.Figure",
        n_boxes=len(boxes),
        collisions=hits,
        notes=notes,
    )


def assert_figure_clean(fig, **kwargs) -> Report:
    report = check_figure(fig, **kwargs)
    if not report.ok:
        detail = "; ".join(
            f"{c.a} × {c.b} (area={c.area:.1f}, iou={c.iou:.3f})"
            for c in report.collisions[:8]
        )
        raise AssertionError(
            f"figure-layout-hygiene: {len(report.collisions)} text collision(s): {detail}"
        )
    return report


# ---------------------------------------------------------------------------
# PNG occupancy mode
# ---------------------------------------------------------------------------


def _load_rgb(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit(
            "PNG mode requires Pillow (pip install Pillow). "
            f"Import error: {exc}"
        ) from exc
    img = Image.open(path).convert("RGB")
    return np.asarray(img)


def ink_mask(
    rgb: np.ndarray,
    *,
    bg_threshold: int = 250,
    dark_bg: bool = False,
) -> np.ndarray:
    """True where pixels count as ink (occupied)."""
    if dark_bg:
        # imaging plates: treat near-black as background
        return np.any(rgb > (255 - bg_threshold), axis=-1)
    # white / near-white canvas
    return np.any(rgb < bg_threshold, axis=-1)


def _label_components(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """Connected components; prefer scipy, else downsampled Union-Find."""
    try:
        from scipy import ndimage  # type: ignore

        labeled, n = ndimage.label(mask)
        return labeled.astype(np.int32), int(n)
    except Exception:
        # Pure-Python UF is slow on full-res plates — downsample for structure.
        step = 2 if max(mask.shape) > 1200 else 1
        small = mask[::step, ::step]
        lab_s, n = _label_components_numpy(small)
        if step == 1:
            return lab_s, n
        # nearest upsample back
        labeled = np.repeat(np.repeat(lab_s, step, axis=0), step, axis=1)
        labeled = labeled[: mask.shape[0], : mask.shape[1]]
        return labeled, n


def _label_components_numpy(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """Two-pass Union-Find labeling."""
    h, w = mask.shape
    parent = np.arange(h * w + 1, dtype=np.int32)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    labels = np.zeros((h, w), dtype=np.int32)
    next_label = 0
    for y in range(h):
        row = mask[y]
        prev = labels[y - 1] if y else None
        for x in range(w):
            if not row[x]:
                continue
            left = labels[y, x - 1] if x else 0
            up = int(prev[x]) if prev is not None else 0
            if left and up:
                labels[y, x] = left
                union(left, up)
            elif left:
                labels[y, x] = left
            elif up:
                labels[y, x] = up
            else:
                next_label += 1
                labels[y, x] = next_label

    remap: dict[int, int] = {}
    dense = np.zeros_like(labels)
    n = 0
    flat = labels.ravel()
    out = dense.ravel()
    for i, lab in enumerate(flat):
        if not lab:
            continue
        root = find(int(lab))
        rid = remap.get(root)
        if rid is None:
            n += 1
            remap[root] = n
            rid = n
        out[i] = rid
    return dense, n


def component_bboxes(
    labeled: np.ndarray,
    n: int,
    *,
    min_area: int,
    max_area: int,
) -> List[BBox]:
    boxes: List[BBox] = []
    for idx in range(1, n + 1):
        ys, xs = np.where(labeled == idx)
        if ys.size == 0:
            continue
        area = int(ys.size)
        if area < min_area or area > max_area:
            continue
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        # skip full-frame almost (axes spines spanning whole fig)
        h, w = labeled.shape
        if (x1 - x0) > 0.85 * w and (y1 - y0) > 0.85 * h:
            continue
        boxes.append(BBox(x0, y0, x1, y1, label=f"cc{idx}:a{area}"))
    return boxes


def check_png(
    path: Path | str,
    *,
    bg_threshold: int = 250,
    dark_bg: bool = False,
    min_component_area: int = 40,
    max_component_area: Optional[int] = None,
    min_overlap_area: float = 12.0,
    min_iou: float = 0.08,
) -> Tuple[Report, np.ndarray, List[BBox]]:
    path = Path(path)
    rgb = _load_rgb(path)
    h, w = rgb.shape[:2]
    if max_component_area is None:
        # Keep label-/glyph-sized blobs only (exclude heatmaps, big fills).
        max_component_area = min(6000, max(800, int(0.0015 * h * w)))

    mask = ink_mask(rgb, bg_threshold=bg_threshold, dark_bg=dark_bg)
    labeled, n = _label_components(mask)
    boxes = component_bboxes(
        labeled,
        n,
        min_area=min_component_area,
        max_area=max_component_area,
    )
    # Drop extreme aspect-ratio strips (spines / rules)
    filtered: List[BBox] = []
    for b in boxes:
        ar = b.width / max(b.height, 1.0)
        if ar > 25 or ar < 1 / 25:
            continue
        filtered.append(b)
    boxes = filtered

    hits = pairwise_collisions(
        boxes,
        min_area=min_overlap_area,
        min_iou=min_iou,
        kind="png-component",
        skip_containment=True,
    )
    notes = [
        f"PNG {w}x{h}, ink_fraction={float(mask.mean()):.3f}, "
        f"components_kept={len(boxes)}/{n}, "
        f"max_component_area={max_component_area}.",
        "PNG mode is heuristic (label-sized components + bbox IoU).",
        "Prefer artist mode inside the plotting script when possible.",
    ]
    report = Report(
        source=str(path),
        n_boxes=len(boxes),
        collisions=hits,
        notes=notes,
    )
    return report, rgb, boxes


def write_overlay(
    rgb: np.ndarray,
    boxes: Sequence[BBox],
    collisions: Sequence[Collision],
    out_path: Path,
) -> None:
    from PIL import Image, ImageDraw

    img = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(img)
    hit_labels = set()
    for c in collisions:
        hit_labels.add(c.a)
        hit_labels.add(c.b)
    for b in boxes:
        color = (220, 40, 40) if b.label in hit_labels else (40, 140, 220)
        width = 3 if b.label in hit_labels else 1
        draw.rectangle([b.x0, b.y0, b.x1 - 1, b.y1 - 1], outline=color, width=width)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def occupancy_stats(path: Path | str, **kwargs) -> dict:
    """Return a compact occupancy summary (for agents / CI logs)."""
    report, rgb, boxes = check_png(path, **kwargs)
    h, w = rgb.shape[:2]
    mask = ink_mask(rgb, bg_threshold=kwargs.get("bg_threshold", 250),
                    dark_bg=kwargs.get("dark_bg", False))
    return {
        **report.to_dict(),
        "width": w,
        "height": h,
        "ink_fraction": float(mask.mean()),
        "boxes": [asdict(b) for b in boxes[:50]],
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _self_test() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Case A: overlapping titles / notes → must fail artist check
    fig1, ax1 = plt.subplots(figsize=(4, 3))
    ax1.plot([0, 1], [0, 1])
    ax1.set_title("Primary title that is quite long", fontsize=12)
    ax1.text(0.5, 1.02, "a  colliding tag", transform=ax1.transAxes, fontsize=14)
    ax1.text(0.02, 0.95, "n=12 inset", transform=ax1.transAxes, fontsize=11)
    ax1.text(0.02, 0.92, "n=12 ghost", transform=ax1.transAxes, fontsize=11)
    r1 = check_figure(fig1, min_area=1.0, min_iou=0.0)
    plt.close(fig1)

    # Case B: clean separated tags → should pass (or few collisions)
    fig2, axes = plt.subplots(1, 2, figsize=(6, 2.5))
    for i, ax in enumerate(axes):
        ax.plot([0, 1], [0, 1])
        ax.text(
            -0.12,
            1.05,
            "ab"[i],
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            clip_on=False,
        )
        ax.set_title(["Left panel", "Right panel"][i], fontsize=9, pad=12)
    r2 = check_figure(fig2, min_area=8.0, min_iou=0.05)
    # export clean-ish png for png mode smoke
    tmp = Path("/tmp/figure_layout_hygiene_selftest.png")
    fig2.savefig(tmp, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig2)

    r3, rgb, boxes = check_png(tmp, min_component_area=20, min_iou=0.08)

    print("self-test artist overlapping:", "FAIL" if not r1.ok else "UNEXPECTED PASS")
    print(f"  collisions={len(r1.collisions)} boxes={r1.n_boxes}")
    print(
        "self-test artist clean-ish:",
        "PASS" if r2.ok else f"hits={len(r2.collisions)}",
    )
    if not r2.ok:
        for c in r2.collisions[:6]:
            print(f"    {c.a} × {c.b}")
    print(
        "self-test png:",
        "PASS" if r3.ok else f"hits={len(r3.collisions)} boxes={r3.n_boxes}",
    )

    if r1.ok:
        print("ERROR: expected collisions in overlapping artist fixture", file=sys.stderr)
        return 1
    if not r2.ok:
        print("ERROR: clean fixture should pass artist check", file=sys.stderr)
        return 1
    print("self-test OK")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_report(report: Report, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        return
    status = "CLEAN" if report.ok else "COLLISIONS"
    print(f"[{status}] {report.source}  boxes={report.n_boxes}  hits={len(report.collisions)}")
    for n in report.notes:
        print(f"  note: {n}")
    for i, c in enumerate(report.collisions, 1):
        print(
            f"  {i}. [{c.kind}] {c.a}  ×  {c.b}  "
            f"overlap_area={c.area:.1f}  iou={c.iou:.3f}"
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="figure-layout-hygiene overlap / occupancy checker (v1)"
    )
    p.add_argument("png", nargs="?", help="PNG preview at final journal width")
    p.add_argument("--overlay", type=Path, help="Write bbox overlay PNG")
    p.add_argument("--json", action="store_true", help="Machine-readable report")
    p.add_argument("--bg-threshold", type=int, default=250)
    p.add_argument(
        "--dark-bg",
        action="store_true",
        help="Treat near-black as background (imaging plates)",
    )
    p.add_argument("--min-area", type=int, default=40, help="Min component area (px)")
    p.add_argument(
        "--max-area",
        type=int,
        default=None,
        help="Max component area (px); default ~4%% of image",
    )
    p.add_argument("--min-iou", type=float, default=0.1)
    p.add_argument(
        "--soft",
        action="store_true",
        help="Always exit 0 (still print collisions)",
    )
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args(argv)

    if args.self_test:
        return _self_test()

    if not args.png:
        p.print_help()
        return 2

    path = Path(args.png)
    if not path.is_file():
        print(f"error: file not found: {path}", file=sys.stderr)
        return 2

    report, rgb, boxes = check_png(
        path,
        bg_threshold=args.bg_threshold,
        dark_bg=args.dark_bg,
        min_component_area=args.min_area,
        max_component_area=args.max_area,
        min_iou=args.min_iou,
    )
    _print_report(report, as_json=args.json)

    if args.overlay:
        write_overlay(rgb, boxes, report.collisions, args.overlay)
        if not args.json:
            print(f"  overlay: {args.overlay}")

    if report.ok or args.soft:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
