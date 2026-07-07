"""
Generates FAKE, WATERMARKED test cards. Output must never be usable as a real identity document.

CLI: batch-generate labeled synthetic fixtures into test_data/id_cards/ (gitignored).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.synthetic.content import generate_content  # noqa: E402
from tests.synthetic.degrade import apply_degradations  # noqa: E402
from tests.synthetic.ground_truth import content_to_ground_truth  # noqa: E402
from tests.synthetic.layout import LAYOUTS  # noqa: E402
from tests.synthetic.render import render_back, render_front  # noqa: E402


def generate_one(
    out_dir: Path,
    *,
    rng: random.Random | None = None,
    tags: list[str] | None = None,
    layout_name: str = "standard",
    with_back: bool = True,
    multiline_address: bool | None = None,
    stem: str | None = None,
) -> tuple[Path, Path | None, dict]:
    """
    Write front image + JSON (+ optional back). Returns (front_path, back_path|None, ground_truth).
    """
    rng = rng or random.Random()
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tag_set = set(tags or [])
    layout = LAYOUTS.get(layout_name, LAYOUTS["standard"])
    if layout_name == "new":
        tag_set.add("new_layout")
    elif layout_name in ("standard", "old"):
        tag_set.add("old_layout")

    content = generate_content(
        rng,
        multiline_address=multiline_address if multiline_address is not None else ("multiline_address" in tag_set),
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    stem = stem or f"synthetic_{ts}_{rng.randint(1000, 9999)}"
    front_path = out_dir / f"{stem}.jpg"
    back_path = out_dir / f"{stem}_back.jpg" if with_back else None

    front = render_front(content, layout)
    front = apply_degradations(front, tag_set)
    front.save(front_path, quality=92)

    back_image_name = ""
    if with_back and back_path:
        back = render_back(content, layout)
        back = apply_degradations(back, tag_set)
        back.save(back_path, quality=92)
        back_image_name = back_path.name

    gt = content_to_ground_truth(content, extra_tags=sorted(tag_set), back_image=back_image_name)
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(gt, ensure_ascii=False, indent=2), encoding="utf-8")

    return front_path, back_path, gt


def generate_batch(
    count: int,
    out_dir: Path,
    *,
    tags: list[str] | None = None,
    layout_name: str = "standard",
    with_back: bool = True,
    seed: int | None = None,
) -> list[tuple[Path, Path | None, dict]]:
    rng = random.Random(seed)
    results = []
    tag_pool = tags or []
    for i in range(count):
        case_tags = list(tag_pool)
        # vary degradations across batch when multiple edge tags requested
        if "rotated" in case_tags and i % 2:
            case_tags.append("skewed")
        if not case_tags and i % 3 == 0:
            case_tags = ["blurry"]
        if i % 4 == 0:
            case_tags = list(set(case_tags) | {"compound_name"})
        stem = f"synthetic_{seed or 0}_{i:03d}"
        results.append(
            generate_one(
                out_dir,
                rng=rng,
                tags=case_tags,
                layout_name=layout_name if i % 2 == 0 else ("new" if layout_name == "standard" else layout_name),
                with_back=with_back,
                stem=stem,
            )
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate FAKE watermarked Egyptian ID test fixtures (not real documents).",
    )
    parser.add_argument("--count", type=int, default=1, help="Number of cards to generate")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "test_data" / "id_cards",
        help="Output directory (default: test_data/id_cards — gitignored)",
    )
    parser.add_argument("--tags", default="", help="Comma-separated edge tags: rotated,blurry,glare,...")
    parser.add_argument("--layout", default="standard", choices=("standard", "old", "new"))
    parser.add_argument("--no-back", action="store_true", help="Front only")
    parser.add_argument("--multiline", action="store_true", help="Force multiline address")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    if args.multiline and "multiline_address" not in tags:
        tags.append("multiline_address")

    if args.count == 1:
        front, back, gt = generate_one(
            args.out_dir,
            tags=tags,
            layout_name=args.layout,
            with_back=not args.no_back,
            multiline_address=True if args.multiline else None,
            rng=random.Random(args.seed),
        )
        print(f"Wrote {front}")
        if back:
            print(f"Wrote {back}")
        print(f"NID (synthetic): {gt['national_id']}  tags={gt['tags']}")
    else:
        rows = generate_batch(
            args.count,
            args.out_dir,
            tags=tags,
            layout_name=args.layout,
            with_back=not args.no_back,
            seed=args.seed,
        )
        print(f"Generated {len(rows)} synthetic fixture(s) in {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
