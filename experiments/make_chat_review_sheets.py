"""Diagnostic contact sheets of immutable CHAT crops; never training input."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/paddle_vl15_poker_sft_20260930"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--progress", type=Path)
    ap.add_argument("--exclude-progress", type=Path)
    ap.add_argument("--unique", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=15)
    args = ap.parse_args()
    rows = [json.loads(s) for s in (OUT / "candidates_manifest.jsonl").read_text(encoding="utf-8").splitlines() if s]
    decisions = json.loads(args.progress.read_text(encoding="utf-8"))["decisions"] if args.progress else {}
    if args.progress:
        rows = [r for r in rows if r["candidate_id"] in decisions]
    else:
        rows = [r for r in rows if r["production_raw_geometry"]]
    if args.exclude_progress:
        previous = json.loads(args.exclude_progress.read_text(encoding="utf-8"))["decisions"]
        rows = [r for r in rows if r["candidate_id"] not in previous]
    if args.unique:
        seen = set()
        selected = []
        for row in rows:
            key = row["image_sha256"]
            if key not in seen:
                selected.append(row)
                seen.add(key)
        rows = selected
    if args.limit:
        rows = rows[: args.limit]
    folder = OUT / ("sheets_user" if args.progress else "sheets_candidates")
    folder.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 18)
    index_rows = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        sheet = Image.new("RGB", (1620, 112 * len(batch)), "#16202d")
        draw = ImageDraw.Draw(sheet)
        for j, row in enumerate(batch):
            k, y = start + j + 1, j * 112
            decision = decisions.get(row["candidate_id"], {})
            with Image.open(row["artifact_path"]) as im:
                # Display-only transform requested for card glyph inspection.
                crop = im.convert("RGB").crop((0, 0, im.width - 12, im.height))
                crop = crop.resize((crop.width * 4, crop.height * 4), Image.Resampling.NEAREST)
            sheet.paste(crop, (8, y + 37))
            label = f"{k:03} {row['candidate_id']}  {row['source_video']} f{row['frame_id']} t{row['table_id']} {row.get('event_type')}"
            draw.text((8, y + 5), label, font=font, fill="white")
            draw.text((960, y + 40), f"USER: {decision.get('verified_gt', '')} {decision.get('kind', '')} {decision.get('status', '')}", font=font, fill="#a6ebba")
            draw.text((960, y + 67), f"OCR: {row.get('proposed_gt') or ''}", font=font, fill="#ffcd75")
            index_rows.append({"number": k, "candidate_id": row["candidate_id"], "sheet": f"{start//args.batch_size+1:02}.png", "decision": decision})
        sheet.save(folder / f"{start//args.batch_size+1:02}.png")
    (folder / "index.json").write_text(json.dumps(index_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"folder": str(folder), "rows": len(rows), "sheets": (len(rows) + args.batch_size - 1)//args.batch_size}))


if __name__ == "__main__":
    main()
