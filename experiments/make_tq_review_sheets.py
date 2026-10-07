"""Create blind contact sheets for all Smoke16 tiles initially labeled T or Q."""
import json
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "runs/card_train_smoke16_test_20260930"
OUT = DATA / "stage2_template"
tiles = [json.loads(line) for line in (DATA / "smoke16_test_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
selected = [row for row in tiles if row["ground_truth"][0] in "TQ"]
manifest = [{"index": i, "blind_test_index": row["blind_test_index"], "card_index": row["card_index"],
             "tile_png": row["tile_png"], "tile_sha256": row["tile_sha256"]}
            for i, row in enumerate(selected)]
(OUT / "tq_review_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
for start in range(0, len(selected), 20):
    sheet = Image.new("RGB", (1080, 650), "#182638")
    draw = ImageDraw.Draw(sheet)
    for j, row in enumerate(selected[start:start+20]):
        x = (j % 4) * 270
        y = (j // 4) * 130
        draw.text((x+5, y+4), f'{start+j:02d} row {row["blind_test_index"]}, card {row["card_index"]}', fill="white")
        with Image.open(row["tile_png"]) as opened:
            image = opened.convert("RGB").resize((200, 100), Image.Resampling.NEAREST)
        sheet.paste(image, (x+5, y+25))
    sheet.save(OUT / f"tq_review_sheet_{start//20+1}.png")
