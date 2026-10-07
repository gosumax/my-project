"""Make readable contact sheets from the frozen blind review manifest."""
import json
from pathlib import Path
from PIL import Image, ImageDraw

root = Path(__file__).resolve().parents[1] / "runs/card_train_smoke16_test_20260930"
rows = [json.loads(line) for line in (root / "smoke16_blind_review_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
for start in range(0, len(rows), 10):
    sheet = Image.new("RGB", (1100, 920), "#182334")
    draw = ImageDraw.Draw(sheet)
    for offset, row in enumerate(rows[start:start+10]):
        image = Image.open(row["chat_row_png"]).convert("RGB")
        image = image.resize((image.width * 4, image.height * 4), Image.Resampling.NEAREST)
        y = offset * 92
        draw.text((10, y + 4), f'{start+offset:02d}  {row["decoded_time_s"]:.1f}s  T{row["table_id"]}', fill="white")
        sheet.paste(image, (110, y + 18))
    sheet.save(root / f"smoke16_review_sheet_{start//10+1:02d}.png")
