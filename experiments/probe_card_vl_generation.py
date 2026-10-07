"""Exercise selected adapter generation on one VALIDATION tile, never Smoke16."""
import json
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from experiments.train_card_vl_lora import attach_lora, CARD_IMAGE_PIXELS, DATA, PROMPT, SPLIT, read_rows

def main():
    row = read_rows(SPLIT / "validation_manifest.jsonl")[0]
    config = json.loads((DATA / "train_run/train_config.json").read_text(encoding="utf-8"))
    selection = json.loads((DATA / "train_run/checkpoint_selection_final.json").read_text(encoding="utf-8"))
    model = AutoModelForImageTextToText.from_pretrained(config["model_dir"], dtype=torch.float16,
                                                         attn_implementation="sdpa", local_files_only=True).to("cuda")
    processor = AutoProcessor.from_pretrained(config["model_dir"], local_files_only=True, use_fast=False)
    attach_lora(model, rank=config["lora_rank"], alpha=config["lora_alpha"])
    state = torch.load(selection["selected_checkpoint"], map_location="cpu", weights_only=True)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            parameter.data.copy_(state[name].to(parameter.device))
    model.eval()
    with Image.open(row["tile_png"]) as opened:
        image = opened.convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT}]}]
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                           return_dict=True, return_tensors="pt",
                                           images_kwargs={"size": {"shortest_edge": CARD_IMAGE_PIXELS,
                                                                    "longest_edge": CARD_IMAGE_PIXELS}}).to("cuda")
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=8, do_sample=False, use_cache=True)
    raw_text = processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    print(json.dumps({"source": "VALIDATION", "ground_truth": row["ground_truth"],
                      "raw_response": raw_text, "selected_epoch": selection["best_epoch"]}, ensure_ascii=False))

if __name__ == "__main__":
    main()
