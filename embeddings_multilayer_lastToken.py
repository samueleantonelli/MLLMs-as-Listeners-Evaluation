h,
                txt=s["description"] or "[EMPTY]",
                y=lab
            )
        )

print("Record tenuti:", len(records))
if not records:
    pd.DataFrame().to_csv(CSV_OUT, index=False)
    np.savez_compressed(NPZ_OUT, embeddings=np.zeros((0,0)), ids=np.empty(0))
    raise SystemExit("Nessun record valido.")


pd.DataFrame({
    "id":         [r["id"]  for r in records],
    "description":[r["txt"] for r in records],
    "label":      [r["y"]   for r in records],
    "img_path":   [r["img"] for r in records],
}).to_csv(CSV_OUT, index=False)
print("[✓] CSV salvato →", CSV_OUT)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if DEVICE=="cuda" else torch.float32
#!/usr/bin/env python3
import os
import glob
import json
import warnings
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import jinja2
from packaging import version
from transformers import Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration


if version.parse(jinja2.__version__) < version.parse("3.1.0"):
    raise RuntimeError(
        f"Jinja2={jinja2.__version__} troppo vecchia. Esegui:\n"
        "pip install --user --upgrade 'jinja2>=3.1.0'"
    )


DESC_DIR = "/projects/0/prjs1481/probing/descriptions/user_rel_jsons"
IMG_DIR  = "/projects/0/prjs1481/probing/boxed_images/boxed_images"
CSV_OUT  = "/projects/0/prjs1481/probing/probing_records_multilayer.csv"
NPZ_OUT  = "/projects/0/prjs1481/probing/probing_embeddings_multilayer_lastToken_Nobox.npz"


records = []
for jf in glob.glob(os.path.join(DESC_DIR, "*.json")):
    for e in json.load(open(jf)):
        s, ev = e["sample_data"], e["evaluation_data"]
        if ev.get("cannot_tell"):           continue
        if s.get("co_occurrence", 0) <= 1:  continue
        if   ev.get("in_mask"):             lab = 1
        elif ev.get("multiple_matches"):    lab = 0
        else:                               continue
        img_path = os.path.join(IMG_DIR, s["boxed_image_path"])
        if not os.path.exists(img_path):
            warnings.warn(f"missing {img_path}")
            continue
        records.append({
            "id":  s["image_path"],
            "img": img_path,
            "txt": s["description"] or "[EMPTY]",
            "y":   lab
        })

print("Record tenuti:", len(records))
if not records:
    pd.DataFrame().to_csv(CSV_OUT, index=False)
    np.savez_compressed(NPZ_OUT, embeddings=np.zeros((0,0)), ids=np.empty(0))
    raise SystemExit("Nessun record valido.")


pd.DataFrame({
    "id":          [r["id"]  for r in records],
    "description": [r["txt"] for r in records],
    "label":       [r["y"]   for r in records],
    "img_path":    [r["img"] for r in records],
}).to_csv(CSV_OUT, index=False)
print("[✓] CSV salvato →", CSV_OUT)



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32
MODEL  = "Qwen/Qwen2.5-VL-7B-Instruct"
HF_TOKEN = os.getenv("HF_HUB_TOKEN")

processor = Qwen2_5_VLProcessor.from_pretrained(
    MODEL,
    token=HF_TOKEN,
    trust_remote_code=True
)
full_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL,
    token=HF_TOKEN,
    torch_dtype=DTYPE,
    device_map="auto" if DEVICE == "cuda" else None,
    trust_remote_code=True
).eval()

backbone = full_model.model


BATCH = 4
embs, ids = [], []

for i in tqdm(range(0, len(records), BATCH), desc="Embeddings"):
    batch = records[i:i + BATCH]
    imgs, chats = [], []
    for r in batch:
        pil = Image.open(r["img"]).convert("RGB")
        chat = processor.apply_chat_template(
            [{"role": "user",
              "content": [
                  {"type": "image", "image": pil},
                  {"type": "text",  "text": r["txt"]}
              ]}],
            tokenize=False,
            add_generation_prompt=False
        )
        imgs.append(pil)
        chats.append(chat)

    inputs = processor(
        text=chats,
        images=imgs,
        padding=True,
        truncation=False,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        out = backbone(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True
        )
        hidden_states = out.hidden_states

        
        all_layers = [h[:, -1, :] for h in hidden_states]  # (B, H) × 29
        stacked    = torch.stack(all_layers, dim=1)         # (B, 29, H)

        embs.append(stacked.cpu().to(torch.float16).numpy())

    ids.extend(r["id"] for r in batch)
    del out, inputs
    torch.cuda.empty_cache()

emb = np.concatenate(embs, axis=0)  # (N, 29, H)

np.savez_compressed(NPZ_OUT, embeddings=emb, ids=np.array(ids))
print("[✓] NPZ salvato →", NPZ_OUT, "con shape", emb.shape)
