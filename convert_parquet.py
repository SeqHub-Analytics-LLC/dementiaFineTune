import base64, pyarrow.parquet as pq
from io import BytesIO
from pathlib import Path
from PIL import Image
from tqdm import tqdm

PARQUETS = [
    "Alzheimer/Data/train-00000-of-00001-c08a401c53fe5312.parquet",
    "Alzheimer/Data/test-00000-of-00001-44110b9df98c5585.parquet",
]
OUT_ROOT = Path("Alzheimer/images")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# integer-to-name mapping from the dataset card
LABEL_MAP = {
    0: "Mild_Demented",
    1: "Moderate_Demented",
    2: "Non_Demented",
    3: "Very_Mild_Demented",
}

for pq_path in PARQUETS:
    table = pq.read_table(pq_path)
    df = table.to_pandas()                       # needs pandas (already installed)

    for i, (img_rec, int_label) in tqdm(
        enumerate(zip(df["image"], df["label"])),
        total=len(df),
        desc=pq_path,
    ):
        # --- unwrap the image field -----------------------------------------
        if isinstance(img_rec, dict) and "bytes" in img_rec:
            img_bytes = img_rec["bytes"]
        elif isinstance(img_rec, (bytes, bytearray)):
            img_bytes = img_rec
        else:                                   # base-64 text fallback
            img_bytes = base64.b64decode(img_rec)

        img = Image.open(BytesIO(img_bytes)).convert("RGB")

        # --- map the integer label to its string name -----------------------
        label = LABEL_MAP.get(int_label, f"class_{int_label}")
        cls_dir = OUT_ROOT / label
        cls_dir.mkdir(exist_ok=True)

        img.save(cls_dir / f"{Path(pq_path).stem}_{i:05}.jpg",
                 format="JPEG", quality=95)
