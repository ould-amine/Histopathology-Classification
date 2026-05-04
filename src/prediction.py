import torch
import numpy as np
import albumentations as A
import pandas as pd
import fsspec
import h5py
import tqdm

# from models.lora import get_backbone_with_classifier
# from peft import LoraConfig, get_peft_model
from models.model import DANNModel


def get_test_results(model):
    """
    Run inference on the test set and save predictions to CSV.
    """
    # === Setup ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TEST_IMAGES_PATH = "https://minio.lab.sspcloud.fr/spovoa/data/test.h5"

    model.eval()
    model.to(device)
    # === Preprocessing ===
    preprocessing = A.Compose(
        [
            A.Resize(height=98, width=98),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            A.ToTensorV2(transpose_mask=True),
        ]
    )

    # === Load test IDs ===
    print("Loading test IDs...")
    with fsspec.open(TEST_IMAGES_PATH, "rb") as f:
        with h5py.File(f, "r") as hdf:
            test_ids = list(hdf.keys())

    results = {"ID": [], "Pred": []}

    # === Inference ===
    print("Running predictions...")
    with fsspec.open(TEST_IMAGES_PATH, "rb") as f:
        with h5py.File(f, "r") as hdf:
            for test_id in tqdm.tqdm(test_ids):
                img = np.array(hdf[test_id]["img"])
                img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)

                img = preprocessing(image=img)["image"].float()
                img = img.unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(img)
                    logits = (
                        torch.sigmoid(output[0])
                        if isinstance(output, tuple)
                        else torch.sigmoid(output)
                    )
                    pred = logits.squeeze().item()

                results["ID"].append(int(test_id))
                results["Pred"].append(int(pred > 0.5))

    # === Save ===
    pd.DataFrame(results).set_index("ID").to_csv("predictions.csv")


# === Model loading ===
# backbone_with_heads = get_backbone_with_classifier(dino=False, dropout=0.3)

# config = LoraConfig(
#     r=8,
#     lora_alpha=16,
#     target_modules=["query", "value"],
#     lora_dropout=0.1,
#     use_rslora=True,
#     bias="none",
#     modules_to_save=["classifier"],
# )

# model = get_peft_model(backbone_with_heads, config)
model = DANNModel(dino=False)
path_best_model = ""

ckpt = torch.load(path_best_model, weights_only=True)
state_dict = ckpt.get("state_dict", ckpt)
state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}

model.load_state_dict(state_dict)

# === Run ===
get_test_results(model)
