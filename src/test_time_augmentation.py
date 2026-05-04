import torch
from tqdm.auto import tqdm
import albumentations as A
import torchmetrics
from data.dataset import DatasetAugmentation

# from data.normalization import HistoNormalization
from torch.utils.data import DataLoader
from utilities import set_seed

# from models.model import DANNModel
from models.lora import get_backbone_with_classifier
from peft import LoraConfig, get_peft_model


def test_time_aug(model, N=10):
    """ "
    Run test time augmentation/normalization on the validation set.
    It uses the mean of the probs (and not the majority vote)
    Print the results on each run and the majority vote.
    Be careful, the augmentation/normalization is hard-coded !
    """

    # === Setup ===
    VAL_IMAGES_PATH = "https://minio.lab.sspcloud.fr/spovoa/data/val.h5"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric = getattr(torchmetrics, "Accuracy")("binary").to(device)

    # === Preprocessing ===
    preprocessing = A.Compose(
        [
            A.Resize(height=98, width=98),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.ToTensorV2(transpose_mask=True),
        ]
    )

    model.eval()
    model.to(device)

    all_probs = []
    all_labels = None

    for i in range(N):
        set_seed(i)

        # === Transform augmentation/normalization ===

        transform = A.Compose(
            [
                A.HEStain(
                    method="random_preset",
                    intensity_scale_range=(0.8, 1.2),
                    intensity_shift_range=(-0.1, 0.1),
                    augment_background=False,
                    p=1,
                ),
                #     # HistoNormalization(
                #     #     method="Modified_Reinhard",
                #     #     p=1,
                #     #     n_images=1,
                #     #     train_images_path=TRAIN_IMAGES_PATH,
                #     #     seed=i),
                preprocessing,
            ]
        )

        val_dataset = DatasetAugmentation(VAL_IMAGES_PATH, transform, "train")
        val_dataloader = DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=128,
            num_workers=16,
            pin_memory=True,
            prefetch_factor=2,
        )

        iter_probs, iter_labels = [], []

        # === Inference==
        for val_x, val_y in tqdm(val_dataloader, leave=False, desc=f"TTA {i + 1}/{N}"):
            with torch.no_grad():
                output = model(val_x.to(device))
                probs = (
                    torch.sigmoid(output[0])
                    if isinstance(output, tuple)
                    else torch.sigmoid(output)
                )
                probs = torch.squeeze(probs)
            iter_probs.append(probs)
            iter_labels.append(val_y.int())

        iter_probs = torch.cat(iter_probs)  # (N_val,)
        iter_labels = torch.cat(iter_labels)  # (N_val,)

        if all_labels is None:
            all_labels = iter_labels

        # === Score ===
        iter_preds = (iter_probs > 0.5).int()
        iter_metric = metric(iter_preds.to(device), all_labels.to(device)).item()
        print(f"[TTA {i + 1:02d}/{N}] accuracy = {iter_metric:.4f}")

        all_probs.append(iter_probs)

    #  === Mean probabilities ===
    mean_probs = torch.stack(all_probs).mean(dim=0)  # (N_val,)
    mean_preds = (mean_probs > 0.5).int()
    mean_metric = metric(mean_preds.to(device), all_labels.to(device)).item()
    print(f"\n── TTA finale (moyenne de {N} augmentations) ──")
    print(f"Accuracy = {mean_metric:.4f}")

    return {
        "per_aug_accuracy": [
            metric((p > 0.5).int().to(device), all_labels.to(device)).item()
            for p in all_probs
        ],
        "tta_accuracy": mean_metric,
        "mean_probs": mean_probs,
    }


#  === Models (hard coded) ===

# model = DANNModel()
backbone_with_heads = get_backbone_with_classifier(dino=False, dropout=0.3)
config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    use_rslora=True,
    bias="none",
    modules_to_save=["classifier"],
)
model = get_peft_model(backbone_with_heads, config)
path_best_model = ""
ckpt = torch.load(path_best_model, weights_only=True)
state_dict = ckpt.get("state_dict", ckpt)  # compatible Lightning et torch.save
state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)

# === Run ===
test_time_aug(model, N=5)
