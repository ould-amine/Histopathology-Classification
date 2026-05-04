import albumentations as A
from albumentations.pytorch import ToTensorV2
from data.normalization import HistoNormalization


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _build_normalizer(cfg, target=None, p=1.0):
    """
    Build histopathology normalization if enabled.
    """
    if not cfg.augmentation.HistoNormalization.enabled:
        return None

    return HistoNormalization(
        target=target,
        method=cfg.augmentation.HistoNormalization.method,
        p=p,
        n_images=cfg.augmentation.HistoNormalization.n_images,
        cfg=cfg,
    )


def get_preprocessing_and_augmentation(cfg):
    """
    Build training and validation preprocessing pipelines.
    """

    image_size = cfg.dataset.image_size

    # === Normalization ===
    histo_norm = _build_normalizer(
        cfg,
        target=None,
        p=cfg.augmentation.HistoNormalization.p,
    )

    histo_norm_val = None
    if histo_norm is not None and cfg.augmentation.HistoNormalization.validation:
        histo_norm_val = _build_normalizer(cfg, target=histo_norm.target, p=1.0)

    # === Common post-processing ===
    base_post = [
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(transpose_mask=True),
    ]

    # === Train transforms ===
    train_transforms = []
    if histo_norm is not None:
        train_transforms.append(histo_norm)

    train_transforms.extend(
        [
            A.RandomResizedCrop(
                size=(image_size, image_size), scale=(0.85, 1.0), ratio=(0.9, 1.1)
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(
                brightness=cfg.augmentation.ColorJitter.brightness,
                contrast=cfg.augmentation.ColorJitter.contrast,
                saturation=cfg.augmentation.ColorJitter.saturation,
                hue=cfg.augmentation.ColorJitter.hue,
                p=cfg.augmentation.ColorJitter.p,
            ),
            A.GaussNoise(std_range=(0.01, 0.06), p=cfg.augmentation.GaussNoise.p),
            A.GaussianBlur(blur_limit=(3, 5), p=cfg.augmentation.GaussianBlur.p),
            A.ImageCompression(
                quality_range=(70, 100), p=cfg.augmentation.ImageCompression.p
            ),
        ]
    )

    if cfg.augmentation.HEStain.enabled:
        train_transforms.append(
            A.HEStain(
                method="random_preset",
                intensity_scale_range=(
                    1 - cfg.augmentation.HEStain.intensity_scale_range,
                    1 + cfg.augmentation.HEStain.intensity_scale_range,
                ),
                intensity_shift_range=(
                    -cfg.augmentation.HEStain.intensity_shift_range,
                    cfg.augmentation.HEStain.intensity_shift_range,
                ),
                augment_background=False,
                p=cfg.augmentation.HEStain.p,
            )
        )

    train_transforms.extend(base_post)
    train_transform = A.Compose(train_transforms)

    # === Validation transforms ===
    val_transforms = []
    if histo_norm_val is not None:
        val_transforms.append(histo_norm_val)
    val_transforms.extend(
        [
            A.Resize(height=image_size, width=image_size),
            *base_post,
        ]
    )
    preprocessing = A.Compose(val_transforms)

    return preprocessing, train_transform

    histo_norm = _build_normalizer(
        cfg,
        target=None,
        p=cfg.augmentation.HistoNormalization.p,
    )

    histo_norm_val = None
    if histo_norm is not None and cfg.augmentation.HistoNormalization.validation:
        histo_norm_val = _build_normalizer(cfg, target=histo_norm.target, p=1.0)

    base_post = [
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(transpose_mask=True),
    ]

    train_transforms = []

    if histo_norm is not None:
        train_transforms.append(histo_norm)

    train_transforms.extend(
        [
            A.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.85, 1.0),
                ratio=(0.9, 1.1),
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(
                brightness=cfg.augmentation.ColorJitter.brightness,
                contrast=cfg.augmentation.ColorJitter.contrast,
                saturation=cfg.augmentation.ColorJitter.saturation,
                hue=cfg.augmentation.ColorJitter.hue,
                p=cfg.augmentation.ColorJitter.p,
            ),
            A.GaussNoise(
                std_range=(0.01, 0.06),
                p=cfg.augmentation.GaussNoise.p,
            ),
            A.GaussianBlur(
                blur_limit=(3, 5),
                p=cfg.augmentation.GaussianBlur.p,
            ),
            A.ImageCompression(
                quality_range=(70, 100),
                p=cfg.augmentation.ImageCompression.p,
            ),
        ]
    )

    if cfg.augmentation.HEStain.enabled:
        train_transforms.append(
            A.HEStain(
                method="random_preset",
                intensity_scale_range=(
                    1 - cfg.augmentation.HEStain.intensity_scale_range,
                    1 + cfg.augmentation.HEStain.intensity_scale_range,
                ),
                intensity_shift_range=(
                    -cfg.augmentation.HEStain.intensity_shift_range,
                    cfg.augmentation.HEStain.intensity_shift_range,
                ),
                augment_background=False,
                p=cfg.augmentation.HEStain.p,
            )
        )

    train_transforms.extend(base_post)
    train_transform = A.Compose(train_transforms)

    val_transforms = []

    if histo_norm_val is not None:
        val_transforms.append(histo_norm_val)

    val_transforms.extend(
        [
            A.Resize(height=image_size, width=image_size),
            *base_post,
        ]
    )

    preprocessing = A.Compose(val_transforms)

    return preprocessing, train_transform
