from albumentations.core.transforms_interface import ImageOnlyTransform
import torchstain
import torch
import numpy as np
import h5py
import fsspec


def load_ref_images(dataset_url, n_images, seed, cache_dir="/tmp/.cache/h5"):
    """
    Load reference images from dataset (with caching).
    """
    cached_url = f"filecache::{dataset_url}"
    cache_opts = {"filecache": {"cache_storage": cache_dir}}

    rng = np.random.default_rng(seed=seed)
    images = []

    with fsspec.open(cached_url, "rb", **cache_opts) as f:
        with h5py.File(f, "r") as hdf:
            keys = list(hdf.keys())
            indices = rng.integers(0, len(keys), size=n_images)

            for idx in indices:
                img = np.array(hdf[keys[idx]]["img"])
                t = torch.from_numpy((img * 255)).float()
                images.append(t)
    if len(images) == 1:
        return images[0]
    return images


class HistoNormalization(ImageOnlyTransform):
    """
    Histopathology stain normalization using torchstain.

    If no target is provided, random images from the training set are used.
    """

    def __init__(
        self,
        target=None,
        method="Macenko",
        p=0.5,
        n_images=1,
        train_images_path=None,
        seed=None,
        cfg=None,
    ):
        super().__init__(p=p)

        self.method = method

        # === Target selection ===
        if target is None:
            if train_images_path is None:
                train_images_path = f"{cfg.dataset.data_dir}/train.h5"
            if seed is None:
                seed = cfg.training.seed

            target = load_ref_images(
                dataset_url=train_images_path,
                n_images=n_images,
                seed=seed,
            )

        self.target = target
        self.normalizer = self._build_normalizer()

    def _build_normalizer(self):
        """
        Initialize stain normalizer.
        """

        if self.method == "Macenko":
            normalizer = torchstain.normalizers.MacenkoNormalizer(backend="torch")

        elif self.method == "Modified_Reinhard":
            normalizer = torchstain.normalizers.ReinhardNormalizer(
                backend="torch", method="modified"
            )

        elif self.method == "MT_Macenko":
            normalizer = torchstain.normalizers.MultiMacenkoNormalizer(backend="torch")

        else:
            raise ValueError(
                "method must be one of: Macenko, Modified_Reinhard, MT_Macenko"
            )

        normalizer.fit(self.target)
        return normalizer

    def apply(self, img, **params):
        """
        Apply normalization to image.
        """

        # HWC uint8 → CHW float
        t = torch.from_numpy(img).permute(2, 0, 1).float()

        try:
            if self.method == "Modified_Reinhard":
                norm = self.normalizer.normalize(I=t)
            else:
                norm, _, _ = self.normalizer.normalize(I=t, stains=False)

            return norm.clamp(0, 255).byte().numpy()

        except (torch._C._LinAlgError, RuntimeError):
            # Numerical instability -> return original image
            return img

        except Exception:
            return img
