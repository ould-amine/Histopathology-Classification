import h5py
import numpy as np
from torch.utils.data import Dataset
import os
import fsspec

CENTER_MAP = {
    0: 0,
    3: 1,
    4: 2,
    1: 3,
    2: 4,
}


class DatasetAugmentation(Dataset):
    """
    Dataset wrapper for HDF5 images with preprocessing and optional metadata.
    """

    def __init__(
        self,
        dataset_url,
        preprocessing,
        mode,
        cache_dir="/tmp/.cache/h5",
        return_center=False,
    ):

        super().__init__()

        self.preprocessing = preprocessing
        self.mode = mode
        self.return_center = return_center

        self.cached_url = f"filecache::{dataset_url}"
        self.cache_opts = {"filecache": {"cache_storage": cache_dir}}

        self._hdf = None
        os.makedirs(cache_dir, exist_ok=True)

        # === Load image IDs ===
        with fsspec.open(self.cached_url, "rb", **self.cache_opts) as f:
            with h5py.File(f, "r") as hdf:
                self.image_ids = sorted(hdf.keys(), key=lambda x: int(x))

    def __len__(self):
        return len(self.image_ids)

    def _ensure_hdf(self):
        """
        Lazy loading of HDF5 file (in case of multiprocessing).
        """
        if self._hdf is None:
            file_obj = fsspec.open(self.cached_url, "rb", **self.cache_opts).open()
            self._file_obj = file_obj
            self._hdf = h5py.File(file_obj, "r")

        return self._hdf

    def __getitem__(self, idx):
        hdf = self._ensure_hdf()
        img_id = self.image_ids[idx]
        data = hdf[img_id]

        # === Image ===
        img = np.array(data["img"], dtype=np.float32)  # CHW float in [0, 1]
        img = np.transpose(img, (1, 2, 0))
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

        # === Label ===
        label = None
        if "label" in data:
            label = np.array(data["label"]).astype("float32")

        # === Center (optional) ===
        center = None
        if self.return_center and self.mode == "train":
            center_raw = int(np.array(data["metadata"])[0])
            center = CENTER_MAP.get(center_raw, center_raw)

        # === Transform ===
        img = self.preprocessing(image=img)["image"].float()
        if self.return_center:
            return img, label, center
        return img, label

    def __del__(self):
        """
        Properly close HDF5 file.
        """
        if getattr(self, "_hdf", None) is not None:
            try:
                self._hdf.close()
            except Exception:
                pass

        if getattr(self, "_file_obj", None) is not None:
            try:
                self._file_obj.close()
            except Exception:
                pass
