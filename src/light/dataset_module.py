import pytorch_lightning as pl
from torch.utils.data import DataLoader

from data.dataset import DatasetAugmentation


class DatasetModule(pl.LightningDataModule):
    """
    Lightning DataModule handling train / val / test datasets.
    """

    def __init__(self, cfg, preprocessing, image_transform):
        super().__init__()

        self.cfg = cfg
        self.preprocessing = preprocessing
        self.image_transform = image_transform

        # Return center labels only for DANN
        self.return_centers = cfg.models.method == "DANN"

    def setup(self, stage=None):
        """
        Initialize datasets.
        """

        directory = f"{self.cfg.dataset.data_dir}/"

        # === Train ===
        self.train_dataset = DatasetAugmentation(
            dataset_url=directory + "train.h5",
            preprocessing=self.image_transform,
            mode="train",
            return_center=self.return_centers,
        )

        # === Validation ===
        self.val_dataset = DatasetAugmentation(
            dataset_url=directory + "val.h5",
            preprocessing=self.preprocessing,
            mode="train",
            return_center=self.return_centers,
        )

        # === Test ===
        self.test_dataset = DatasetAugmentation(
            dataset_url=directory + "test.h5",
            preprocessing=self.preprocessing,
            mode="test",
        )

    def _loader_kwargs(self, shuffle=False):
        """
        Common DataLoader parameters.
        """

        use_workers = self.cfg.dataset.num_workers > 0

        kwargs = dict(
            batch_size=self.cfg.dataset.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.dataset.num_workers,
            pin_memory=True,
        )

        if use_workers:
            kwargs.update(
                persistent_workers=True,
                prefetch_factor=2,
            )

        return kwargs

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self._loader_kwargs(shuffle=True))

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self._loader_kwargs(shuffle=False))

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self._loader_kwargs(shuffle=False))
