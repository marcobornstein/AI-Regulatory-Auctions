"""FairFace gender classification with race (0 = White, 1 = Black) as the sensitive attribute."""

import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


def sample_mixture(image_list, num_majority, minority_fraction, seed):
    """Sample num_majority White images plus enough Black images to make up minority_fraction of the total."""
    df = pd.read_csv(image_list)
    total = num_majority // (1 - minority_fraction)
    majority = df[df["race"] == 0].sample(n=num_majority, replace=False, random_state=seed)
    minority = df[df["race"] == 1].sample(n=int(total * minority_fraction), replace=False, random_state=seed)
    return pd.concat([majority, minority], ignore_index=True)


class FairFaceDataset(Dataset):
    def __init__(self, root, frame, transform=None):
        self.root, self.frame, self.transform = root, frame, transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        assert row["gender"] in (0, 1) and row["race"] in (0, 1)
        image = Image.open(os.path.join(self.root, row["file"]).rstrip()).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {"image": image, "label": {"age": row["age"], "gender": row["gender"], "race": row["race"]}}
