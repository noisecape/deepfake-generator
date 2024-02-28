from src.datasets.mask_face import MaskFace
import pandas as pd
from torch.utils.data import DataLoader
from albumentations.pytorch.transforms import ToTensorV2
import albumentations as A

if __name__ == "__main__":

    df = pd.read_pickle(r"C:\Users\Noisecape\AI\Datasets\dataset.pkl")
    transforms = A.Compose({
        A.Resize(224, 224),
        ToTensorV2()
        })
    dataset = MaskFace(df, perc=0.45, transforms=transforms)

    dataloader = DataLoader(dataset, batch_size=3, shuffle=True)

    for batch in dataloader:

        masked_img = batch[0]
        gt_img = batch[0]

        print()