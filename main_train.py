from src.datasets.mask_face import MaskFace
import pandas as pd
from torch.utils.data import DataLoader
from albumentations.pytorch.transforms import ToTensorV2
import albumentations as A

N_LANDMARKS = 468

if __name__ == "__main__":

    df = pd.read_pickle(r"C:\Users\Noisecape\AI\Datasets\dataset_processed.pkl")

    transforms = A.Compose({
        A.Resize(224, 224),
        ToTensorV2()
        })

    min_triangles = df['#_triangles'].min()

    n_triangles_predictor = int(min_triangles * 0.55)
    n_triangles_encoder = int(min_triangles * 0.75)
    
    dataset = MaskFace(
        df,
        n_triangles_predictor=n_triangles_predictor, 
        n_triangles_encoder=n_triangles_encoder, 
        transforms=transforms
        )

    dataloader = DataLoader(dataset, batch_size=3, shuffle=True)

    for batch in dataloader:

        masked_img = batch[0]
        gt_img = batch[0]

        print()