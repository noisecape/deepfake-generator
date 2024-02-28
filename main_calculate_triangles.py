import pandas as pd
from src.datasets.mask_face import MaskFace
from tqdm.auto import tqdm
from src.utils.utils import read_rgb, calculate_triangles
import mediapipe as mp

N_LANDMARKS = 468

if __name__ == "__main__":

    df = pd.read_pickle(r"C:\Users\Noisecape\AI\Datasets\dataset.pkl")
    calculate_triangles_loop = tqdm(df.iterrows(), total=len(df))
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()
    for idx, row in calculate_triangles_loop:
        try:
            path = row['path']
            img = read_rgb(path)
            height, width, _ = img.shape
            
            processed_img = face_mesh.process(img)
            triangles = calculate_triangles(processed_img, height, width)
            df.loc[idx, '#_triangles'] = triangles.shape[0]
        except Exception as e:
            print(f"Error, {e}")

    df.to_pickle(r"C:\Users\Noisecape\AI\Datasets\dataset_processed.pkl")
    
