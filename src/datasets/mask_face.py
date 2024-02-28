from torch.utils.data import Dataset
import mediapipe as mp
import cv2
import numpy as np
import random
from src.utils.utils import calculate_triangles

N_LANDMARKS = 468

class MaskFace(Dataset):

    def __init__(self, df, n_triangles_predictor, n_triangles_encoder, transforms=None):
        super(MaskFace, self).__init__()
        self.df = df
        self.n_triangles_predictor = n_triangles_predictor
        self.n_triangles_encoder = n_triangles_encoder
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh()
        self.transforms = transforms

    def read_rgb(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def mask_triangles(self, img, triangles):
        masked_img = img.copy()
        population = np.arange(triangles.shape[0])
        random_triangles = random.sample(list(population), k=self.n_triangles_encoder)

        for idx in random_triangles:
            coordinates = triangles[idx]
            pts1 = [coordinates[0], coordinates[1]]
            pts2 = [coordinates[2], coordinates[3]]
            pts3 = [coordinates[4], coordinates[5]]
            points = np.array([[pts1, pts2], [pts2, pts3], [pts1, pts3]], dtype=np.int32)
            points = points.reshape(-1, 1, 2)

            masked_img = cv2.fillPoly(masked_img, [points], color=(0,0,0))
        
        target_triangles = random.sample(list(population), k=self.n_triangles_predictor) # NB: these are the triangles you keep! So you need to mask all the others
            
        return masked_img, target_triangles

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        img_path = self.df.loc[idx, 'path']
        img = self.read_rgb(img_path)
        height, width, _ = img.shape
        processed_img = self.face_mesh.process(img)
        triangles = calculate_triangles(processed_img, height, width)
        masked_img, target_triangles = self.mask_triangles(img, triangles)
        
        masked_img = self.transforms(image=masked_img)['image']
        assert masked_img.shape == (3, 224, 224)
         
        return masked_img, target_triangles