from torch.utils.data import Dataset
import mediapipe as mp
import cv2
import numpy as np
import random

N_LANDMARKS = 468

class MaskFace(Dataset):

    def __init__(self, df, perc_mask_predictor=0.65, perc_mask_target=0.55, transforms=None):
        super(MaskFace, self).__init__()
        self.df = df
        self.perc_mask_predictor = perc_mask_predictor
        self.perc_mask_target = perc_mask_target
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh()
        self.transforms = transforms

    def read_rgb(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def calculate_triangles(self, img, height, width):
        landmarks = []

        for lms in img.multi_face_landmarks:
            for idx in range(N_LANDMARKS):
                pts = lms.landmark[idx]
                x = int(pts.x * width)
                y = int(pts.y * height)
                landmarks.append((x, y))

        points = np.array(landmarks, dtype=np.int32)
        convex_hull = cv2.convexHull(points)
        rect = cv2.boundingRect(convex_hull)
        subdiv = cv2.Subdiv2D(rect)
        subdiv.insert(landmarks)
        triangles = np.array(subdiv.getTriangleList(), dtype=np.uint32)

        return triangles
    
    def mask_triangles(self, img, triangles):
        masked_img = img.copy()
        n_triangles_predictor = int(triangles.shape[0]*self.perc_mask_predictor)
        population = np.arange(triangles.shape[0])
        random_triangles = random.sample(list(population), k=n_triangles_predictor)

        for idx in random_triangles:
            coordinates = triangles[idx]
            pts1 = [coordinates[0], coordinates[1]]
            pts2 = [coordinates[2], coordinates[3]]
            pts3 = [coordinates[4], coordinates[5]]
            points = np.array([[pts1, pts2], [pts2, pts3], [pts1, pts3]], dtype=np.int32)
            points = points.reshape(-1, 1, 2)

            masked_img = cv2.fillPoly(masked_img, [points], color=(0,0,0))
        
        n_triangles_targets = int(triangles.shape[0]*self.perc_mask_target)
        target_triangles = random.sample(list(population), k=n_triangles_targets) # NB: these are the triangles you keep! So you need to mask all the others
            
        return masked_img, target_triangles

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        img_path = self.df.loc[idx, 'path']
        img = self.read_rgb(img_path)
        height, width, _ = img.shape
        processed_img = self.face_mesh.process(img)
        triangles = self.calculate_triangles(processed_img, height, width)
        masked_img, target_triangles = self.mask_triangles(img, triangles)

        masked_img = self.transforms(image=masked_img)['image']
        
        return masked_img, target_triangles