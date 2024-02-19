from torch.utils.data import Dataset
import mediapipe as mp
import cv2
import numpy as np
import random

N_LANDMARKS = 468

class MaskFace(Dataset):

    def __init__(self, df, perc=0.75, transforms=None):
        super(MaskFace, self).__init__()
        self.df = df
        self.perc = perc
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
        n_triangles = int(triangles.shape[0]*self.perc)
        population = np.arange(triangles.shape[0])
        random_triangles = random.sample(list(population), k=n_triangles)

        for idx in random_triangles:
            coordinates = triangles[idx]
            pts1 = [coordinates[0], coordinates[1]]
            pts2 = [coordinates[2], coordinates[3]]
            pts3 = [coordinates[4], coordinates[5]]
            points = np.array([[pts1, pts2], [pts2, pts3], [pts1, pts3]], dtype=np.int32)
            points = points.reshape(-1, 1, 2)

            masked_img = cv2.fillPoly(masked_img, [points], color=(0,0,0))

        return masked_img

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        img_path = self.df.loc[idx, 'path']
        img = self.read_rgb(img_path)
        height, width, _ = img.shape
        processed_img = self.face_mesh.process(img)
        triangles = self.calculate_triangles(processed_img, height, width)
        masked_img = self.mask_triangles(img, triangles)
        masked_img = self.transforms(image=masked_img)['image']
        img = self.transforms(image=img)['image']
        return masked_img, img