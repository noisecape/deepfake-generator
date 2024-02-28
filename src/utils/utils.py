import cv2 
import numpy as np

N_LANDMARKS=468

def read_rgb(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def calculate_triangles(img, height, width):
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