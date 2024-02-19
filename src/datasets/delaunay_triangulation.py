import mediapipe as mp
import cv2
import numpy as np
import random


def draw_landmarks(processed_face, img):
    for lms in processed_face.multi_face_landmarks:
        for idx in range(468):
            pts = lms.landmark[idx]
            x = int(pts.x * width)
            y = int(pts.y * height)

            cv2.circle(img, (x,y), 1, (0, 255, 0), -1)


def draw_triangles(triangles, img):
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        cv2.line(img, pt1, pt2, (0, 0, 255), 1)
        cv2.line(img, pt2, pt3, (0, 0, 255), 1)
        cv2.line(img, pt1, pt3, (0, 0, 255), 1)


def calculate_triangles(processed_face):
    landmarks = []
    for lms in processed_face.multi_face_landmarks:
        for idx in range(468):
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


def mask_triangles(img, triangles, perc=0.75):

    n_triangles = int(triangles.shape[0]*perc)
    population = np.arange(triangles.shape[0])
    random_triangles = random.sample(list(population), k=n_triangles)

    for idx in random_triangles:
        coordinates = triangles[idx]
        pts1 = [coordinates[0], coordinates[1]]
        pts2 = [coordinates[2], coordinates[3]]
        pts3 = [coordinates[4], coordinates[5]]
        points = np.array([[pts1, pts2], [pts2, pts3], [pts1, pts3]], dtype=np.int32)
        points = points.reshape(-1, 1, 2)

        cv2.fillPoly(img, [points], color=(0,0,0))

    
if __name__ == "__main__":

    mp_face_mesh = mp.solutions.face_mesh

    face_mesh = mp_face_mesh.FaceMesh()
    # video_capture = cv2.VideoCapture(2)
    img = cv2.imread(r'C:\Users\Noisecape\AI\Datasets\CelebA-HQ\53956.png')

    while True:
        # ret, img = video_capture.read()

        height, width, _ = img.shape
        processed_face = face_mesh.process(img)
         # img = draw_landmarks(processed_face, img)
        triangles = calculate_triangles(processed_face)
        # draw_triangles(triangles, img)
        # draw_landmarks(processed_face, img)
        mask_triangles(img, triangles)

        
        cv2.imshow("Image", img)

        # if cv2.waitKey(1) & 0xFF == ord('q'): 
        #     break
        break

    cv2.waitKey(0)
    cv2.destroyAllWindows()