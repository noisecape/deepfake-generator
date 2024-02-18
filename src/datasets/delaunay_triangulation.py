import mediapipe as mp
import cv2
import numpy as np


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

    
if __name__ == "__main__":

    mp_face_mesh = mp.solutions.face_mesh

    face_mesh = mp_face_mesh.FaceMesh()
    video_capture = cv2.VideoCapture(2)

    while True:
        ret, img = video_capture.read()

        height, width, _ = img.shape
        processed_face = face_mesh.process(img)
         # img = draw_landmarks(processed_face, img)
        triangles = calculate_triangles(processed_face)
        draw_triangles(triangles, img)
        draw_landmarks(processed_face, img)
        
        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    cv2.waitKey(0)
    cv2.destroyAllWindows()