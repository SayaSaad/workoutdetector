import mediapipe as mp
import cv2
import csv
import os
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

class_name = "Pull UP"

cap = cv2.VideoCapture('/Users/saja/Desktop/workout detector/trainingmodel/.mp4')
# Initiate holistic model
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()


        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = pose.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        num_coords = len(results.pose_landmarks.landmark)
        num_coords
        landmarks = ['class']
        for val in range(1, num_coords + 1):
            landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]

        # Pose Detections
        mp_drawing.draw_landmarks(image,
                                  results.pose_landmarks,
                                  mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )
        try:
            # Extract Pose landmarks
            body = results.pose_landmarks.landmark
            body_row = list(
                np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in body]).flatten())


            # Concate rows
            row = body_row
            # Append class name
            row.insert(0, class_name)

            # Export to CSV
            with open('coords.csv', mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(row)


        except:
            pass

        cv2.imshow('Raw Webcam Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()






