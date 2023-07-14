import speech_recognition as sr
import cv2
import pandas as pd
import mediapipe as mp
import data as dt
import evaluate as tr
import numpy as np
from gtts import gTTS
import os
import tkinter as tk
import json
import datetime

counter_left = 0
counter_right = 0
counter_leg = 0
stage_left = None
stage_right = None
stage_leg = None
session_data = []

def add_goal():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something!")
        audio = r.listen(source)
        text = r.recognize_google(audio)
        print(f"You said: {text}")
        if "goal" in text:
            goal = text.split("goal")[1].strip()
            with open("goals.txt", "a") as file:
                file.write(f"goals: {goal}\n")
                print(f"You have set a goal: {goal}")
                tts = gTTS(f"You have set a goal: {goal}")
                tts.save("goal.mp3")
                os.system("mpg321 goal.mp3")
        else:
            tts = gTTS("Sorry, I didn't understand your goal.")
            tts.save("goal.mp3")
            os.system("mpg321 goal.mp3")

X_train, X_test, y_train, y_test = dt.load_and_split_data()
model = tr.train_model(X_train, y_train)


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def workout():
    global counter_left
    global counter_right
    global counter_leg
    global stage_left
    global stage_right
    global stage_leg
    counter_left = 0
    counter_right = 0
    counter_leg = 0
    stage_left = None
    stage_right = None
    stage_leg = None
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(0)

    frame_width = 450
    frame_height = 650
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            # Make Detections
            results = pose.process(image)

            # Recolor image back to BGR for rendering
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Pose Detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=3, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=3, circle_radius=2)
                                      )

            # Export coordinates
            try:
                # Extract Pose landmarks
                body = results.pose_landmarks.landmark
                body_row = list(
                    np.array(
                        [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in body]).flatten())

                # Make Detections
                X = pd.DataFrame([body_row])
                body_language_class = model.predict(X)[0]
                print(body_language_class)

                landmarks = results.pose_landmarks.landmark
                # Left arm
                leftshoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                leftelbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                leftwrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                # Right arm
                rightshoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                rightelbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                rightwrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                # Left leg
                lefthip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                leftknee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                leftankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                # Right leg
                righthip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                rightknee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                rightankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                # Calculate angle
                leftangle = calculate_angle(leftshoulder, leftelbow, leftwrist)
                rightangle = calculate_angle(rightshoulder, rightelbow, rightwrist)

                leftlegangle = calculate_angle(lefthip, leftknee, leftankle)
                rightlegangle = calculate_angle(righthip, rightknee, rightankle)


                # Counter logic for left arm
                if leftangle > 160:
                    stage_left = 'down'
                elif leftangle < 50 and stage_left == 'down':
                    stage_left = 'up'
                    counter_left += 1
                    print(f"Left counter: {counter_left}")

                # Counter logic for right arm
                if rightangle > 160:
                    stage_right = 'down'
                elif rightangle < 50 and stage_right == 'down':
                    stage_right = 'up'
                    counter_right += 1
                    print(f"Right counter: {counter_right}")

                # Counter logic for legs
                if leftlegangle > 160:
                    stage_leg = 'down'
                elif leftlegangle < 50 and stage_leg == 'down':
                    stage_leg = 'up'
                    counter_leg += 1
                    print(f"Leg counter: {counter_leg}")



                # Get status box
                cv2.rectangle(image, (0, 0), (210, 600), (0, 0, 0), -1)
                
                # Display Class
                cv2.putText(image, 'CLASS:'
                            , (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(image, body_language_class.split(' ')[0]
                            , (120, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(image, 'RIGHT:'
                            , (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter_right),
                            (160, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                
                cv2.putText(image, 'LEFT:'
                            , (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter_left),
                            (160, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


                cv2.putText(image, 'LEG:'
                            , (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter_leg),
                            (160, 190),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                

                print('X', X.shape)
                print(body_language_class)

                with open("save.json", "w") as file:
                    data = {'you did the exercise': body_language_class, '': counter, 'times': '','session_data': {}}
                    json.dump(data, file)

                # Load objects
                with open("save.json", "r") as file:
                    data = json.load(file)
                    body_language_class = data['body_language_class']
                    counter = data['counter']
                    session_data = data.get('session_data', {})
                    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                    data['session_data'] = session_data
                    with open("save.json", "w") as file:
                    # data = {'counter': counter, 'body_language_class': body_language_class,
                    #        'session_data': session_data}
                           json.dump(data, file)


            except:
                pass

            cv2.imshow('Raw Webcam Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                cap.release()

def progress():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something!")
        audio = r.listen(source)
        text = r.recognize_google(audio)
        print(f"You said: {text}")
        if "progress" in text:

            with open("save.json", "r") as f:
                file_content = f.read()
                tts = gTTS(f"your progress is: {file_content}")
                tts.save("progress.mp3")
                os.system("mpg321 progress.mp3")
        else:
            tts = gTTS("Sorry, I didn't understand.")
            tts.save("progress.mp3")
            os.system("mpg321 progress.mp3")

root = tk.Tk()
root.title("Workout detector")
root.configure(bg='#F0F0F0')
frame = tk.Frame(root, bg='#F0F0F0')
frame.pack(pady=20)

button = tk.Button(frame,
                   text="Set Goals",
                   fg="#000000",bg="#ffffff", height=2, width=20, font=("Helvetica", 14),
                   command=add_goal)
button.pack(side=tk.LEFT, padx=20)

button = tk.Button(frame,
                   text="Workout",
                   fg="#000000", bg="#ffffff", height=2, width=20, font=("Helvetica", 14),
                   command=workout)
button.pack(side=tk.LEFT, padx=20)

button = tk.Button(frame,
                   text="Progress",
                   fg="#000000", bg="#ffffff", height=2, width=20, font=("Helvetica", 14),
                   command=progress)
button.pack(side=tk.LEFT, padx=20)

root.mainloop()

cv2.destroyAllWindows()