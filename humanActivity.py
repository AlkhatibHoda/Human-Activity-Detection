import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import datetime


mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()


screen_width = 1920
screen_height = 1080


prev_wrist_x = None
last_wave_time = time.time()
cooldown_time = 2
is_waving = False
prev_wrist_distance = None
clap_threshold = 0.1
is_clapping = False
user_present = False
last_presence_status = None
prev_hip_y = None
is_jumping = False
jump_threshold = 0.05
prev_left_foot_y = None
prev_right_foot_y = None
is_walking = False
walking_threshold = 0.02
cooldown_frames = 15
inactivity_counter = 0


prev_time = time.time()


log_file = open("activity_log.txt", "a")


while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame.")
        break

    
    frame_height, frame_width, _ = frame.shape
    aspect_ratio = frame_width / frame_height
    new_width = int(screen_width / 2)
    new_height = int(new_width / aspect_ratio)

    if new_height > screen_height:
        new_height = screen_height
        new_width = int(new_height * aspect_ratio)

    frame_resized = cv2.resize(frame, (new_width, new_height))

    
    black_frame = np.ones((screen_height, screen_width, 3), dtype=np.uint8) * 0
    light_blue_frame = np.ones((screen_height, screen_width, 3), dtype=np.uint8) * np.array([203, 192, 255], dtype=np.uint8)
    
    
    x_offset = (screen_width - frame_resized.shape[1]) // 2
    y_offset = (screen_height - frame_resized.shape[0]) // 2
    black_frame[y_offset:y_offset + frame_resized.shape[0], x_offset:x_offset + frame_resized.shape[1]] = frame_resized
    light_blue_frame[y_offset:y_offset + frame_resized.shape[0], x_offset:x_offset + frame_resized.shape[1]] = frame_resized

    
    rgb_frame = cv2.cvtColor(light_blue_frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    
    activity = "Unknown"
    waving_status = "Not waving"
    clapping_status = "Not clapping"
    hand_raise_status = "No hand raised"
    confidence_display = ""

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        
        mp_draw.draw_landmarks(light_blue_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        
        left_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
        right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y
        left_knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y
        right_knee_y = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y
        avg_hip_y = (left_hip_y + right_hip_y) / 2
        avg_knee_y = (left_knee_y + right_knee_y) / 2

        if avg_hip_y < avg_knee_y - 0.05:
            activity = "Standing"
        else:
            activity = "Sitting"

        
        right_wrist_x = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x
        nose_x = landmarks[mp_pose.PoseLandmark.NOSE].x

        if prev_wrist_x is not None and abs(right_wrist_x - prev_wrist_x) > 0.05 and abs(nose_x - landmarks[mp_pose.PoseLandmark.NOSE].x) < 0.02:
            last_wave_time = time.time()
            is_waving = True

        prev_wrist_x = right_wrist_x

        if is_waving and (time.time() - last_wave_time < cooldown_time):
            waving_status = "Waving!"
        elif time.time() - last_wave_time >= cooldown_time:
            is_waving = False

       
        left_wrist_x = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x
        right_wrist_x = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x
        wrist_distance = abs(left_wrist_x - right_wrist_x)

        if wrist_distance < clap_threshold:
            is_clapping = True
        else:
            is_clapping = False

        if is_clapping:
            clapping_status = "Clapping!"

        prev_wrist_distance = wrist_distance

        
        left_wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y
        right_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y
        left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
        right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y

        if left_wrist_y < left_shoulder_y:
            hand_raise_status = "Left Hand Raised"
        if right_wrist_y < right_shoulder_y:
            hand_raise_status = "Right Hand Raised"

        
        if prev_hip_y is not None:
            if avg_hip_y < prev_hip_y - jump_threshold:
                is_jumping = True
            elif avg_hip_y > prev_hip_y + jump_threshold:
                is_jumping = False

        prev_hip_y = avg_hip_y

        
        left_foot_y = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y
        right_foot_y = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y

        if prev_left_foot_y is not None and prev_right_foot_y is not None:
            left_foot_movement = abs(left_foot_y - prev_left_foot_y)
            right_foot_movement = abs(right_foot_y - prev_right_foot_y)

            if left_foot_movement > walking_threshold or right_foot_movement > walking_threshold:
                is_walking = True
                inactivity_counter = 0
            else:
                inactivity_counter += 1
                if inactivity_counter > cooldown_frames:
                    is_walking = False

        prev_left_foot_y = left_foot_y
        prev_right_foot_y = right_foot_y

        walking_status = "Walking!" if is_walking else "Not Walking"
        cv2.putText(light_blue_frame, f"Walking Status: {walking_status}", (5, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        if is_walking:
            log_file.write(f"{datetime.now()}: User is walking\n")

        
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        
        cv2.putText(light_blue_frame, "Human Activity Recognition Detection", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(light_blue_frame, f"Activity: {activity}", (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(light_blue_frame, f"Action: {waving_status}", (5, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(light_blue_frame, f"Action: {clapping_status}", (5, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(light_blue_frame, f"Hand Status: {hand_raise_status}", (5, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        presence_status = "User Present" if user_present else "No User Detected"
        cv2.putText(light_blue_frame, f"Presence: {presence_status}", (5, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        jumping_status = "Jumping!" if is_jumping else "Not Jumping"
        cv2.putText(light_blue_frame, f"Jump Status: {jumping_status}", (5, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        
        if user_present != last_presence_status:
            last_presence_status = user_present
            log_file.write(f"{datetime.now()}: User {'present' if user_present else 'absent'}\n")

    
    cv2.imshow("Activity Recognition", light_blue_frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
log_file.close()
