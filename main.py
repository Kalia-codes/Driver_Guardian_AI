import cv2
import dlib
import imutils
from imutils import face_utils
import time
import pygame
import numpy as np
import os

# Shared live status for mobile app
from status_data import driver_status

# NEW: Alert screenshot + history
from alert_utils import save_alert

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from utils.facial_features import eye_aspect_ratio, mouth_aspect_ratio, get_head_pose
from utils.object_detection import detect_objects

# -------- Configuration / Thresholds --------
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 20

MOUTH_AR_THRESH = 0.65
MOUTH_AR_CONSEC_FRAMES = 15

HEAD_TILT_THRESH = 30.0
HEAD_TILT_CONSEC_FRAMES = 15

# -------- Initialize Pygame for Alarm --------
pygame.mixer.init()
pygame.mixer.music.set_volume(1.0)


def play_alarm():
    if not pygame.mixer.music.get_busy():
        try:
            alarm_path = os.path.join(BASE_DIR, 'alarm.wav')

            if not os.path.exists(alarm_path):
                print(f"[ERROR] Alarm file not found: {alarm_path}")
                return

            pygame.mixer.music.load(alarm_path)

            # Continuous loop
            pygame.mixer.music.play(-1)

            print("[ALARM] Alarm Started")

        except Exception as e:
            print("[ERROR] Could not play alarm:", e)


def stop_alarm():
    try:
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
            print("[ALARM] Alarm Stopped")

    except Exception as e:
        print("[ERROR] Could not stop alarm:", e)


# -------- Load Dlib's face detector and shape predictor --------
print("[INFO] Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()

predictor_path = os.path.join(BASE_DIR, "shape_predictor_68_face_landmarks.dat")

if not os.path.exists(predictor_path):
    print(f"[ERROR] Could not find {predictor_path}. Ensure it is downloaded.")

predictor = dlib.shape_predictor(predictor_path)

# Extract landmark indices
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = (60, 68)

# Counters
ear_counter = 0
tilt_counter = 0
yawn_counter = 0

# -------- Session Monitoring Statistics --------
session_start_time = time.time()

metrics = {
    "drowsiness_alerts": 0,
    "yawning_alerts": 0,
    "distraction_alerts": 0,
    "phone_usage_alerts": 0
}

# Cooldown timers
last_drowsy_log = 0
last_yawn_log = 0
last_distract_log = 0
last_phone_log = 0

COOLDOWN = 5.0

print("[INFO] Starting video stream for driving session...")
cap = cv2.VideoCapture(0)
time.sleep(2.0)

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("[WARNING] Empty frame received from webcam. Retrying...")
        time.sleep(0.1)
        continue

    frame = imutils.resize(frame, width=800)

    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        print(f"[ERROR] Could not convert frame to grayscale: {e}")
        continue

    size = gray.shape

    # -------- Object Detection --------
    person_detected, phone_detected, frame_yolo = detect_objects(frame.copy())

    display_frame = frame_yolo

    # -------- Face Detection --------
    rects = detector(gray, 0)

    alert_triggered = False

    if len(rects) > 0:
        cv2.putText(
            display_frame,
            "Face Detected",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

        for rect in rects:

            # -------- Facial landmarks --------
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # -------- Eye Detection --------
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            ear = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)

            cv2.drawContours(display_frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(display_frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # -------- Drowsiness Detection --------
            if ear < EYE_AR_THRESH:
                ear_counter += 1

                if ear_counter >= EYE_AR_CONSEC_FRAMES:
                    cv2.putText(
                        display_frame,
                        "DROWSINESS ALERT!",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

                    driver_status["status"] = "DROWSY"
                    driver_status["alert"] = "Eyes Closed Detected"

                    alert_triggered = True

                    if time.time() - last_drowsy_log > COOLDOWN:
                        metrics["drowsiness_alerts"] += 1
                        last_drowsy_log = time.time()

                        save_alert("DROWSY", display_frame)

            else:
                ear_counter = 0

            # -------- Mouth Detection --------
            mouth = shape[mStart:mEnd]
            mar = mouth_aspect_ratio(mouth)

            mouthHull = cv2.convexHull(mouth)
            cv2.drawContours(display_frame, [mouthHull], -1, (0, 255, 0), 1)

            # -------- Yawning Detection --------
            if mar > MOUTH_AR_THRESH:
                yawn_counter += 1

                if yawn_counter >= MOUTH_AR_CONSEC_FRAMES:
                    cv2.putText(
                        display_frame,
                        "YAWNING ALERT!",
                        (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

                    driver_status["status"] = "YAWNING"
                    driver_status["alert"] = "Driver Yawning"

                    alert_triggered = True

                    if time.time() - last_yawn_log > COOLDOWN:
                        metrics["yawning_alerts"] += 1
                        last_yawn_log = time.time()

                        save_alert("YAWNING", display_frame)

            else:
                yawn_counter = 0

            # -------- Head Pose Detection --------
            image_points = np.array([
                shape[33],
                shape[8],
                shape[36],
                shape[45],
                shape[48],
                shape[54]
            ], dtype="double")

            for p in image_points:
                cv2.circle(
                    display_frame,
                    (int(p[0]), int(p[1])),
                    3,
                    (0, 0, 255),
                    -1
                )

            head_tilt_degree, sp, ep, ep_alt = get_head_pose(
                size,
                image_points,
                frame.shape[0]
            )

            cv2.line(display_frame, sp, ep, (255, 0, 0), 2)
            cv2.line(display_frame, sp, ep_alt, (0, 0, 255), 2)

            # -------- Distraction Detection --------
            if head_tilt_degree > HEAD_TILT_THRESH:
                tilt_counter += 1

                if tilt_counter >= HEAD_TILT_CONSEC_FRAMES:
                    cv2.putText(
                        display_frame,
                        "DISTRACTION ALERT! (Head Tilt)",
                        (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

                    driver_status["status"] = "DISTRACTED"
                    driver_status["alert"] = "Head Tilt / Distraction"

                    alert_triggered = True

                    if time.time() - last_distract_log > COOLDOWN:
                        metrics["distraction_alerts"] += 1
                        last_distract_log = time.time()

                        save_alert("DISTRACTED", display_frame)

            else:
                tilt_counter = 0

            # -------- Stats Overlay --------
            tilt_val = (
                head_tilt_degree[0]
                if isinstance(head_tilt_degree, (list, np.ndarray))
                else head_tilt_degree
            )

            cv2.putText(
                display_frame,
                f"EAR: {ear:.2f}",
                (600, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            cv2.putText(
                display_frame,
                f"MAR: {mar:.2f}",
                (600, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            cv2.putText(
                display_frame,
                f"TILT: {tilt_val:.2f}",
                (600, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

    # -------- Phone Detection --------
    if phone_detected:
        cv2.putText(
            display_frame,
            "PHONE USAGE ALERT!",
            (10, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        driver_status["status"] = "PHONE DETECTED"
        driver_status["alert"] = "Phone Usage Detected"

        alert_triggered = True

        if time.time() - last_phone_log > COOLDOWN:
            metrics["phone_usage_alerts"] += 1
            last_phone_log = time.time()

            save_alert("PHONE", display_frame)

    # -------- Safe Status --------
    if not alert_triggered:
        driver_status["status"] = "SAFE"
        driver_status["alert"] = "Driver Normal"

    # -------- Alarm --------
    if alert_triggered:
        play_alarm()
    else:
        stop_alarm()

    # -------- Display --------
    cv2.imshow("Continuous Driver Session Monitoring", display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# -------- Session Report --------
session_end_time = time.time()
session_duration = session_end_time - session_start_time

print("\n" + "=" * 40)
print(" DRIVING SESSION COMPLETED ")
print("=" * 40)
print(f"Total Session Duration: {session_duration:.2f} seconds")
print("--- Infraction Summary ---")
print(f"Drowsiness Instances: {metrics['drowsiness_alerts']}")
print(f"Yawning Instances:    {metrics['yawning_alerts']}")
print(f"Distraction Events:   {metrics['distraction_alerts']}")
print(f"Phone Usage Events:   {metrics['phone_usage_alerts']}")
print("=" * 40)

cv2.destroyAllWindows()
cap.release()