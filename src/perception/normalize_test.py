import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import time
import numpy as np
from normalize import normalize_landmarks

MODEL_PATH = "../../models/hand_landmarker.task"

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.VIDEO,
)
landmarker = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    timestamp_ms = int(time.time() * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    h, w, _ = frame.shape
    status = "no hand detected"

    if result.hand_landmarks:
        hand_landmarks = result.hand_landmarks[0]
        norm = normalize_landmarks(hand_landmarks)  # (21, 3)

        # 화면 좌표(원래 x,y)에 초록 점 오버레이
        for lm in hand_landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        wrist_dist = np.linalg.norm(norm[0])          # 항상 0이어야 함
        mcp_dist = np.linalg.norm(norm[9])             # 항상 1이어야 함
        status = f"wrist={wrist_dist:.3f}  mcp9={mcp_dist:.3f}"

    now = time.time()
    fps = 1.0 / (now - prev_time) if now != prev_time else 0
    prev_time = now

    cv2.putText(frame, f"FPS: {fps:.1f}  {status}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Normalize Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()
