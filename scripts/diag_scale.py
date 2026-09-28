import cv2, time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

opts = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="models/hand_landmarker.task"),
    num_hands=1, running_mode=vision.RunningMode.VIDEO,
)
lm = vision.HandLandmarker.create_from_options(opts)
cap = cv2.VideoCapture(0)
n = 0
while n < 60:
    ret, frame = cap.read()
    if not ret: break
    rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                              int(time.time()*1000))
    if res.hand_world_landmarks:
        wl = np.array([[p.x, p.y, p.z] for p in res.hand_world_landmarks[0]])
        n += 1
        if n % 20 == 0:
            d = np.linalg.norm(wl[[4,8,12,16,20]] - wl[0], axis=1)
            print(f"wrist->tip distances (m): {np.round(d,4)}")
            print(f"  world bbox: {np.round(wl.min(0),3)} .. {np.round(wl.max(0),3)}")
cap.release(); lm.close()
