import cv2, time
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

opts = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="../../models/hand_landmarker.task"),
    num_hands=1,
    running_mode=vision.RunningMode.VIDEO,
)
lm = vision.HandLandmarker.create_from_options(opts)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    flipped = cv2.flip(frame, 1)          # collect_task.py와 동일
    for name, img in [("FLIPPED (as collected)", flipped), ("RAW", frame)]:
        pass
    rgb = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                              int(time.time()*1000))
    txt = "no hand"
    if res.handedness:
        c = res.handedness[0][0]
        txt = f"MediaPipe says: {c.category_name}  ({c.score:.2f})"
        for p in res.hand_landmarks[0]:
            cv2.circle(flipped, (int(p.x*flipped.shape[1]), int(p.y*flipped.shape[0])), 4, (0,255,0), -1)
    cv2.putText(flipped, txt, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
    cv2.imshow("handedness check", flipped)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release(); cv2.destroyAllWindows(); lm.close()
