#!/usr/bin/env python3
# 기준 자세 F0 측정: 손바닥을 카메라로, 손가락을 위로, 손을 편 상태로 3초 유지
import time, numpy as np, cv2
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw, ImageFont

_F = ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
      "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]
KF = None
for f in _F:
    if Path(f).exists():
        KF = ImageFont.truetype(f, 22); break
def put_kr(img, text, org, color=(255, 255, 255)):
    if KF is None:
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1); return img
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil).text(org, text, font=KF, fill=color)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def estimate_frame(kp):  # 인수인계 문서 4.2 참조 구현 그대로 — 수정 금지
    points = kp[[0, 5, 9], :]
    x_vector = points[0] - points[2]
    points = points - np.mean(points, axis=0, keepdims=True)
    u, s, v = np.linalg.svd(points)
    normal = v[2, :]
    x = x_vector - np.sum(x_vector * normal) * normal
    x = x / np.linalg.norm(x)
    z = np.cross(x, normal)
    if np.sum(z * (points[1] - points[2])) < 0:
        normal *= -1; z *= -1
    return np.stack([x, normal, z], axis=1)

opts = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="models/hand_landmarker.task"),
    num_hands=1, running_mode=vision.RunningMode.VIDEO)
lm = vision.HandLandmarker.create_from_options(opts)
cap = cv2.VideoCapture(0)
if not cap.isOpened(): raise SystemExit("웹캠 열기 실패 — pkill 후 다시 시도")

buf, HOLD, t0 = [], 90, None
print("손바닥을 카메라로, 손가락을 위로, 손을 편 상태로 유지하세요. q=취소")
while True:
    ok, frame = cap.read()
    if not ok: break
    frame = cv2.flip(frame, 1)
    mpi = mp.Image(image_format=mp.ImageFormat.SRGB,
                   data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    res = lm.detect_for_video(mpi, int(time.time() * 1000))
    msg, col = "손을 보여주세요", (0, 0, 255)
    if res.hand_world_landmarks:
        w = np.array([[p.x, p.y, p.z] for p in res.hand_world_landmarks[0]], dtype=np.float64)
        h = res.handedness[0][0].category_name
        if h == "Left": w = w.copy(); w[:, 0] *= -1
        F = estimate_frame(w)
        if buf and np.linalg.norm(F - buf[-1]) > 0.3:
            buf = []                      # 크게 움직이면 처음부터
        buf.append(F)
        if len(buf) > HOLD: buf.pop(0)
        msg, col = f"유지 중  {len(buf)}/{HOLD}", (0, 200, 255)
        if len(buf) == HOLD:
            msg, col = "완료", (0, 255, 0)
    else:
        buf = []
    frame = put_kr(frame, msg, (20, 30), col)
    frame = put_kr(frame, "손바닥을 카메라로 · 손가락 위로 · 손 펴기", (20, 65))
    cv2.imshow("calib F0", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'): buf = []; break
    if len(buf) == HOLD:
        cv2.waitKey(400); break
cap.release(); cv2.destroyAllWindows()

if len(buf) < HOLD: raise SystemExit("취소됨 — 저장하지 않음")
A = np.mean(buf, axis=0)                  # 평균 후 SVD로 가장 가까운 정규직교 행렬
U, S, Vt = np.linalg.svd(A)
F0 = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
spread = np.median([np.linalg.norm(F - F0) for F in buf])
np.save("configs/F0.npy", F0)
print("\nF0 =\n", np.array_str(F0, precision=4, suppress_small=True))
print(f"det = {np.linalg.det(F0):+.4f}  (반드시 +1)")
print(f"유지 중 흔들림 중앙값 {spread:.4f}  (0.15 이하 권장)")
print("저장: configs/F0.npy")
