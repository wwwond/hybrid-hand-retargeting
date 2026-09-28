import cv2, time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from pathlib import Path
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.constants import (
    RobotName, RetargetingType, HandType, get_default_config_path
)

OPERATOR2MANO_RIGHT = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
OPERATOR2MANO_LEFT  = np.array([[0, 0, -1], [ 1, 0, 0], [0, -1, 0]])


def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
    """dex-retargeting 참조 구현 그대로"""
    assert keypoint_3d_array.shape == (21, 3)
    points = keypoint_3d_array[[0, 5, 9], :]

    x_vector = points[0] - points[2]

    points = points - np.mean(points, axis=0, keepdims=True)
    u, s, v = np.linalg.svd(points)
    normal = v[2, :]

    x = x_vector - np.sum(x_vector * normal) * normal
    x = x / np.linalg.norm(x)
    z = np.cross(x, normal)

    if np.sum(z * (points[1] - points[2])) < 0:
        normal *= -1
        z *= -1
    return np.stack([x, normal, z], axis=1)


ASSETS = Path.home() / "hand-retarget/dex-retargeting-repo/assets/robots/hands"
RetargetingConfig.set_default_urdf_dir(str(ASSETS))
cfg = get_default_config_path(RobotName.inspire, RetargetingType.vector, HandType.right)
rt = RetargetingConfig.load_from_file(cfg).build()
idx = np.array(rt.optimizer.target_link_human_indices)
INDEP = [rt.joint_names.index(n) for n in
         ["index_proximal_joint", "middle_proximal_joint", "ring_proximal_joint",
          "pinky_proximal_joint", "thumb_proximal_yaw_joint", "thumb_proximal_pitch_joint"]]
LABELS = ["idx", "mid", "rng", "pky", "thY", "thP"]

opts = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="models/hand_landmarker.task"),
    num_hands=1, running_mode=vision.RunningMode.VIDEO,
)
lm = vision.HandLandmarker.create_from_options(opts)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                              int(time.time() * 1000))

    line1, line2 = "no hand", ""
    if res.hand_world_landmarks:
        handed = res.handedness[0][0].category_name
        o2m = OPERATOR2MANO_RIGHT if handed == "Right" else OPERATOR2MANO_LEFT

        wl = np.array([[p.x, p.y, p.z] for p in res.hand_world_landmarks[0]],
                      dtype=np.float32)
        wrist_rot = estimate_frame_from_hand_points(wl)
        joint_pos = wl @ wrist_rot @ o2m

        ref = joint_pos[idx[1, :], :] - joint_pos[idx[0, :], :]
        q = rt.retarget(ref)

        line1 = "  ".join(f"{L}:{q[i]:5.2f}" for L, i in zip(LABELS, INDEP))
        line2 = f"{handed}  ref|mid|={np.linalg.norm(ref[2]):.3f}m"

        for p in res.hand_landmarks[0]:
            cv2.circle(frame, (int(p.x * frame.shape[1]), int(p.y * frame.shape[0])),
                       3, (0, 255, 0), -1)

    cv2.putText(frame, line1, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
    cv2.putText(frame, line2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 255), 1)
    cv2.imshow("live retarget", frame)
    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
        break

cap.release(); cv2.destroyAllWindows(); lm.close()
