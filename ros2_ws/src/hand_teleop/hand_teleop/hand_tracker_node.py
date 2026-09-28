"""웹캠 → MediaPipe → 하이브리드 리타게팅 → /hand_joint_states 발행."""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as Rot

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.constants import (
    RobotName, RetargetingType, HandType, get_default_config_path)

BASE = Path.home() / "hand-retarget"
ASSETS = BASE / "dex-retargeting-repo/assets/robots/hands"
OP2MANO_R = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])

INDEP = ["index_proximal_joint", "middle_proximal_joint", "ring_proximal_joint",
         "pinky_proximal_joint", "thumb_proximal_yaw_joint",
         "thumb_proximal_pitch_joint"]

EMA_LAMBDA = 0.3   # 과거값 가중치. scripts/curve.py 의 ema(x, lam) 와 같은 규약 (곡선: 지연 0, jitter 0.056)
CALIB_N     = 90       # 시작 시 손 편 상태로 유지할 프레임 수 (3초)
CALIB_MOVE  = 0.08     # 이 이상 흔들리면 카운터 리셋
ROT_MAX_DEG = 45.0     # 한 프레임 최대 회전. 초과분은 일단 버림
ROT_REJECT_MAX = 2     # 연속 거부가 이만큼이면 진짜 움직임으로 보고 즉시 추종
QMAX        = np.array([1.470, 1.470, 1.470, 1.470, 1.308, 0.600])
CALIB_SKIP  = 4        # thumb_yaw 는 q_open≈0 이라 캘리브레이션 제외
ROT_LAMBDA = 0.7        # 방향 전용 스무딩(과거값 가중치). 관절각과 별개
GAP_LIMIT = 40           # 관측 갭 최대치. 초과 시 안전 정지


def hand_frame(kp):
    pts = kp[[0, 5, 9], :]
    xv = pts[0] - pts[2]
    pts = pts - pts.mean(0, keepdims=True)
    _, _, v = np.linalg.svd(pts)
    n = v[2, :]
    x = xv - np.dot(xv, n) * n
    x /= np.linalg.norm(x)
    z = np.cross(x, n)
    if np.dot(z, pts[1] - pts[2]) < 0:
        n, z = -n, -z
    return np.stack([x, n, z], axis=1)


class HandTracker(Node):
    def __init__(self):
        super().__init__("hand_tracker")
        self.pub = self.create_publisher(JointState, "/hand_joint_states", 10)
        self.pub_pose = self.create_publisher(PoseStamped, "/hand_pose", 10)
        self.F0 = np.load(BASE / "configs/F0.npy").astype(np.float64)
        self.quat = None      # 기준 자세 대비 손 회전 Δ (x, y, z, w)
        self.rjct = 0         # 연속 거부 횟수
        qf = BASE / "configs/q_open.npy"
        recal = "--recalib" in sys.argv
        self.q_open = None if recal or not qf.exists() else np.load(qf)
        self.cbuf = []        # 캘리브레이션 수집 버퍼

        ck = torch.load(BASE / "models/policy.pt", weights_only=False)
        self.K, self.H = ck["K"], ck["H"]
        self.xmu, self.xsd, self.dsd = ck["xmu"], ck["xsd"], ck["dsd"]
        a = ck["arch"]
        self.policy = nn.Sequential(
            nn.Linear(a[0], a[1]), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(a[1], a[2]), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(a[2], a[3]))
        self.policy.load_state_dict(ck["state_dict"])
        self.policy.eval()

        RetargetingConfig.set_default_urdf_dir(str(ASSETS))
        self.rt = RetargetingConfig.load_from_file(
            str(BASE / "configs/inspire_right_nofilter.yml")).build()
        self.hidx = np.array(self.rt.optimizer.target_link_human_indices)
        self.jsel = [self.rt.joint_names.index(n) for n in INDEP]

        self.lm = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=str(BASE / "models/hand_landmarker.task")),
                num_hands=1, running_mode=vision.RunningMode.VIDEO))

        self.cap = cv2.VideoCapture(0)
        self.buf = []                 # 최근 K개 특징
        self.q = np.zeros(6)          # 발행 중인 관절각
        self.anchor = np.zeros(6)
        self.delta = None
        self.gap = 0
        self.prev_t = time.time()
        self.timer = self.create_timer(1.0 / 30.0, self.tick)
        self.get_logger().info("hand_tracker 시작 — q 또는 ESC 로 종료")

    def tick(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.lm.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
            int(time.time() * 1000))
        h, w, _ = frame.shape
        tracked = bool(res.hand_world_landmarks)

        if tracked:
            wl = np.array([[p.x, p.y, p.z] for p in res.hand_world_landmarks[0]],
                          dtype=np.float32)
            if res.handedness[0][0].category_name == "Left":
                wl = wl.copy(); wl[:, 0] *= -1.0
            F = hand_frame(wl)
            jp = wl @ F @ OP2MANO_R

            d = Rot.from_matrix(self.F0.T @ F.astype(np.float64)).as_quat()
            if self.quat is None:
                self.quat = d
            else:
                if np.dot(self.quat, d) < 0.0:
                    d = -d                      # q 와 -q 는 같은 회전 — 부호 정렬
                dot = abs(float(np.dot(self.quat, d)))
                deg = np.degrees(2 * np.arccos(min(dot, 1.0)))
                if deg <= ROT_MAX_DEG:
                    self.quat = ROT_LAMBDA * self.quat + (1 - ROT_LAMBDA) * d
                    self.quat /= np.linalg.norm(self.quat)
                    self.rjct = 0
                else:                           # 1프레임 스파이크만 버린다
                    self.rjct += 1
                    if self.rjct >= ROT_REJECT_MAX:
                        self.quat = d.copy()    # 실제 빠른 움직임 — 즉시 추종
                        self.rjct = 0

            ref = jp[self.hidx[1, :], :] - jp[self.hidx[0, :], :]
            q_opt = np.asarray(self.rt.retarget(ref))[self.jsel]
            self.q = EMA_LAMBDA * self.q + (1 - EMA_LAMBDA) * q_opt

            if self.q_open is None:
                if self.cbuf and np.abs(self.q - self.cbuf[-1]).max() > CALIB_MOVE:
                    self.cbuf = []              # 움직이면 처음부터
                self.cbuf.append(self.q.copy())
                if len(self.cbuf) >= CALIB_N:
                    self.q_open = np.median(np.array(self.cbuf), axis=0)
                    self.q_open[CALIB_SKIP] = 0.0
                    np.save(BASE / 'configs/q_open.npy', self.q_open)
                    self.get_logger().info(
                        'q_open = ' + np.array_str(self.q_open, precision=3))

            self.buf.append(jp.ravel())
            self.buf = self.buf[-self.K:]
            self.anchor, self.delta, self.gap = self.q.copy(), None, 0
            mode, col = "OPTIMIZER", (0, 255, 0)

            for p in res.hand_landmarks[0]:
                cv2.circle(frame, (int(p.x * w), int(p.y * h)), 3, (0, 255, 0), -1)
        else:
            self.gap += 1
            if self.delta is None and len(self.buf) == self.K:
                x = ((np.array(self.buf) - self.xmu) / self.xsd).ravel()
                with torch.no_grad():
                    d = self.policy(torch.tensor(x[None], dtype=torch.float32))
                d = d.numpy().reshape(self.H, 6) * self.dsd
                self.delta = d - d[0]
            if self.delta is not None and self.gap <= GAP_LIMIT:
                self.q = self.anchor + self.delta[min(self.gap - 1, self.H - 1)]
                mode, col = f"POLICY  gap {self.gap}", (0, 200, 255)
            else:
                mode, col = "HOLD (timeout)", (0, 0, 255)

        if self.q_open is None:
            q_out = self.q
            mode = f'CALIB  {len(self.cbuf)}/{CALIB_N}  손 펴고 유지'
            col = (255, 200, 0)
        else:                       # 분기 이후 한 곳에서만 — 정책 재학습 불필요
            den = np.maximum(QMAX - self.q_open, 1e-3)
            q_out = np.clip((self.q - self.q_open) / den * QMAX, 0.0, QMAX)
            q_out[CALIB_SKIP] = self.q[CALIB_SKIP]

        m = JointState()
        m.header.stamp = self.get_clock().now().to_msg()
        m.name = INDEP
        m.position = [float(v) for v in q_out]
        self.pub.publish(m)

        if self.quat is not None:      # 갭 구간에서는 마지막 방향 유지
            ps = PoseStamped()
            ps.header.stamp = m.header.stamp
            ps.header.frame_id = 'hand_ref'
            ps.pose.orientation.x = float(self.quat[0])
            ps.pose.orientation.y = float(self.quat[1])
            ps.pose.orientation.z = float(self.quat[2])
            ps.pose.orientation.w = float(self.quat[3])
            self.pub_pose.publish(ps)

        now = time.time()
        fps = 1.0 / max(now - self.prev_t, 1e-6)
        self.prev_t = now
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (w, 78), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, mode, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
        cv2.putText(frame, f"{fps:4.1f} fps   " +
                    "  ".join(f"{v:5.2f}" for v in q_out),
                    (14, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        cv2.imshow("hand tracker", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            raise KeyboardInterrupt


def main():
    rclpy.init()
    node = HandTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
