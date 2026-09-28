"""/hand_joint_states 구독 → PyBullet Inspire Hand 제어."""
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as Rot

URDF = (Path.home() / "hand-retarget/dex-retargeting-repo/assets/robots/hands"
        / "inspire_hand/inspire_hand_right_obj.urdf")

# URDF mimic 관계 — PyBullet 은 mimic 태그를 자동 적용하지 않으므로 직접 반영.
# 배수는 PyBullet 실측(proximal 을 단독 구동했을 때 intermediate 추종값)으로 확인.
MIMIC = {
    "index_intermediate_joint":  ("index_proximal_joint",        1.030),
    "middle_intermediate_joint": ("middle_proximal_joint",       1.033),
    "ring_intermediate_joint":   ("ring_proximal_joint",         1.033),
    "pinky_intermediate_joint":  ("pinky_proximal_joint",        1.032),
    "thumb_intermediate_joint":  ("thumb_proximal_pitch_joint",  1.334),
    "thumb_distal_joint":        ("thumb_proximal_pitch_joint",  0.667),
}


class RobotHand(Node):
    def __init__(self):
        super().__init__("robot_hand")
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(0.32, 60, -20, [0, 0, 0.30])
        self.base_pos = [0, 0, 0.30]
        self.hand = p.loadURDF(str(URDF), basePosition=self.base_pos,
                               useFixedBase=True)
        self.S = self._sim_hand_frame()
        self.jmap = {p.getJointInfo(self.hand, i)[1].decode(): i
                     for i in range(p.getNumJoints(self.hand))}
        self.lo = {n: p.getJointInfo(self.hand, i)[8] for n, i in self.jmap.items()}
        self.hi = {n: p.getJointInfo(self.hand, i)[9] for n, i in self.jmap.items()}
        self.n = 0
        self.create_subscription(JointState, "/hand_joint_states", self.cb, 10)
        self.create_subscription(PoseStamped, "/hand_pose", self.cb_pose, 10)
        self.create_timer(1.0 / 120.0, lambda: p.stepSimulation())
        self.get_logger().info("robot_hand 시작 — /hand_joint_states 대기 중")

    def _sim_hand_frame(self):
        """URDF 기본 자세에서 사람 손 좌표계와 같은 의미의 축 3개를 실측."""
        lmap = {p.getJointInfo(self.hand, i)[12].decode(): i
                for i in range(p.getNumJoints(self.hand))}
        try:
            def lp(name):
                return np.array(p.getLinkState(self.hand, lmap[name])[4])
            x = np.array(self.base_pos) - lp('middle_tip')   # 손가락 → 손목
            x /= np.linalg.norm(x)
            s = lp('index_tip') - lp('ring_tip')             # 검지 쪽
            z = s - np.dot(s, x) * x
            z /= np.linalg.norm(z)
            y = np.cross(z, x)                              # 손바닥 법선
            S = np.stack([x, y, z], axis=1)
        except (KeyError, ZeroDivisionError, FloatingPointError) as e:
            self.get_logger().warn(f'손 좌표계 실측 실패({e}) — 베이스 회전 비활성')
            return None
        if abs(np.linalg.det(S) - 1.0) > 1e-3:
            self.get_logger().warn(f'S 가 정규직교가 아님 det={np.linalg.det(S):.4f}'
                                   ' — 베이스 회전 비활성')
            return None
        self.get_logger().info('시뮬 손 좌표계 S =\n' + np.array_str(S, precision=3))
        return S

    def cb_pose(self, msg):
        if self.S is None:
            return
        o = msg.pose.orientation
        D = Rot.from_quat([o.x, o.y, o.z, o.w]).as_matrix()
        Rb = self.S @ D @ self.S.T
        p.resetBasePositionAndOrientation(
            self.hand, self.base_pos, Rot.from_matrix(Rb).as_quat())

    def set_joint(self, name, val):
        i = self.jmap.get(name)
        if i is None:
            return
        v = float(np.clip(val, self.lo[name], self.hi[name]))
        p.resetJointState(self.hand, i, v)

    def cb(self, msg):
        q = dict(zip(msg.name, msg.position))
        for name, val in q.items():
            self.set_joint(name, val)
        for dep, (src, mult) in MIMIC.items():
            if src in q:
                self.set_joint(dep, q[src] * mult)
        self.n += 1
        if self.n % 150 == 0:
            self.get_logger().info(f"{self.n} 프레임 수신")


def main():
    rclpy.init()
    node = RobotHand()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        p.disconnect()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
