import pybullet as p
import pybullet_data
import time
import os

# dex-retargeting-repo 안의 URDF 경로
URDF_PATH = os.path.expanduser(
    "~/hand-retarget/dex-retargeting-repo/assets/robots/hands/inspire_hand/inspire_hand_right_obj.urdf"
)

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
p.loadURDF("plane.urdf")
p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=[0,0,0.3])

hand_id = p.loadURDF(URDF_PATH, basePosition=[0, 0, 0.3], useFixedBase=True)

num_joints = p.getNumJoints(hand_id)
print(f"Loaded hand with {num_joints} joints:")
movable_joints = []
for i in range(num_joints):
    info = p.getJointInfo(hand_id, i)
    joint_name = info[1].decode('utf-8')
    joint_type = info[2]
    if joint_type != p.JOINT_FIXED:
        movable_joints.append(i)
    print(f"  [{i}] {joint_name}  type={joint_type}")

print(f"\nMovable joints: {len(movable_joints)}")

# 관절을 천천히 움직여서 확인 (사인파로 왔다갔다)
import math
t = 0
while True:
    t += 0.02
    for j in movable_joints:
        target = 0.5 + 0.5 * math.sin(t)
        p.setJointMotorControl2(hand_id, j, p.POSITION_CONTROL, targetPosition=target)
    p.stepSimulation()
    time.sleep(1.0 / 240.0)
