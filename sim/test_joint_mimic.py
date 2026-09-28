import pybullet as p
import pybullet_data
import time
import os

URDF_PATH = os.path.expanduser(
    "~/hand-retarget/dex-retargeting-repo/assets/robots/hands/inspire_hand/inspire_hand_right_obj.urdf"
)

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
p.loadURDF("plane.urdf")
p.resetDebugVisualizerCamera(cameraDistance=0.4, cameraYaw=50, cameraPitch=-35,
                              cameraTargetPosition=[0, 0, 0.3])

hand_id = p.loadURDF(URDF_PATH, basePosition=[0, 0, 0.3], useFixedBase=True)

num_joints = p.getNumJoints(hand_id)
movable = []
for i in range(num_joints):
    info = p.getJointInfo(hand_id, i)
    name = info[1].decode('utf-8')
    jtype = info[2]
    if jtype != p.JOINT_FIXED:
        movable.append((i, name))

print(f"Movable joints ({len(movable)}):")
for idx, name in movable:
    print(f"  [{idx}] {name}")

proximal_joints = [(idx, name) for idx, name in movable if "proximal" in name and "base" not in name]

print("\n--- Testing proximal joints one at a time ---")
for idx, name in proximal_joints:
    print(f"Moving: {name}")
    for step in range(120):
        target = 0.8 * (step / 120)
        p.setJointMotorControl2(hand_id, idx, p.POSITION_CONTROL, targetPosition=target)
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

    state = p.getJointState(hand_id, idx)
    print(f"  {name} position: {state[0]:.3f}")

    finger = name.split("_proximal")[0]
    for jidx, jname in movable:
        if jname.startswith(finger) and "intermediate" in jname:
            jstate = p.getJointState(hand_id, jidx)
            print(f"  -> {jname} followed to: {jstate[0]:.3f}")

    time.sleep(0.5)
    for step in range(120):
        target = 0.8 * (1 - step / 120)
        p.setJointMotorControl2(hand_id, idx, p.POSITION_CONTROL, targetPosition=target)
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

print("\nDone. Close the window or Ctrl+C to exit.")
while True:
    p.stepSimulation()
    time.sleep(1.0 / 240.0)
