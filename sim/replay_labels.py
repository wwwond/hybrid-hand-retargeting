import numpy as np, pybullet as p, pybullet_data, time, argparse
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("session")
ap.add_argument("--speed", type=float, default=1.0)
args = ap.parse_args()

URDF = Path.home() / "hand-retarget/dex-retargeting-repo/assets/robots/hands/inspire_hand/inspire_hand_right_obj.urdf"
d = np.load(Path("data/raw") / args.session / "labels.npz", allow_pickle=True)
qpos, names = d["qpos"], [str(x) for x in d["joint_names"]]
task, phase = d["task"], d["phase"]

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.resetDebugVisualizerCamera(0.35, 50, -25, [0, 0, 0.3])
hand = p.loadURDF(str(URDF), basePosition=[0, 0, 0.3], useFixedBase=True)

jmap = {}
for i in range(p.getNumJoints(hand)):
    jmap[p.getJointInfo(hand, i)[1].decode()] = i
cols = [jmap[n] for n in names if n in jmap]
src = [k for k, n in enumerate(names) if n in jmap]

# hold 구간만 재생
mask = phase == "hold"
idxs = np.where(mask)[0]
print(f"replaying {len(idxs)} hold frames")

txt = None
for k in idxs:
    for j, s in zip(cols, src):
        p.resetJointState(hand, j, float(qpos[k, s]))
    if txt is not None:
        p.removeUserDebugItem(txt)
    txt = p.addUserDebugText(str(task[k]), [0, 0, 0.42], [1, 1, 1], 1.4)
    p.stepSimulation()
    time.sleep(1.0 / (30 * args.speed))

print("done")
input("Enter to quit...")
