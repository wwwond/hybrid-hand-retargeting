import numpy as np, pybullet as p, pybullet_data, argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("session")
args = ap.parse_args()

URDF = Path.home() / "hand-retarget/dex-retargeting-repo/assets/robots/hands/inspire_hand/inspire_hand_right_obj.urdf"
d = np.load(Path("data/raw") / args.session / "labels.npz", allow_pickle=True)
qpos, names = d["qpos"], [str(x) for x in d["joint_names"]]
task, phase = d["task"].astype(str), d["phase"].astype(str)

TASKS = ["open", "large_diameter", "small_diameter", "tip_pinch",
         "tripod", "lateral", "index_extension"]

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
hand = p.loadURDF(str(URDF), basePosition=[0, 0, 0], useFixedBase=True)
jmap = {p.getJointInfo(hand, i)[1].decode(): i for i in range(p.getNumJoints(hand))}
pairs = [(jmap[n], k) for k, n in enumerate(names) if n in jmap]

proj = p.computeProjectionMatrixFOV(fov=45, aspect=1.0, nearVal=0.01, farVal=2.0)
VIEWS = [("front", 90, -10), ("side", 0, -10)]

fig, axes = plt.subplots(len(VIEWS), len(TASKS), figsize=(3 * len(TASKS), 3 * len(VIEWS)))
for col, t in enumerate(TASKS):
    m = (task == t) & (phase == "hold")
    if m.sum() == 0:
        for row in range(len(VIEWS)):
            axes[row, col].axis("off")
        continue
    # 해당 태스크 hold 구간의 중앙값 자세 (대표 프레임)
    q = np.median(qpos[m], axis=0)
    for j, k in pairs:
        p.resetJointState(hand, j, float(q[k]))

    for row, (vname, yaw, pitch) in enumerate(VIEWS):
        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0, 0, 0.08], distance=0.30,
            yaw=yaw, pitch=pitch, roll=0, upAxisIndex=2)
        _, _, rgb, _, _ = p.getCameraImage(400, 400, view, proj,
                                            renderer=p.ER_TINY_RENDERER)
        img = np.reshape(rgb, (400, 400, 4))[:, :, :3]
        ax = axes[row, col]
        ax.imshow(img)
        ax.axis("off")
        if row == 0:
            ax.set_title(t, fontsize=11)
        if col == 0:
            ax.text(-0.1, 0.5, vname, transform=ax.transAxes,
                    rotation=90, va="center", fontsize=10)

p.disconnect()
plt.tight_layout()
out = Path("data") / f"task_grid_{args.session}.png"
plt.savefig(out, dpi=110, bbox_inches="tight")
print(f"saved -> {out}")
