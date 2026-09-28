import json
import numpy as np
from pathlib import Path
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.constants import (
    RobotName, RetargetingType, HandType, get_default_config_path
)

ASSETS = Path.home() / "hand-retarget/dex-retargeting-repo/assets/robots/hands"
SESSION = Path.home() / "hand-retarget/data/raw/task_left_1790408204"
TASKS = ["open", "large_diameter", "small_diameter", "tip_pinch",
         "tripod", "lateral", "index_extension"]

RetargetingConfig.set_default_urdf_dir(str(ASSETS))
cfg = get_default_config_path(RobotName.inspire, RetargetingType.vector, HandType.right)

# 태스크별 중앙 프레임 하나씩 수집
buckets = {t: [] for t in TASKS}
for line in open(SESSION / "landmarks.jsonl"):
    r = json.loads(line)
    if r["phase"] == "hold" and r["task"] in buckets:
        buckets[r["task"]].append(r["landmarks"])

INDEP = ["index_proximal_joint", "middle_proximal_joint", "ring_proximal_joint",
         "pinky_proximal_joint", "thumb_proximal_yaw_joint", "thumb_proximal_pitch_joint"]

results = {}
for task in TASKS:
    frames = buckets[task]
    if not frames:
        continue
    # 매번 새 retargeting 객체 (시간 평활 항의 영향 제거)
    rt = RetargetingConfig.load_from_file(cfg).build()
    idx = np.array(rt.optimizer.target_link_human_indices)
    mid = np.array(frames[len(frames) // 2], dtype=np.float32)
    ref = mid[idx[1, :], :] - mid[idx[0, :], :]
    q = rt.retarget(ref)
    results[task] = dict(zip(rt.joint_names, q))

names = rt.joint_names
print(f"{'joint':30s}" + "".join(f"{t[:9]:>10s}" for t in results))
print("-" * (30 + 10 * len(results)))
for n in INDEP:
    row = f"{n:30s}"
    for t in results:
        row += f"{results[t][n]:10.3f}"
    print(row)

print("\n=== joint limits (URDF) ===")
import pybullet as p
p.connect(p.DIRECT)
hand = p.loadURDF(str(ASSETS / "inspire_hand/inspire_hand_right_obj.urdf"), useFixedBase=True)
for i in range(p.getNumJoints(hand)):
    info = p.getJointInfo(hand, i)
    nm = info[1].decode()
    if nm in INDEP:
        print(f"  {nm:30s} [{info[8]:.3f}, {info[9]:.3f}]")
p.disconnect()

np.save(Path.home() / "hand-retarget/data/probe_qpos_all.npy",
        {t: results[t] for t in results}, allow_pickle=True)
print("\nsaved -> data/probe_qpos_all.npy")
