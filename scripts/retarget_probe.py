import json
import numpy as np
from pathlib import Path
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.constants import (
    RobotName, RetargetingType, HandType, get_default_config_path
)

ASSETS = Path.home() / "hand-retarget/dex-retargeting-repo/assets/robots/hands"
SESSION = Path.home() / "hand-retarget/data/raw/task_left_1790408204"

RetargetingConfig.set_default_urdf_dir(str(ASSETS))
cfg_path = get_default_config_path(
    RobotName.inspire, RetargetingType.vector, HandType.right
)
print(f"config: {cfg_path}")

retargeting = RetargetingConfig.load_from_file(cfg_path).build()
opt = retargeting.optimizer

print(f"\nrobot joint names ({len(retargeting.joint_names)}):")
for i, n in enumerate(retargeting.joint_names):
    print(f"  [{i}] {n}")

idx = opt.target_link_human_indices
print(f"\ntarget_link_human_indices shape={np.array(idx).shape}")
print(np.array(idx))

# open 포즈 한 프레임 가져오기
frame = None
for line in open(SESSION / "landmarks.jsonl"):
    r = json.loads(line)
    if r["task"] == "open" and r["phase"] == "hold":
        frame = r
        break

print(f"\nframe: task={frame['task']} handedness={frame.get('handedness')}")
joint_pos = np.array(frame["landmarks"], dtype=np.float32)   # (21, 3)
print(f"landmarks shape: {joint_pos.shape}")

# vector retargeting 입력: 지정된 쌍의 상대 벡터
indices = np.array(idx)
origin_indices = indices[0, :]
task_indices = indices[1, :]
ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]

qpos = retargeting.retarget(ref_value)
print(f"\nqpos ({len(qpos)}):")
for n, q in zip(retargeting.joint_names, qpos):
    print(f"  {n:32s} {q: .4f}")

np.save(Path.home() / "hand-retarget/data/probe_qpos_open.npy", qpos)
print("\nsaved -> data/probe_qpos_open.npy")
