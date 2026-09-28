import json, time, argparse
import numpy as np
from pathlib import Path
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.constants import (
    RobotName, RetargetingType, HandType, get_default_config_path
)

OPERATOR2MANO_RIGHT = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
OPERATOR2MANO_LEFT  = np.array([[0, 0, -1], [ 1, 0, 0], [0, -1, 0]])

INDEP = ["index_proximal_joint", "middle_proximal_joint", "ring_proximal_joint",
         "pinky_proximal_joint", "thumb_proximal_yaw_joint", "thumb_proximal_pitch_joint"]


def estimate_frame_from_hand_points(kp):
    assert kp.shape == (21, 3)
    points = kp[[0, 5, 9], :]
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

ap = argparse.ArgumentParser()
ap.add_argument("session")
ap.add_argument("--config", default=None, help="커스텀 yml 경로 (미지정 시 기본 teleop)")
ap.add_argument("--out", default="labels.npz")
args = ap.parse_args()

SESSION = Path("data/raw") / args.session
ASSETS = Path.home() / "hand-retarget/dex-retargeting-repo/assets/robots/hands"
RetargetingConfig.set_default_urdf_dir(str(ASSETS))
cfg = args.config if args.config else get_default_config_path(RobotName.inspire, RetargetingType.vector, HandType.right)
rt = RetargetingConfig.load_from_file(cfg).build()
idx = np.array(rt.optimizer.target_link_human_indices)
indep_i = [rt.joint_names.index(n) for n in INDEP]

records = [json.loads(l) for l in open(SESSION / "landmarks.jsonl")]
print(f"{args.session}: {len(records)} frames with landmarks")

qpos_all, meta_rows, latencies = [], [], []
t_start = time.time()

for i, r in enumerate(records):
    if "world_landmarks" not in r:
        raise SystemExit("world_landmarks 없음 — v1 세션입니다")
    wl = np.array(r["world_landmarks"], dtype=np.float32)
    if r.get("handedness") == "Left":          # 거울 반전으로 오른손 규약 통일
        wl = wl.copy(); wl[:, 0] *= -1.0
    joint_pos = wl @ estimate_frame_from_hand_points(wl) @ OPERATOR2MANO_RIGHT
    ref = joint_pos[idx[1, :], :] - joint_pos[idx[0, :], :]

    t0 = time.perf_counter()
    q = rt.retarget(ref)
    latencies.append((time.perf_counter() - t0) * 1000)

    qpos_all.append(q)
    meta_rows.append((r["frame_id"], r["ts"], r["task"], r["phase"], r["rep"]))

    if (i + 1) % 2000 == 0:
        print(f"  {i+1}/{len(records)}")

qpos_all = np.array(qpos_all, dtype=np.float32)
lat = np.array(latencies)
print(f"\ndone in {time.time()-t_start:.1f}s")
print(f"retarget latency: mean={lat.mean():.2f}ms  p50={np.percentile(lat,50):.2f}  "
      f"p95={np.percentile(lat,95):.2f}  p99={np.percentile(lat,99):.2f}  max={lat.max():.2f}")

out = SESSION / args.out
np.savez_compressed(
    out,
    qpos=qpos_all,
    qpos_indep=qpos_all[:, indep_i],
    joint_names=np.array(rt.joint_names),
    indep_names=np.array(INDEP),
    frame_id=np.array([m[0] for m in meta_rows]),
    ts=np.array([m[1] for m in meta_rows]),
    task=np.array([m[2] for m in meta_rows]),
    phase=np.array([m[3] for m in meta_rows]),
    rep=np.array([m[4] for m in meta_rows]),
    latency_ms=lat,
)
print(f"saved -> {out}")

# 태스크별 독립 관절 통계 (hold 구간만)
tasks = np.array([m[2] for m in meta_rows])
phases = np.array([m[3] for m in meta_rows])
qi = qpos_all[:, indep_i]
print(f"\n{'task':17s}" + "".join(f"{n.split('_')[0][:5]:>8s}" for n in INDEP) + "     n")
print("-" * 80)
for t in ["open", "large_diameter", "small_diameter", "tip_pinch",
          "tripod", "lateral", "index_extension"]:
    m = (tasks == t) & (phases == "hold")
    if m.sum() == 0:
        continue
    mean = qi[m].mean(0)
    print(f"{t:17s}" + "".join(f"{v:8.2f}" for v in mean) + f"  {m.sum():6d}")
print("\n--- std (재현성; 작을수록 일관됨) ---")
for t in ["open", "large_diameter", "small_diameter", "tip_pinch",
          "tripod", "lateral", "index_extension"]:
    m = (tasks == t) & (phases == "hold")
    if m.sum() == 0:
        continue
    print(f"{t:17s}" + "".join(f"{v:8.2f}" for v in qi[m].std(0)))
