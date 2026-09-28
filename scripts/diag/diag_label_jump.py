#!/usr/bin/env python3
# 랜드마크 급변(θ>20°)이 최종 관절각 라벨에 얼마나 전달되는지 측정
import json, sys, numpy as np

def estimate_frame(kp):  # 인수인계 문서 4.2 참조 구현 그대로 — 수정 금지
    points = kp[[0, 5, 9], :]
    x_vector = points[0] - points[2]
    points = points - np.mean(points, axis=0, keepdims=True)
    u, s, v = np.linalg.svd(points)
    normal = v[2, :]
    x = x_vector - np.sum(x_vector * normal) * normal
    x = x / np.linalg.norm(x)
    z = np.cross(x, normal)
    if np.sum(z * (points[1] - points[2])) < 0:
        normal *= -1; z *= -1
    return np.stack([x, normal, z], axis=1)

sess = sys.argv[1]
base = f"data/raw/{sess}"
z = np.load(f"{base}/labels_nofilter.npz", allow_pickle=True)
Q = z["qpos_indep"].astype(np.float64)
fid_q, phase = z["frame_id"], z["phase"]
pos = {int(f): i for i, f in enumerate(fid_q)}

# 랜드마크에서 프레임별 좌표계 회전각 계산
prevF, prevfid, theta = None, None, {}
for line in open(f"{base}/landmarks.jsonl"):
    r = json.loads(line)
    wl = r.get("world_landmarks")
    if not wl:
        prevF = None; continue
    w = np.asarray(wl, dtype=np.float64).reshape(21, 3)
    if r.get("handedness") == "Left": w = w.copy(); w[:, 0] *= -1
    F, fid = estimate_frame(w), r["frame_id"]
    if prevF is not None and fid == prevfid + 1:
        c = np.clip((np.trace(prevF.T @ F) - 1) / 2, -1, 1)
        theta[fid] = np.degrees(np.arccos(c))
    prevF, prevfid = F, fid

# 라벨 인덱스 기준으로 정렬
N = len(Q)
th = np.full(N, np.nan)
for f, t in theta.items():
    if f in pos: th[pos[f]] = t
cont = np.zeros(N, bool)          # i-1 → i 가 연속 프레임인가
cont[1:] = (fid_q[1:] - fid_q[:-1]) == 1
dQ = np.zeros((N, 6)); dQ[1:] = Q[1:] - Q[:-1]
nrm = np.linalg.norm(dQ, axis=1)

J = cont & (th > 20.0)            # 급변
OK = cont & (th <= 20.0)          # 평상시
hold = phase == "hold"
Jh = J & hold

# 되돌아옴 판정
rev = 0
for i in np.where(Jh)[0]:
    if i + 1 < N and cont[i + 1] and nrm[i] > 1e-9:
        if np.dot(dQ[i], dQ[i + 1]) < 0 and nrm[i + 1] > 0.5 * nrm[i]:
            rev += 1

mo, mj = np.median(nrm[OK]), np.median(nrm[Jh]) if Jh.sum() else np.nan
print(f"\n[{sess}]  라벨 {N}  hold {hold.sum()}")
print(f"  급변 {J.sum()} ({J.sum()/max(cont.sum(),1):.2%})   그중 hold 구간 {Jh.sum()} ({Jh.sum()/max(hold.sum(),1):.2%} of hold)")
print(f"  |Δq| 중앙  평상시 {mo:.4f}   급변(hold) {mj:.4f}   배율 {mj/mo:.1f}x")
print(f"  |Δq| p99(평상시) {np.percentile(nrm[OK],99):.4f}   급변(hold) 최대 {nrm[Jh].max() if Jh.sum() else float('nan'):.4f}")
print(f"  되돌아옴 {rev}/{Jh.sum()} ({rev/max(Jh.sum(),1):.1%})   → 높으면 노이즈 확정")
