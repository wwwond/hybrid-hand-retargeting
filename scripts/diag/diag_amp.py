#!/usr/bin/env python3
# 튐이 입력 노이즈인가(충실한 추종) 최적화기 증폭인가 + 지속 길이 분포
import json, sys, numpy as np
from collections import Counter

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

sess = sys.argv[1]; base = f"data/raw/{sess}"
z = np.load(f"{base}/labels_nofilter.npz", allow_pickle=True)
Q, fid_q, phase = z["qpos_indep"].astype(np.float64), z["frame_id"], z["phase"]
task = z["task"]; pos = {int(f): i for i, f in enumerate(fid_q)}

JP = np.full((len(Q), 21, 3), np.nan)
for line in open(f"{base}/landmarks.jsonl"):
    r = json.loads(line); wl = r.get("world_landmarks")
    if not wl: continue
    i = pos.get(r["frame_id"])
    if i is None: continue
    w = np.asarray(wl, dtype=np.float64).reshape(21, 3)
    if r.get("handedness") == "Left": w = w.copy(); w[:, 0] *= -1
    JP[i] = w @ estimate_frame(w)

N = len(Q)
cont = np.zeros(N, bool); cont[1:] = (fid_q[1:] - fid_q[:-1]) == 1
cont &= ~np.isnan(JP).any(axis=(1, 2)) & np.r_[False, ~np.isnan(JP[:-1]).any(axis=(1, 2))]
hold = phase == "hold"
dq = np.zeros(N); dq[1:] = np.linalg.norm(Q[1:] - Q[:-1], axis=1)
dj = np.zeros(N); dj[1:] = np.linalg.norm((JP[1:] - JP[:-1]).reshape(N-1, -1), axis=1)

M = cont & hold
S = M & (dq > 0.3)                      # 튐
base_ratio = np.median(dq[M & ~S] / np.maximum(dj[M & ~S], 1e-9))
print(f"\n[{sess}]  hold 연속 {M.sum()}  튐 {S.sum()} ({S.sum()/max(M.sum(),1):.2%})")
print(f"  입력 |Δjp| (m)  평상시 중앙 {np.median(dj[M & ~S]):.4f}   튐 시 중앙 {np.median(dj[S]):.4f}"
      f"   배율 {np.median(dj[S])/max(np.median(dj[M & ~S]),1e-9):.1f}x")
print(f"  증폭비 |Δq|/|Δjp|  평상시 중앙 {base_ratio:.1f}   튐 시 중앙 {np.median(dq[S]/np.maximum(dj[S],1e-9)):.1f}"
      f"   → 튐 쪽이 훨씬 크면 최적화기 증폭")
print(f"  튐 중 입력이 조용한 경우(|Δjp| < 평상시 p75) {np.sum(dj[S] < np.percentile(dj[M & ~S], 75))} "
      f"({np.mean(dj[S] < np.percentile(dj[M & ~S], 75)):.1%})")
runs, c = Counter(), 0
for i in range(N):
    if S[i]: c += 1
    elif c: runs[min(c, 6)] += 1; c = 0
if c: runs[min(c, 6)] += 1
print("  지속 길이  " + "  ".join(f"{k}{'+' if k==6 else ''}f:{runs[k]}" for k in sorted(runs)))
tt = Counter(); ts = Counter()
for i in np.where(M)[0]:
    tt[str(task[i])] += 1
    if S[i]: ts[str(task[i])] += 1
print("  동작별 튐률  " + "   ".join(f"{k} {ts[k]/tt[k]:.1%}" for k in sorted(tt)))
