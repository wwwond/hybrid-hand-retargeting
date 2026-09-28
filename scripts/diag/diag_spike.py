#!/usr/bin/env python3
# 큰 |Δq| 스파이크의 출처와, 중앙값/EMA 필터의 억제 효과 비교
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

def med3(Y):                      # 인과적 3탭 중앙값 (과거 2프레임만 사용)
    out = Y.copy()
    for t in range(2, len(Y)):
        out[t] = np.median(Y[t-2:t+1], axis=0)
    return out

def ema(Y, lam):                  # scripts/curve.py 와 동일 규약
    out = Y.copy()
    for t in range(1, len(Y)):
        out[t] = lam * out[t-1] + (1 - lam) * Y[t]
    return out

sess = sys.argv[1]; base = f"data/raw/{sess}"
z = np.load(f"{base}/labels_nofilter.npz", allow_pickle=True)
Q, fid_q, phase = z["qpos_indep"].astype(np.float64), z["frame_id"], z["phase"]
pos = {int(f): i for i, f in enumerate(fid_q)}

prevF, prevfid, theta = None, None, {}
for line in open(f"{base}/landmarks.jsonl"):
    r = json.loads(line); wl = r.get("world_landmarks")
    if not wl: prevF = None; continue
    w = np.asarray(wl, dtype=np.float64).reshape(21, 3)
    if r.get("handedness") == "Left": w = w.copy(); w[:, 0] *= -1
    F, fid = estimate_frame(w), r["frame_id"]
    if prevF is not None and fid == prevfid + 1:
        c = np.clip((np.trace(prevF.T @ F) - 1) / 2, -1, 1)
        theta[fid] = np.degrees(np.arccos(c))
    prevF, prevfid = F, fid

N = len(Q); th = np.full(N, np.nan)
for f, t in theta.items():
    if f in pos: th[pos[f]] = t
cont = np.zeros(N, bool); cont[1:] = (fid_q[1:] - fid_q[:-1]) == 1
hold = phase == "hold"
nrm = np.zeros(N); nrm[1:] = np.linalg.norm(Q[1:] - Q[:-1], axis=1)

# 연속 구간별로 필터 적용 (구간 경계를 넘어 섞지 않음)
segs, s0 = [], 0
for i in range(1, N + 1):
    if i == N or not cont[i]:
        if i - s0 >= 3: segs.append((s0, i))
        s0 = i
def seg_jitter(fn):
    v = []
    for a, b in segs:
        Yf = fn(Q[a:b]); d = np.linalg.norm(Yf[1:] - Yf[:-1], axis=1)
        m = hold[a+1:b]
        if m.sum(): v.append(d[m])
    return np.concatenate(v)

raw = seg_jitter(lambda Y: Y)
TH = 0.30
M = cont & hold & (nrm > TH)
J = M & (th > 20.0)
print(f"\n[{sess}]")
print(f"  hold 중 |Δq| > {TH} 스파이크 {M.sum()} ({M.sum()/max((cont&hold).sum(),1):.2%})")
print(f"    그중 좌표계 급변 동반 {J.sum()} ({J.sum()/max(M.sum(),1):.1%})   → 낮으면 급변은 주범 아님")
for name, fn in [("무필터", lambda Y: Y), ("EMA λ=0.3", lambda Y: ema(Y, 0.3)),
                 ("EMA λ=0.7", lambda Y: ema(Y, 0.7)), ("중앙값3", med3),
                 ("중앙값3+EMA0.3", lambda Y: ema(med3(Y), 0.3))]:
    d = seg_jitter(fn)
    print(f"    {name:16s} 중앙 {np.median(d):.4f}  p99 {np.percentile(d,99):.4f}  "
          f"최대 {d.max():.4f}  >{TH} 개수 {(d>TH).sum()}")
