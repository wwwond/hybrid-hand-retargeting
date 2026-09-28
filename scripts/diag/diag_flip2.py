#!/usr/bin/env python3
# 급변(θ>20°)의 종류 판정: 강체 회전 / 거울 반사 / 비강체
import json, sys, numpy as np
from collections import defaultdict

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

def kabsch_residuals(A, B):
    """A·R ≈ B 의 최적 잔차(rms, m). (적정 회전, 반사) 두 개를 반환."""
    A = A - A.mean(0); B = B - B.mean(0)
    U, S, Vt = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(U @ Vt))
    out = []
    for s in (d, -d):
        R = U @ np.diag([1.0, 1.0, s]) @ Vt
        out.append(np.sqrt(((A @ R - B) ** 2).sum(1).mean()))
    return out

TH_DEG = 20.0
path = sys.argv[1]
rows = [json.loads(l) for l in open(path)]
prev = None
recs = []   # (task, θ, d_hand, r_rot, r_ref)
for r in rows:
    wl = r.get("world_landmarks")
    if not wl:
        prev = None; continue
    w = np.asarray(wl, dtype=np.float64).reshape(21, 3)
    if r.get("handedness") == "Left": w = w.copy(); w[:, 0] *= -1   # 런타임과 동일
    F = estimate_frame(w)
    cur = dict(fid=r["frame_id"], task=r.get("task"), w=w, F=F, jp=w @ F)
    if prev is not None and cur["fid"] == prev["fid"] + 1:
        c = np.clip((np.trace(prev["F"].T @ F) - 1) / 2, -1, 1)
        theta = np.degrees(np.arccos(c))
        d_hand = np.linalg.norm(cur["jp"] - prev["jp"], axis=1).mean()
        rr, rf = kabsch_residuals(prev["w"], w)
        recs.append((cur["task"], theta, d_hand, rr, rf))
    prev = cur

th = np.array([x[1] for x in recs]); dh = np.array([x[2] for x in recs])
rr = np.array([x[3] for x in recs]); rf = np.array([x[4] for x in recs])
J = th > TH_DEG
base = np.median(dh[~J])
rigid = J & (dh <= 3 * base)
refl  = J & ~rigid & (rf < 0.5 * rr)
nonr  = J & ~rigid & ~refl
print(f"\n[{path.split('/')[-2]}]  연속 쌍 {len(recs)}  급변(θ>{TH_DEG:.0f}°) {J.sum()} ({J.mean():.2%})")
print(f"  θ(급변)  중앙 {np.median(th[J]):.0f}°   150° 이상 {np.mean(th[J] > 150):.1%}")
print(f"  손 좌표계 기준 변화 (mm)  평상시 중앙 {base*1e3:.1f}   급변 시 중앙 {np.median(dh[J])*1e3:.1f}")
print(f"  Kabsch 잔차 (mm, 급변 시 중앙)  회전 {np.median(rr[J])*1e3:.1f}   반사 {np.median(rf[J])*1e3:.1f}")
print(f"  분류  강체 회전 {rigid.sum()}  |  거울 반사 {refl.sum()}  |  비강체 {nonr.sum()}")
tot, jmp = defaultdict(int), defaultdict(int)
for (t, *_), j in zip(recs, J):
    tot[t] += 1; jmp[t] += int(j)
print("  동작별 급변률  " + "   ".join(f"{t} {jmp[t]/tot[t]:.1%}" for t in sorted(tot, key=str)))
