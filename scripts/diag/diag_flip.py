#!/usr/bin/env python3
# 좌표계 급변 원인 분리: ⓐ근-공선 / ⓑhandedness 흔들림 / ⓒ그 외(깊이 반전 등)
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

def sign_margin(kp):
    """부호 판정 여유(0~1). 0에 가까우면 손목-검지MCP-중지MCP가 일직선."""
    p0, p1, p2 = kp[0], kp[5], kp[9]
    x = (p0 - p2) / np.linalg.norm(p0 - p2)
    d = p1 - p2
    return np.linalg.norm(d - np.dot(d, x) * x) / np.linalg.norm(d)

path = sys.argv[1]
rows = [json.loads(l) for l in open(path)]
print(f"\n[{path.split('/')[-2]}]  레코드 {len(rows)}  첫 레코드 키: {sorted(rows[0].keys())}")
hcount = Counter(r.get("handedness") for r in rows if r.get("world_landmarks"))
print(f"  handedness 라벨 분포: {dict(hcount)}")

prev, flips = None, []
n_pair = n_raw = n_hchg = 0
m_all, m_flip = [], []
for i, r in enumerate(rows):
    wl = r.get("world_landmarks")
    if not wl:
        prev = None; continue
    raw = np.asarray(wl, dtype=np.float64).reshape(21, 3)
    h = r.get("handedness")
    mir = raw.copy()
    if h == "Left": mir[:, 0] *= -1
    cur = dict(fid=r.get("frame_id", i), h=h, m=sign_margin(raw),
               Fm=estimate_frame(mir), Fr=estimate_frame(raw))
    m_all.append(cur["m"])
    if prev is not None and cur["fid"] == prev["fid"] + 1:
        n_pair += 1
        if np.linalg.norm(cur["Fr"] - prev["Fr"]) > 0.5: n_raw += 1
        if np.linalg.norm(cur["Fm"] - prev["Fm"]) > 0.5:
            flips.append(i)
            m_flip.append(min(cur["m"], prev["m"]))
            if cur["h"] != prev["h"]: n_hchg += 1
    prev = cur

if n_pair == 0:
    sys.exit("  연속 프레임 쌍이 0개 — frame_id 키 이름을 확인해야 함")
nf = max(len(flips), 1)
spike = sum(1 for a, b in zip(flips, flips[1:]) if b - a == 1)
mf = np.median(m_flip) if m_flip else float("nan")
print(f"  연속 프레임 쌍 {n_pair}")
print(f"  급변 비율   미러 적용(런타임과 동일) {len(flips)/n_pair:.2%}  |  미러 미적용 {n_raw/n_pair:.2%}")
print(f"  급변 중 handedness 라벨 변경 동반  {n_hchg/nf:.1%}   → 높으면 ⓑ")
print(f"  부호 여유 중앙값  전체 {np.median(m_all):.3f}  급변 {mf:.3f}   → 급변 쪽이 훨씬 작으면 ⓐ")
print(f"  1프레임 스파이크(바로 되돌아감) {spike}/{len(flips)}")
