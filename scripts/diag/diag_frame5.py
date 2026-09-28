#!/usr/bin/env python3
# 손 좌표계 법선의 프레임 간 흔들림: 3점 vs 5점 (느린 구간만)
import json, sys, numpy as np

def frame_from(kp, idx):
    """참조 구현(문서 4.2)과 같은 구조. 평면 피팅에 쓰는 점 집합만 바꿈."""
    pts = kp[idx, :]
    x_vector = kp[0] - kp[9]                      # 손목 → 중지MCP
    P = pts - pts.mean(0, keepdims=True)
    u, s, v = np.linalg.svd(P)
    normal = v[2, :]
    x = x_vector - np.sum(x_vector * normal) * normal
    x = x / np.linalg.norm(x)
    z = np.cross(x, normal)
    if np.sum(z * (kp[5] - kp[9])) < 0:           # 부호 규칙도 동일
        normal = -normal; z = -z
    return np.stack([x, normal, z], axis=1), s

I3, I5 = [0, 5, 9], [0, 5, 9, 13, 17]
sess = sys.argv[1]
rows = [json.loads(l) for l in open(f"data/raw/{sess}/landmarks.jsonl")]
prev = None; rec = []
for r in rows:
    wl = r.get("world_landmarks")
    if not wl or r.get("phase") != "hold":
        prev = None; continue
    w = np.asarray(wl, dtype=np.float64).reshape(21, 3)
    if r.get("handedness") == "Left": w = w.copy(); w[:, 0] *= -1
    F3, s3 = frame_from(w, I3)
    F5, s5 = frame_from(w, I5)
    cur = dict(fid=r["frame_id"], w=w, F3=F3, F5=F5,
               c3=s3[1]/s3[0], c5=s5[1]/s5[0])
    if prev is not None and cur["fid"] == prev["fid"] + 1:
        mv = np.linalg.norm(cur["w"] - prev["w"], axis=1).mean()
        ang = lambda a, b: np.degrees(np.arccos(
            np.clip(np.dot(a[:, 1], b[:, 1]), -1, 1)))   # 법선 사이 각
        rec.append((mv, ang(prev["F3"], cur["F3"]), ang(prev["F5"], cur["F5"]),
                    cur["c3"], cur["c5"]))
    prev = cur

a = np.array(rec)
slow = a[:, 0] < np.percentile(a[:, 0], 50)        # 느린 절반만
print(f"\n[{sess}]  hold 연속 {len(a)}  느린 구간 {slow.sum()}")
print(f"  평면 조건수 s1/s0   3점 {np.median(a[:,3]):.3f}   5점 {np.median(a[:,4]):.3f}")
print(f"  법선 흔들림(느린 구간, 도)")
print(f"    3점  중앙 {np.median(a[slow,1]):5.2f}  p95 {np.percentile(a[slow,1],95):6.2f}  최대 {a[slow,1].max():6.2f}")
print(f"    5점  중앙 {np.median(a[slow,2]):5.2f}  p95 {np.percentile(a[slow,2],95):6.2f}  최대 {a[slow,2].max():6.2f}")
