import json
import numpy as np
from pathlib import Path

OP2MANO_R = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
OP2MANO_L = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])

# 모두 Right 규약 — 세션 분할이 손 교체와 엉키지 않게
TRAIN = ["task_left_1790439137", "task_left_1790489302"]
TEST = ["task_left_1790418447"]
CROSS = ["task_left_1790439514"]          # Left 규약 — 교차 손 일반화 별도 평가


def hand_frame(kp):
    pts = kp[[0, 5, 9], :]
    xv = pts[0] - pts[2]
    pts = pts - pts.mean(0, keepdims=True)
    _, _, v = np.linalg.svd(pts)
    n = v[2, :]
    x = xv - np.dot(xv, n) * n
    x /= np.linalg.norm(x)
    z = np.cross(x, n)
    if np.dot(z, pts[1] - pts[2]) < 0:
        n, z = -n, -z
    return np.stack([x, n, z], axis=1)


def load_session(sid):
    d = Path("data/raw") / sid
    recs = [json.loads(l) for l in open(d / "landmarks.jsonl")]
    lab = np.load(d / "labels_nofilter.npz", allow_pickle=True)
    q = lab["qpos_indep"].astype(np.float32)
    assert len(recs) == len(q), f"{sid}: {len(recs)} vs {len(q)}"

    feats, keep = [], []
    for i, r in enumerate(recs):
        if r["phase"] != "hold":
            continue
        wl = np.array(r["world_landmarks"], dtype=np.float32)
        if r.get("handedness") == "Left":      # 거울 반전으로 오른손 규약 통일
            wl = wl.copy(); wl[:, 0] *= -1.0
        feats.append((wl @ hand_frame(wl) @ OP2MANO_R).ravel())
        keep.append(i)

    keep = np.array(keep)
    feat = np.array(feats, dtype=np.float32)
    fid = lab["frame_id"][keep]
    task = lab["task"].astype(str)[keep]
    rep = lab["rep"][keep]

    # 연속 구간: frame_id 연속 + 같은 (task, rep)
    brk = (np.diff(fid) != 1) | (task[1:] != task[:-1]) | (np.diff(rep) != 0)
    seg = np.concatenate([[0], np.cumsum(brk)])
    return feat, q[keep], seg, task, np.full(len(keep), sid)


def build(sids, offset):
    F, Q, S, T, I = [], [], [], [], []
    for sid in sids:
        f, q, s, t, i = load_session(sid)
        S.append(s + offset)
        offset = S[-1].max() + 1
        F.append(f); Q.append(q); T.append(t); I.append(i)
        print(f"  {sid}: {len(f)} frames, {len(np.unique(s))} segments")
    return (np.concatenate(F), np.concatenate(Q), np.concatenate(S),
            np.concatenate(T), np.concatenate(I), offset)


print("TRAIN")
Xtr, Ytr, Str, Ttr, Itr, off = build(TRAIN, 0)
print("TEST")
Xte, Yte, Ste, Tte, Ite, off = build(TEST, off)
print("CROSS-HAND")
Xcr, Ycr, Scr, Tcr, Icr, _ = build(CROSS, off)

out = Path("data/dataset.npz")
np.savez_compressed(out,
                    X_train=Xtr, Y_train=Ytr, seg_train=Str, task_train=Ttr, sess_train=Itr,
                    X_test=Xte, Y_test=Yte, seg_test=Ste, task_test=Tte, sess_test=Ite,
                    X_cross=Xcr, Y_cross=Ycr, seg_cross=Scr)
print(f"\ntrain {Xtr.shape} -> {Ytr.shape}   seg {len(np.unique(Str))}")
print(f"test  {Xte.shape} -> {Yte.shape}   seg {len(np.unique(Ste))}")
print(f"saved -> {out}  ({out.stat().st_size/1e6:.1f} MB)")
