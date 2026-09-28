import numpy as np, torch, torch.nn as nn
torch.manual_seed(0); np.random.seed(0)

D = np.load("data/dataset.npz", allow_pickle=True)
Xtr, Ytr, Str = D["X_train"], D["Y_train"], D["seg_train"]
Xte, Yte, Ste = D["X_test"], D["Y_test"], D["seg_test"]
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
K, H, FPS = 16, 16, 29.0


def windows(X, seg, k, H):
    xi, yi, ti = [], [], []
    for s in np.unique(seg):
        m = np.where(seg == s)[0]
        if len(m) < k + H: continue
        for i in range(k - 1, len(m) - H + 1):
            xi.append(m[i - k + 1:i + 1]); yi.append(m[i:i + H]); ti.append(m[i])
    return np.array(xi), np.array(yi), np.array(ti)


def jitter_seg(a, t_idx, seg):
    v = []
    for s in np.unique(seg[t_idx]):
        m = seg[t_idx] == s
        if m.sum() > 30:
            v.append(np.mean([np.std(np.diff(a[m][:, j])) for j in range(a.shape[1])]))
    return float(np.mean(v))


def lag_seg(pred, true, t_idx, seg, mx=20):
    ls = []
    for s in np.unique(seg[t_idx]):
        m = seg[t_idx] == s
        if m.sum() < 100: continue
        p, q = pred[m] - pred[m].mean(0), true[m] - true[m].mean(0)
        best, bs = -np.inf, 0
        for sh in range(mx + 1):
            a = p[sh:] if sh else p
            b = q[:len(q) - sh] if sh else q
            v = float(np.sum(a * b))
            if v > best: best, bs = v, sh
        ls.append(bs)
    return float(np.mean(ls))


def ema(x, lam):
    y = np.empty_like(x); y[0] = x[0]
    for t in range(1, len(x)):
        y[t] = lam * y[t - 1] + (1 - lam) * x[t]
    return y


# ---------- 청크 모델 학습 (검증 분할 + 조기 종료)
xi, yi, ti = windows(Xtr, Str, K, H)
perm = np.random.permutation(len(xi)); nv = len(xi) // 10
vi, tri = perm[:nv], perm[nv:]
Xw = torch.tensor(Xtr[xi].reshape(len(xi), -1))
Yw = torch.tensor(Ytr[yi].reshape(len(yi), -1))

model = nn.Sequential(nn.Linear(63 * K, 256), nn.ReLU(), nn.Dropout(0.1),
                      nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.1),
                      nn.Linear(256, 6 * H))
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
best, bstate, patience = 1e9, None, 0
for ep in range(200):
    model.train()
    p = np.random.permutation(tri)
    for i in range(0, len(p), 256):
        b = p[i:i + 256]
        opt.zero_grad()
        nn.functional.mse_loss(model(Xw[b]), Yw[b]).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        v = nn.functional.mse_loss(model(Xw[vi]), Yw[vi]).item()
    if v < best - 1e-6:
        best, bstate, patience = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
    else:
        patience += 1
        if patience >= 15: break
    if (ep + 1) % 20 == 0: print(f"  ep{ep+1:3d} val {v:.5f} (best {best:.5f})")
model.load_state_dict(bstate); model.eval()
print(f"조기 종료 ep{ep+1}, val MSE {best:.5f}\n")

# ---------- 테스트 추론
xe, ye, te = windows(Xte, Ste, K, H)
with torch.no_grad():
    ch = model(torch.tensor(Xte[xe].reshape(len(xe), -1))).numpy().reshape(-1, H, 6)
true = Yte[ye[:, 0]]
pos = {t: i for i, t in enumerate(te)}


def ens(m):
    out = np.zeros((len(te), 6)); w = np.zeros((len(te), 1))
    for i, t in enumerate(te):
        for j in range(H):
            src = pos.get(t - j)
            if src is None: continue
            ww = np.exp(-m * j)
            out[i] += ww * ch[src, j]; w[i] += ww
    return out / np.maximum(w, 1e-9)


print("=== A. 최적화 + EMA (기준 곡선) ===")
print(f"{'method':22s}{'MAE':>9s}{'jitter':>11s}{'lag(f)':>9s}{'lag(ms)':>10s}")
raw = Yte[ye[:, 0]]
for lam in [0.0, 0.3, 0.5, 0.7, 0.8, 0.9]:
    sm = np.concatenate([ema(Yte[np.where(Ste == s)[0]], lam) for s in np.unique(Ste)])
    idx = ye[:, 0]
    p = sm[idx]
    lg = lag_seg(p, raw, te, Ste)
    print(f"{'EMA λ=' + str(lam):22s}{np.mean(np.abs(p - raw)):9.4f}"
          f"{jitter_seg(p, te, Ste):11.5f}{lg:9.2f}{lg / FPS * 1000:10.1f}")

print("\n=== B. 학습 정책 + 시간 앙상블 (우리 곡선) ===")
print(f"{'method':22s}{'MAE':>9s}{'jitter':>11s}{'lag(f)':>9s}{'lag(ms)':>10s}")
for m in [99.0, 1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.0]:
    p = ens(m)
    lg = lag_seg(p, true, te, Ste)
    tag = "latest only" if m > 10 else f"m={m}"
    print(f"{tag:22s}{np.mean(np.abs(p - true)):9.4f}"
          f"{jitter_seg(p, te, Ste):11.5f}{lg:9.2f}{lg / FPS * 1000:10.1f}")
