import numpy as np, torch, torch.nn as nn, time, json
from pathlib import Path

torch.manual_seed(0); np.random.seed(0)
D = np.load("data/dataset.npz", allow_pickle=True)
Xtr, Ytr, Str = D["X_train"], D["Y_train"], D["seg_train"]
Xte, Yte, Ste = D["X_test"], D["Y_test"], D["seg_test"]

mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd


def windows(X, Y, seg, k, H):
    xi, yi, ti = [], [], []
    for s in np.unique(seg):
        m = np.where(seg == s)[0]
        if len(m) < k + H:
            continue
        for i in range(k - 1, len(m) - H + 1):
            xi.append(m[i - k + 1:i + 1]); yi.append(m[i:i + H]); ti.append(m[i])
    return np.array(xi), np.array(yi), np.array(ti)


def jitter(a):
    return float(np.mean([np.std(np.diff(a[:, j])) for j in range(a.shape[1])]))


def seg_jitter(pred, t_idx, seg):
    vals = []
    for s in np.unique(seg[t_idx]):
        m = seg[t_idx] == s
        if m.sum() > 30:
            vals.append(jitter(pred[m]))
    return float(np.mean(vals))


def mlp(din, dout, hidden=256, depth=2):
    layers, d = [], din
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    layers += [nn.Linear(d, dout)]
    return nn.Sequential(*layers)


def train(model, Xw, Yw, epochs=60, bs=256, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt, Yt = torch.tensor(Xw), torch.tensor(Yw)
    n = len(Xt)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(Xt[b]), Yt[b])
            loss.backward(); opt.step()
            tot += loss.item() * len(b)
        if (ep + 1) % 20 == 0:
            print(f"    ep{ep+1:3d} loss {tot/n:.5f}")
    return model


def ensemble(chunks, t_idx, H, m=0.1):
    """chunks: (N, H, 6) — 시점 t에서 시작한 청크. 시간 앙상블로 a_t 복원"""
    out = np.zeros((len(t_idx), 6), dtype=np.float64)
    wsum = np.zeros((len(t_idx), 1))
    pos = {t: i for i, t in enumerate(t_idx)}
    for i, t in enumerate(t_idx):
        for j in range(H):
            src = pos.get(t - j)
            if src is None:
                continue
            w = np.exp(-m * j)
            out[i] += w * chunks[src, j]
            wsum[i] += w
    return out / np.maximum(wsum, 1e-9)


results = []


def evaluate(name, pred, true, t_idx):
    mae = float(np.mean(np.abs(pred - true)))
    jt = seg_jitter(pred, t_idx, Ste)
    results.append((name, mae, jt))
    print(f"  -> MAE {mae:.4f}   jitter {jt:.5f}")


# 기준선: 최적화 라벨 자체의 떨림 (test 세션)
base_idx = np.arange(len(Yte))
print("=== baseline (optimization labels, test session) ===")
print(f"  jitter {seg_jitter(Yte, base_idx, Ste):.5f}")

# 1) 선형, k=1 H=1
print("\n=== 1. linear  k=1 H=1 ===")
xi, yi, ti = windows(Xtr, Ytr, Str, 1, 1)
A = np.linalg.lstsq(np.c_[Xtr[xi[:, 0]], np.ones(len(xi))], Ytr[yi[:, 0]], rcond=None)[0]
xe, ye, te = windows(Xte, Yte, Ste, 1, 1)
pred = np.c_[Xte[xe[:, 0]], np.ones(len(xe))] @ A
evaluate("linear k=1", pred, Yte[ye[:, 0]], te)

# 2) MLP, k=1 H=1
print("\n=== 2. MLP  k=1 H=1 ===")
m2 = train(mlp(63, 6), Xtr[xi[:, 0]], Ytr[yi[:, 0]])
with torch.no_grad():
    pred = m2(torch.tensor(Xte[xe[:, 0]])).numpy()
evaluate("MLP k=1", pred, Yte[ye[:, 0]], te)

# 3) 시간창, k=16 H=1
K = 16
print(f"\n=== 3. MLP  k={K} H=1 ===")
xi3, yi3, ti3 = windows(Xtr, Ytr, Str, K, 1)
xe3, ye3, te3 = windows(Xte, Yte, Ste, K, 1)
m3 = train(mlp(63 * K, 6), Xtr[xi3].reshape(len(xi3), -1), Ytr[yi3[:, 0]])
with torch.no_grad():
    pred = m3(torch.tensor(Xte[xe3].reshape(len(xe3), -1))).numpy()
evaluate(f"MLP k={K}", pred, Yte[ye3[:, 0]], te3)

# 4) 액션 청킹 + 시간 앙상블
H = 16
print(f"\n=== 4. MLP  k={K} H={H} + temporal ensemble ===")
xi4, yi4, ti4 = windows(Xtr, Ytr, Str, K, H)
xe4, ye4, te4 = windows(Xte, Yte, Ste, K, H)
m4 = train(mlp(63 * K, 6 * H), Xtr[xi4].reshape(len(xi4), -1),
           Ytr[yi4].reshape(len(yi4), -1))
with torch.no_grad():
    ch = m4(torch.tensor(Xte[xe4].reshape(len(xe4), -1))).numpy().reshape(-1, H, 6)
evaluate(f"chunk k={K} H={H} (no ens)", ch[:, 0], Yte[ye4[:, 0]], te4)
pred = ensemble(ch, te4, H)
evaluate(f"chunk k={K} H={H} + ens", pred, Yte[ye4[:, 0]], te4)

print("\n" + "=" * 58)
print(f"{'model':32s}{'MAE':>10s}{'jitter':>12s}")
print("-" * 58)
for n, a, j in results:
    print(f"{n:32s}{a:10.4f}{j:12.5f}")
