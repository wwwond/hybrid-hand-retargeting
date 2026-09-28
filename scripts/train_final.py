import numpy as np, torch, torch.nn as nn, json
from pathlib import Path
torch.manual_seed(0); np.random.seed(0)

K, H = 16, 40
D = np.load("data/dataset.npz", allow_pickle=True)
Xtr, Ytr, Str = D["X_train"], D["Y_train"], D["seg_train"]
xmu, xsd = Xtr.mean(0), Xtr.std(0) + 1e-6
Xn = (Xtr - xmu) / xsd


def windows(seg, k, H):
    xi, yi = [], []
    for s in np.unique(seg):
        m = np.where(seg == s)[0]
        if len(m) < k + H: continue
        for i in range(k - 1, len(m) - H + 1):
            xi.append(m[i - k + 1:i + 1]); yi.append(m[i:i + H])
    return np.array(xi), np.array(yi)


xi, yi = windows(Str, K, H)
Dl = Ytr[yi] - Ytr[yi[:, 0]][:, None, :]
dsd = Dl.reshape(-1, 6).std(0) + 1e-6
perm = np.random.permutation(len(xi)); nv = len(xi) // 10
vi, tri = perm[:nv], perm[nv:]
Xw = torch.tensor(Xn[xi].reshape(len(xi), -1))
Yw = torch.tensor((Dl / dsd).reshape(len(yi), -1).astype(np.float32))

model = nn.Sequential(nn.Linear(63 * K, 512), nn.ReLU(), nn.Dropout(0.1),
                      nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.1),
                      nn.Linear(512, 6 * H))
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
best, bst, pat = 1e9, None, 0
for ep in range(400):
    model.train()
    p = np.random.permutation(tri)
    for i in range(0, len(p), 256):
        b = p[i:i + 256]; opt.zero_grad()
        nn.functional.huber_loss(model(Xw[b]), Yw[b], delta=0.5).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        v = nn.functional.huber_loss(model(Xw[vi]), Yw[vi], delta=0.5).item()
    if v < best - 1e-6:
        best, bst, pat = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
    else:
        pat += 1
        if pat >= 35: break
    if (ep + 1) % 50 == 0: print(f"  ep{ep+1} val {v:.5f}")
model.load_state_dict(bst); model.eval()

Path("models").mkdir(exist_ok=True)
torch.save({"state_dict": bst, "K": K, "H": H,
            "xmu": xmu, "xsd": xsd, "dsd": dsd,
            "arch": [63 * K, 512, 512, 6 * H]}, "models/policy.pt")

# 추론 속도 측정
with torch.no_grad():
    x = torch.randn(1, 63 * K)
    import time
    for _ in range(20): model(x)
    t0 = time.perf_counter()
    for _ in range(500): model(x)
    ms = (time.perf_counter() - t0) / 500 * 1000
print(f"\n학습 완료 ep{ep+1}, val {best:.5f}")
print(f"추론 지연 {ms:.3f} ms/frame")
print("saved -> models/policy.pt")
