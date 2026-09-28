import numpy as np, torch, torch.nn as nn
torch.manual_seed(0); np.random.seed(0)

D = np.load("data/dataset.npz", allow_pickle=True)
Xtr, Ytr, Str = D["X_train"], D["Y_train"], D["seg_train"]
Xte, Yte, Ste = D["X_test"], D["Y_test"], D["seg_test"]
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
K, H = 16, 40


def windows(seg, k, H):
    xi, yi = [], []
    for s in np.unique(seg):
        m = np.where(seg == s)[0]
        if len(m) < k + H: continue
        for i in range(k - 1, len(m) - H + 1):
            xi.append(m[i - k + 1:i + 1]); yi.append(m[i:i + H])
    return np.array(xi), np.array(yi)


xi, yi = windows(Str, K, H)
Dl = Ytr[yi] - Ytr[yi[:, 0]][:, None, :]          # 차분 목표
dsd = Dl.reshape(-1, 6).std(0) + 1e-6
print(f"train windows {len(xi)}, 차분 std {np.round(dsd, 3)}")

perm = np.random.permutation(len(xi)); nv = len(xi) // 10
vi, tri = perm[:nv], perm[nv:]
Xw = torch.tensor(Xtr[xi].reshape(len(xi), -1))
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
    if (ep + 1) % 40 == 0: print(f"  ep{ep+1} val {v:.5f}")
model.load_state_dict(bst); model.eval()
print(f"학습 완료 ep{ep+1}, val {best:.5f}\n")

GAPS = [1, 5, 10, 20, 40]
rows = []
for n in GAPS:
    eh, ed = [], []
    for s in np.unique(Ste):
        m = np.where(Ste == s)[0]
        if len(m) < K + n + 5: continue
        for start in range(K, len(m) - n, 37):
            t0 = start - 1
            truth = Yte[m[start:start + n]]
            eh.append(np.abs(Yte[m[t0]][None] - truth).mean())
            win = Xte[m[t0 - K + 1:t0 + 1]].reshape(1, -1)
            with torch.no_grad():
                dl = model(torch.tensor(win)).numpy().reshape(H, 6) * dsd
            dl = dl - dl[0]                            # Δ0 = 0 강제
            pr = dl[:n] if n <= H else np.vstack([dl, np.repeat(dl[-1:], n - H, 0)])
            ed.append(np.abs(Yte[m[t0]][None] + pr - truth).mean())
    rows.append((n, np.mean(eh), np.mean(ed)))

print("=== 차분 학습 + H=40 ===")
print(f"{'갭':>6s}{'마지막값':>11s}{'차분정책':>11s}{'개선':>9s}")
for n, h, d in rows:
    print(f"{str(n)+'f':>6s}{h:11.4f}{d:11.4f}{(h-d)/h*100:8.1f}%")
