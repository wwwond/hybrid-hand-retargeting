import numpy as np, torch, torch.nn as nn
torch.manual_seed(0); np.random.seed(0)

D = np.load("data/dataset.npz", allow_pickle=True)
Xtr, Ytr, Str = D["X_train"], D["Y_train"], D["seg_train"]
Xte, Yte, Ste = D["X_test"], D["Y_test"], D["seg_test"]
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
ymu, ysd = Ytr.mean(0), Ytr.std(0) + 1e-6
Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
K, H = 16, 16


def windows(seg, k, H):
    xi, yi = [], []
    for s in np.unique(seg):
        m = np.where(seg == s)[0]
        if len(m) < k + H: continue
        for i in range(k - 1, len(m) - H + 1):
            xi.append(m[i - k + 1:i + 1]); yi.append(m[i:i + H])
    return np.array(xi), np.array(yi)


xi, yi = windows(Str, K, H)
perm = np.random.permutation(len(xi)); nv = len(xi) // 10
vi, tri = perm[:nv], perm[nv:]
Xw = torch.tensor(Xtr[xi].reshape(len(xi), -1))
Yw = torch.tensor(((Ytr[yi] - ymu) / ysd).reshape(len(yi), -1).astype(np.float32))

model = nn.Sequential(nn.Linear(63 * K, 512), nn.ReLU(), nn.Dropout(0.1),
                      nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.1),
                      nn.Linear(512, 6 * H))
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
best, bst, pat = 1e9, None, 0
for ep in range(300):
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
        if pat >= 30: break
    if (ep + 1) % 30 == 0: print(f"  ep{ep+1} val {v:.5f}")
model.load_state_dict(bst); model.eval()
print(f"학습 완료 ep{ep+1}, val {best:.5f}\n")

# ---- 인위적 갭 평가
GAPS = [1, 5, 10, 20, 40]
rows = []
for n in GAPS:
    errs = {"hold": [], "extrap": [], "policy": [], "anchor": []}
    for s in np.unique(Ste):
        m = np.where(Ste == s)[0]
        if len(m) < K + n + 5: continue
        for start in range(K, len(m) - n, 37):          # 갭 시작 지점 샘플링
            t0 = start - 1                               # 마지막 유효 프레임
            gap = m[start:start + n]
            truth = Yte[gap]

            # 1) 마지막 값 유지
            errs["hold"].append(np.abs(Yte[m[t0]][None] - truth).mean())

            # 2) 선형 외삽
            v = Yte[m[t0]] - Yte[m[t0 - 1]]
            ext = Yte[m[t0]][None] + v[None] * np.arange(1, n + 1)[:, None]
            errs["extrap"].append(np.abs(ext - truth).mean())

            # 3) 학습 정책 — 갭 직전 창으로 청크 예측, 부족하면 마지막 예측 유지
            win = Xte[m[t0 - K + 1:t0 + 1]].reshape(1, -1)
            with torch.no_grad():
                ch = model(torch.tensor(win)).numpy().reshape(H, 6) * ysd + ymu
            pred = ch[:n] if n <= H else np.vstack([ch, np.repeat(ch[-1:], n - H, 0)])
            errs["policy"].append(np.abs(pred - truth).mean())

            # 4) 앵커링: 마지막 관측값 + 정책이 예측한 변화량
            anc = Yte[m[t0]][None] + (pred - ch[0][None])
            errs["anchor"].append(np.abs(anc - truth).mean())

    rows.append((n, np.mean(errs["hold"]), np.mean(errs["extrap"]),
                 np.mean(errs["policy"]), np.mean(errs["anchor"]), len(errs["hold"])))

print("=== 추적 소실 구간 MAE (최적화는 출력 불가) ===")
print(f"{'갭':>6s}{'마지막값':>11s}{'선형외삽':>11s}{'정책(raw)':>12s}{'정책+앵커':>12s}{'개선':>9s}")
for n, h, e, p, a, c in rows:
    imp = (h - a) / h * 100
    print(f"{str(n) + 'f':>6s}{h:11.4f}{e:11.4f}{p:12.4f}{a:12.4f}{imp:8.1f}%")
print("\n(29fps 기준: 5f=172ms, 10f=345ms, 20f=690ms, 40f=1379ms)")
