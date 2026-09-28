import numpy as np, torch, torch.nn as nn, time
torch.manual_seed(0); np.random.seed(0)
torch.set_num_threads(4)

D = np.load("data/dataset.npz", allow_pickle=True)
Xtr, Ytr, Str = D["X_train"], D["Y_train"], D["seg_train"]
Xte, Yte, Ste = D["X_test"], D["Y_test"], D["seg_test"]
xmu, xsd = Xtr.mean(0), Xtr.std(0) + 1e-6
ymu, ysd = Ytr.mean(0), Ytr.std(0) + 1e-6      # 출력도 표준화
Xtr, Xte = (Xtr - xmu) / xsd, (Xte - xmu) / xsd
K, H, FPS = 16, 16, 29.0


def windows(seg, k, H):
    xi, yi, ti = [], [], []
    for s in np.unique(seg):
        m = np.where(seg == s)[0]
        if len(m) < k + H: continue
        for i in range(k - 1, len(m) - H + 1):
            xi.append(m[i - k + 1:i + 1]); yi.append(m[i:i + H]); ti.append(m[i])
    return np.array(xi), np.array(yi), np.array(ti)


class GRUHead(nn.Module):
    def __init__(self, din=63, hid=256, out=6, H=16):
        super().__init__()
        self.enc = nn.GRU(din, hid, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(hid, 512), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(512, 512), nn.ReLU(),
                                  nn.Linear(512, out * H))
        self.H, self.out = H, out

    def forward(self, x):                     # x: (B, K, 63)
        h, _ = self.enc(x)
        return self.head(h[:, -1]).view(-1, self.H, self.out)


xi, yi, ti = windows(Str, K, H)
perm = np.random.permutation(len(xi)); nv = len(xi) // 10
vi, tri = perm[:nv], perm[nv:]
Xw = torch.tensor(Xtr[xi])                                   # (N, K, 63)
Yw = torch.tensor(((Ytr[yi] - ymu) / ysd).astype(np.float32))  # (N, H, 6)

model = GRUHead(H=H)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=400)
lossfn = nn.HuberLoss(delta=0.5)

best, bstate, pat, t0 = 1e9, None, 0, time.time()
for ep in range(400):
    model.train()
    p = np.random.permutation(tri)
    for i in range(0, len(p), 256):
        b = p[i:i + 256]
        opt.zero_grad(); lossfn(model(Xw[b]), Yw[b]).backward(); opt.step()
    sch.step()
    model.eval()
    with torch.no_grad():
        v = lossfn(model(Xw[vi]), Yw[vi]).item()
    if v < best - 1e-6:
        best, bstate, pat = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
    else:
        pat += 1
        if pat >= 40: break
    if (ep + 1) % 20 == 0:
        print(f"  ep{ep+1:3d} val {v:.5f} best {best:.5f}  {time.time()-t0:.0f}s")
model.load_state_dict(bstate); model.eval()
print(f"조기 종료 ep{ep+1}, val {best:.5f}, {time.time()-t0:.0f}s\n")

xe, ye, te = windows(Ste, K, H)
with torch.no_grad():
    ch = model(torch.tensor(Xte[xe])).numpy() * ysd + ymu      # (N, H, 6)

print("=== 미래 예측 정확도 (청크 단계별) ===")
print(f"{'step':>6s}{'MAE':>10s}   ← EMA는 원리적으로 불가")
for d in [0, 1, 2, 3, 5, 8, 15]:
    if d >= H: continue
    print(f"{'t+' + str(d):>6s}{np.mean(np.abs(ch[:, d] - Yte[ye[:, d]])):10.4f}")
np.savez("data/pred_test.npz", chunks=ch, t_idx=te, y_idx=ye, seg=Ste, Y=Yte)
print("\nsaved -> data/pred_test.npz")
