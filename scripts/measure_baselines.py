import numpy as np, argparse
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("session")
ap.add_argument("--labels", default="labels.npz")
args = ap.parse_args()

d = np.load(Path("data/raw") / args.session / args.labels, allow_pickle=True)
q = d["qpos_indep"].astype(np.float64)          # (N, 6)
fid = d["frame_id"]
names = [str(x) for x in d["indep_names"]]
lat = d["latency_ms"]

# 추적 끊김으로 프레임이 빠졌을 수 있으니 연속 구간으로 분할
breaks = np.where(np.diff(fid) != 1)[0] + 1
segments = np.split(np.arange(len(fid)), breaks)
segments = [s for s in segments if len(s) > 30]
print(f"{len(segments)} contiguous segments, "
      f"lengths {min(len(s) for s in segments)}~{max(len(s) for s in segments)}")


def jitter(x):
    """프레임 간 변화량의 표준편차 (관절 평균)"""
    return np.mean([np.std(np.diff(x[:, j])) for j in range(x.shape[1])])


def ema(x, lam):
    y = np.empty_like(x)
    y[0] = x[0]
    for t in range(1, len(x)):
        y[t] = lam * y[t - 1] + (1 - lam) * x[t]
    return y


def lag_frames(y, x, max_shift=20):
    """y가 x보다 몇 프레임 뒤처지는지 (교차상관 최대 지점)"""
    best, best_s = -np.inf, 0
    xc = x - x.mean(0)
    yc = y - y.mean(0)
    for s in range(max_shift + 1):
        a = yc[s:] if s else yc
        b = xc[:len(xc) - s] if s else xc
        v = np.sum(a * b)
        if v > best:
            best, best_s = v, s
    return best_s


FPS = 29.0
print(f"\n{'method':22s}{'jitter σ(Δa)':>15s}{'lag(frames)':>13s}{'lag(ms)':>10s}")
print("-" * 60)

raw_j = np.mean([jitter(q[s]) for s in segments])
print(f"{'optimization (raw)':22s}{raw_j:15.5f}{0:13d}{0.0:10.1f}")

for lam in [0.3, 0.5, 0.7, 0.9]:
    js, ls = [], []
    for s in segments:
        y = ema(q[s], lam)
        js.append(jitter(y))
        ls.append(lag_frames(y, q[s]))
    j, l = np.mean(js), np.mean(ls)
    print(f"{'+ EMA λ=' + str(lam):22s}{j:15.5f}{l:13.2f}{l/FPS*1000:10.1f}")

print(f"\nretarget latency: mean={lat.mean():.2f}ms p95={np.percentile(lat,95):.2f}ms")

# 관절별 떨림 (어느 축이 제일 떠는지)
print(f"\n{'joint':30s}{'σ(Δa) raw':>12s}{'range used':>14s}")
for j, n in enumerate(names):
    sj = np.mean([np.std(np.diff(q[s][:, j])) for s in segments])
    print(f"{n:30s}{sj:12.5f}{q[:,j].max()-q[:,j].min():14.3f}")

np.savez(Path("data") / f"baselines_{args.session}.npz",
         raw_jitter=raw_j, latency=lat)
print(f"\nsaved -> data/baselines_{args.session}.npz")
