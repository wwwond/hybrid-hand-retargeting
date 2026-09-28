import cv2, time, json, argparse, math
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from pathlib import Path
from telemetry import Telemetry

MODEL_PATH = "../../models/hand_landmarker.task"

# (이름, 지속시간, 안내문, 메트로놈 주기(초) 또는 None)
MOTIONS = [
    ("sweep_slow",   60, "OPEN <-> CLOSE  slowly, follow the bar", 8.0),
    ("sweep_fast",   60, "OPEN <-> CLOSE  quickly, follow the bar", 2.5),
    ("grasp_cycle",  60, "reach -> grasp -> hold -> release, repeat", None),
    ("rotate_grasp", 60, "hold a grasp, rotate your wrist slowly", None),
    ("free_grasp",   60, "grasp varied imaginary objects, keep moving", None),
]
PREP_SEC = 6.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--session-id", default=None)
    p.add_argument("--lighting", default="bright")
    p.add_argument("--hand", default="left", choices=["left", "right"])
    p.add_argument("--notes", default="")
    p.add_argument("--only", default=None, help="한 가지 동작만 (이름)")
    return p.parse_args()


def main():
    args = parse_args()
    session_id = args.session_id or f"dyn_{args.hand}_{int(time.time())}"
    out_dir = Path("../../data/raw") / session_id
    out_dir.mkdir(parents=True, exist_ok=True)

    motions = [m for m in MOTIONS if (args.only is None or m[0] == args.only)]

    json.dump({
        "session_id": session_id, "protocol": "dynamic_continuous",
        "lighting": args.lighting, "hand": args.hand, "notes": args.notes,
        "motions": [m[0] for m in motions],
        "prep_sec": PREP_SEC, "created_at": time.time(),
    }, open(out_dir / "meta.json", "w"), indent=2, ensure_ascii=False)

    lm_file = open(out_dir / "landmarks.jsonl", "a")
    landmarker = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            num_hands=1, running_mode=vision.RunningMode.VIDEO))
    telemetry = Telemetry(session_id, out_dir=str(out_dir))

    cap = cv2.VideoCapture(0)
    frame_idx = 0
    total = sum(m[1] for m in motions) + PREP_SEC * len(motions)
    print(f"Session: {session_id}\nEstimated: {total/60:.1f} min\n")

    stopped = False
    for mi, (name, dur, prompt, beat) in enumerate(motions):
        for phase, plen in [("prep", PREP_SEC), ("motion", dur)]:
            t0 = time.time()
            while time.time() - t0 < plen:
                ret, frame = cap.read()
                if not ret:
                    stopped = True
                    break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                    int(time.time() * 1000))
                tracked = bool(res.hand_world_landmarks)
                h, w, _ = frame.shape
                el = time.time() - t0

                if tracked:
                    lm_file.write(json.dumps({
                        "frame_id": frame_idx, "ts": time.time(),
                        "motion": name, "phase": phase,
                        "handedness": res.handedness[0][0].category_name,
                        "landmarks": [[p.x, p.y, p.z] for p in res.hand_landmarks[0]],
                        "world_landmarks": [[p.x, p.y, p.z] for p in res.hand_world_landmarks[0]],
                    }) + "\n")
                    for p in res.hand_landmarks[0]:
                        cv2.circle(frame, (int(p.x*w), int(p.y*h)), 3, (0, 255, 0), -1)

                telemetry.log(tracking_valid=tracked,
                              extra={"motion": name, "phase": phase})
                frame_idx += 1

                ov = frame.copy()
                cv2.rectangle(ov, (0, 0), (w, 120), (0, 0, 0), -1)
                cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)

                if phase == "prep":
                    cv2.putText(frame, f"GET READY  {plen-el:.0f}", (15, 42),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
                else:
                    cv2.putText(frame, f"{name.upper()}  {plen-el:4.0f}s", (15, 42),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 0), 2)
                cv2.putText(frame, prompt, (15, 76),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f"[{mi+1}/{len(motions)}]  "
                                   f"{'TRACKED' if tracked else 'LOST'}", (15, 106),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 255, 255) if tracked else (0, 0, 255), 1)

                # 메트로놈: 사인파 막대로 펴기/쥐기 타이밍 안내
                if beat and phase == "motion":
                    ph = (el / beat) % 1.0
                    s = abs(1.0 - 2.0 * ph)          # 1=OPEN, 0=CLOSE, 등속
                    bx = int(60 + s * (w - 120))
                    cv2.line(frame, (60, h - 60), (w - 60, h - 60), (90, 90, 90), 4)
                    cv2.circle(frame, (bx, h - 60), 22, (0, 220, 255), -1)
                    cv2.putText(frame, "CLOSE", (18, h - 52),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
                    cv2.putText(frame, "OPEN", (w - 110, h - 52),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
                    cue = "OPENING" if ph < 0.5 else "CLOSING"
                    cv2.putText(frame, cue, (int(w/2) - 110, h - 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 220, 255), 3)
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    stopped = True
                    break
            if stopped:
                break
        if stopped:
            break

    cap.release(); cv2.destroyAllWindows()
    landmarker.close(); telemetry.close(); lm_file.close()
    n = sum(1 for _ in open(out_dir / "landmarks.jsonl"))
    print(f"\n{'STOPPED EARLY' if stopped else 'COMPLETE'}  {n} frames -> {out_dir}")


if __name__ == "__main__":
    main()
