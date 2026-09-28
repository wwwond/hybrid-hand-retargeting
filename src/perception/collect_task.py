import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import time
import json
import argparse
from pathlib import Path
from telemetry import Telemetry

MODEL_PATH = "../../models/hand_landmarker.task"

TASKS = [
    ("open",            "손가락을 최대로 쫙 펴세요"),
    ("large_diameter",  "물병을 쥔 것처럼 둥글게"),
    ("small_diameter",  "펜을 쥔 것처럼 좁게"),
    ("tip_pinch",       "엄지-검지 끝을 맞대세요"),
    ("tripod",          "엄지-검지-중지 끝을 모으세요"),
    ("lateral",         "엄지를 검지 옆면에 붙이세요"),
    ("index_extension", "검지만 펴고 나머지는 접으세요"),
]

HOLD_SEC = 5.0
SETTLE_SEC = 1.0
REST_SEC = 3.0
REPS = 6


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--session-id", type=str, default=None)
    p.add_argument("--lighting", type=str, default="bright")
    p.add_argument("--hand", type=str, default="right", choices=["left", "right"])
    p.add_argument("--reps", type=int, default=REPS)
    p.add_argument("--notes", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    session_id = args.session_id or f"task_{args.hand}_{int(time.time())}"

    out_dir = Path("../../data/raw") / session_id
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "session_id": session_id,
        "protocol": "ninapro_style",
        "lighting": args.lighting,
        "hand": args.hand,
        "tasks": [t[0] for t in TASKS],
        "hold_sec": HOLD_SEC,
        "rest_sec": REST_SEC,
        "reps": args.reps,
        "notes": args.notes,
        "created_at": time.time(),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    landmarks_file = open(out_dir / "landmarks.jsonl", "a")

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        running_mode=vision.RunningMode.VIDEO,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)
    telemetry = Telemetry(session_id, out_dir=str(out_dir))

    cap = cv2.VideoCapture(0)
    frame_idx = 0

    # (task, rep, phase, duration) 시퀀스 생성
    schedule = []
    for task_name, prompt in TASKS:
        for rep in range(args.reps):
            schedule.append((task_name, prompt, rep, "rest", REST_SEC))
            schedule.append((task_name, prompt, rep, "transition", SETTLE_SEC))
            schedule.append((task_name, prompt, rep, "hold", HOLD_SEC))

    total_sec = sum(s[4] for s in schedule)
    print(f"Session: {session_id}")
    print(f"Tasks: {len(TASKS)}, reps: {args.reps}")
    print(f"Estimated duration: {total_sec/60:.1f} min")
    print("Press 'q' anytime to stop early.\n")

    # 시작 전 카운트다운
    countdown_start = time.time()
    while time.time() - countdown_start < 10.0:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        remain = 10.0 - (time.time() - countdown_start)
        cv2.putText(frame, f"START IN {remain:.0f}", (50, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 200, 255), 4)
        cv2.putText(frame, "Hold hand in front of camera, palm facing you",
                    (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Task Collection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release(); cv2.destroyAllWindows(); return

    stopped = False
    for si, (task_name, prompt, rep, phase, duration) in enumerate(schedule):
        phase_start = time.time()
        label = task_name if phase == "hold" else "rest"

        while time.time() - phase_start < duration:
            ret, frame = cap.read()
            if not ret:
                stopped = True
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            infer_start = time.time()
            result = landmarker.detect_for_video(mp_image, int(time.time() * 1000))
            inference_ms = (time.time() - infer_start) * 1000

            h, w, _ = frame.shape
            tracked = bool(result.hand_landmarks)

            if tracked:
                hand_landmarks = result.hand_landmarks[0]
                world_landmarks = result.hand_world_landmarks[0]
                record = {
                    "frame_id": frame_idx,
                    "ts": time.time(),
                    "task": label,
                    "phase": phase,
                    "rep": rep,
                    "handedness": result.handedness[0][0].category_name,
                    "landmarks": [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
                    "world_landmarks": [[lm.x, lm.y, lm.z] for lm in world_landmarks],
                }
                landmarks_file.write(json.dumps(record) + "\n")

                for lm in hand_landmarks:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

            telemetry.log(tracking_valid=tracked, inference_ms=inference_ms,
                          extra={"task": label, "phase": phase, "rep": rep})
            frame_idx += 1

            remain = duration - (time.time() - phase_start)
            if phase == "hold":
                color = (0, 255, 0)
                banner = f"{task_name.upper()}  rep {rep+1}/{args.reps}"
                sub = prompt
            elif phase == "transition":
                color = (0, 200, 255)
                banner = f"-> {task_name.upper()}  GET READY"
                sub = prompt
            else:
                color = (128, 128, 128)
                banner = "REST"
                sub = f"next: {task_name}  -  {prompt}"

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 110), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            cv2.putText(frame, banner, (15, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(frame, sub, (15, 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"{remain:.1f}s   [{si+1}/{len(schedule)}]   {'TRACKED' if tracked else 'LOST'}",
                        (15, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 255) if tracked else (0, 0, 255), 1)

            # 진행 바
            prog = int(w * (si + (time.time() - phase_start) / duration) / len(schedule))
            cv2.rectangle(frame, (0, h - 8), (prog, h), (0, 200, 255), -1)

            cv2.imshow("Task Collection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stopped = True
                break

        if stopped:
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    telemetry.close()
    landmarks_file.close()

    n_lines = sum(1 for _ in open(out_dir / "landmarks.jsonl"))
    print(f"\n{'STOPPED EARLY' if stopped else 'COMPLETE'}")
    print(f"{n_lines} frames with landmarks -> {out_dir/'landmarks.jsonl'}")


if __name__ == "__main__":
    main()
