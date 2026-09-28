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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--session-id", type=str, default=None,
                   help="세션 이름. 생략하면 타임스탬프로 자동 생성")
    p.add_argument("--lighting", type=str, required=True,
                   choices=["bright", "normal", "dim", "backlit"],
                   help="조명 조건 태그")
    p.add_argument("--motion", type=str, required=True,
                   choices=["grasp_release", "partial_flex", "free_motion", "extremes"],
                   help="동작 패턴 태그 (extremes = 완전히 펴기/쥐기 캘리브레이션용)")
    p.add_argument("--hand", type=str, default="right", choices=["left", "right"])
    p.add_argument("--notes", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    session_id = args.session_id or f"{args.lighting}_{args.motion}_{int(time.time())}"

    out_dir = Path("../../data/raw") / session_id
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "session_id": session_id,
        "lighting": args.lighting,
        "motion": args.motion,
        "hand": args.hand,
        "notes": args.notes,
        "created_at": time.time(),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    landmarks_path = out_dir / "landmarks.jsonl"
    landmarks_file = open(landmarks_path, "a")

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        running_mode=vision.RunningMode.VIDEO,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)
    telemetry = Telemetry(session_id, out_dir=str(out_dir))

    cap = cv2.VideoCapture(0)
    prev_time = time.time()
    frame_idx = 0

    print(f"Session: {session_id}")
    print(f"  lighting={args.lighting}  motion={args.motion}  hand={args.hand}")
    print(f"Saving to: {out_dir}")
    print("Press 'q' to stop.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        infer_start = time.time()
        timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        inference_ms = (time.time() - infer_start) * 1000

        h, w, _ = frame.shape

        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            record = {
                "frame_id": frame_idx,
                "ts": time.time(),
                "landmarks": [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
            }
            landmarks_file.write(json.dumps(record) + "\n")

            for lm in hand_landmarks:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

            status = f"tracked  inf={inference_ms:.1f}ms  saved={frame_idx+1}"
            telemetry.log(tracking_valid=True, inference_ms=inference_ms)
        else:
            status = f"NOT tracked  inf={inference_ms:.1f}ms"
            telemetry.log(tracking_valid=False, inference_ms=inference_ms)

        now = time.time()
        fps = 1.0 / (now - prev_time) if now != prev_time else 0
        prev_time = now
        frame_idx += 1

        cv2.putText(frame, f"[{session_id}] FPS:{fps:.1f}  {status}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Data Collection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    telemetry.close()
    landmarks_file.close()

    n_lines = sum(1 for _ in open(landmarks_path))
    print(f"\nDone. {n_lines} frames with landmarks saved to {landmarks_path}")
    print(f"Telemetry: {telemetry.file_path}")


if __name__ == "__main__":
    main()
