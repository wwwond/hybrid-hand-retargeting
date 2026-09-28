import numpy as np

WRIST_IDX = 0
MIDDLE_MCP_IDX = 9

def normalize_landmarks(landmarks) -> np.ndarray:
    """
    landmarks: MediaPipe HandLandmarker 결과의 한 손 (21개 Landmark 객체, x/y/z 속성)
    반환: (21, 3) numpy array, 손목 원점 + 손 크기로 스케일 정규화됨
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)  # (21, 3)

    wrist = pts[WRIST_IDX]
    scale = np.linalg.norm(pts[MIDDLE_MCP_IDX] - wrist)
    if scale < 1e-6:
        scale = 1e-6  # 0 나누기 방지

    normalized = (pts - wrist) / scale
    return normalized  # (21, 3), 손목=원점, 중지MCP 거리=1.0
