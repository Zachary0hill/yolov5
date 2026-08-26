# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Preview MediaPipe hand landmarks and raised-finger counts from a webcam."""

import argparse
import math
import os
import tempfile
import time
import urllib.request
from pathlib import Path

import cv2

try:
    import mediapipe as mp
except ImportError as error:
    raise SystemExit("MediaPipe is required. Install it with: python -m pip install -e '.[hands]'") from error

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/"
    "hand_landmarker.task"
)
FINGER_JOINTS = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
HAND_COLORS = ((255, 80, 40), (190, 50, 255))


def parse_opt():
    """Parse webcam and hand-tracking options."""
    parser = argparse.ArgumentParser(description="Draw hand landmarks and count raised fingers from a webcam.")
    parser.add_argument("--source", type=int, default=0, help="camera index, usually 0 for the MacBook camera")
    parser.add_argument("--model", type=Path, help="path to a MediaPipe hand_landmarker.task model")
    parser.add_argument("--max-hands", type=int, default=2, help="maximum number of hands to track")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="minimum detection and tracking confidence")
    parser.add_argument("--width", type=int, default=1280, help="requested camera width")
    parser.add_argument("--height", type=int, default=720, help="requested camera height")
    parser.add_argument("--no-mirror", action="store_true", help="do not mirror the selfie-camera preview")
    return parser.parse_args()


def model_path(path=None):
    """Return a local hand-landmarker model, downloading the official model once when needed."""
    if path:
        if not path.is_file():
            raise FileNotFoundError(f"Hand landmark model not found: {path}")
        return path

    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "yolov5"
    cached_model = cache_root / "hand_landmarker-v1.task"
    if cached_model.is_file():
        return cached_model

    cache_root.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MediaPipe hand model to {cached_model}")
    with tempfile.NamedTemporaryFile(dir=cache_root, delete=False) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        urllib.request.urlretrieve(MODEL_URL, temporary_path)
        temporary_path.replace(cached_model)
    finally:
        temporary_path.unlink(missing_ok=True)
    return cached_model


def angle(a, b, c):
    """Return the angle ABC in degrees for three 3D landmarks."""
    ba = (a.x - b.x, a.y - b.y, a.z - b.z)
    bc = (c.x - b.x, c.y - b.y, c.z - b.z)
    denominator = math.sqrt(sum(value * value for value in ba) * sum(value * value for value in bc))
    if denominator == 0:
        return 0.0
    cosine = max(-1.0, min(1.0, sum(x * y for x, y in zip(ba, bc)) / denominator))
    return math.degrees(math.acos(cosine))


def distance(a, b):
    """Return the Euclidean distance between two 3D landmarks."""
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def extended_fingers(landmarks):
    """Estimate which fingers are extended using orientation-independent joint geometry."""
    wrist = landmarks[0]
    states = []
    for finger_index, (mcp, pip, dip, tip) in enumerate(FINGER_JOINTS):
        joint = pip if finger_index else dip
        base = mcp if finger_index else pip
        straight = angle(landmarks[base], landmarks[joint], landmarks[tip]) > 155
        away_from_palm = distance(wrist, landmarks[tip]) > distance(wrist, landmarks[joint]) * 1.08
        states.append(straight and away_from_palm)
    return states


def draw_hand(frame, image_landmarks, world_landmarks, handedness, color):
    """Draw one hand skeleton, fingertip states, bounding box, and count."""
    height, width = frame.shape[:2]
    points = [(int(landmark.x * width), int(landmark.y * height)) for landmark in image_landmarks]
    states = extended_fingers(world_landmarks)

    for connection in mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS:
        cv2.line(frame, points[connection.start], points[connection.end], color, 3, cv2.LINE_AA)
    for index, point in enumerate(points):
        tip_index = index in {4, 8, 12, 16, 20}
        tip_is_up = tip_index and states[(index - 4) // 4]
        point_color = (40, 220, 40) if tip_is_up else ((40, 40, 255) if tip_index else color)
        cv2.circle(frame, point, 7 if tip_index else 4, point_color, -1, cv2.LINE_AA)

    x_values, y_values = zip(*points)
    left, top = max(0, min(x_values) - 20), max(0, min(y_values) - 35)
    right, bottom = min(width - 1, max(x_values) + 20), min(height - 1, max(y_values) + 20)
    count = sum(states)
    label = f"{handedness}: {count} finger{'s' if count != 1 else ''}"
    label_bottom = max(30, top)
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2, cv2.LINE_AA)
    cv2.rectangle(frame, (left, label_bottom - 30), (min(width - 1, left + 230), label_bottom), color, -1)
    cv2.putText(
        frame,
        label,
        (left + 7, label_bottom - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return count


def run(opt):
    """Open the selected camera and render live hand landmark results."""
    if opt.max_hands < 1:
        raise ValueError("--max-hands must be at least 1")
    if not 0 <= opt.conf_thres <= 1:
        raise ValueError("--conf-thres must be between 0 and 1")

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path(opt.model))),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=opt.max_hands,
        min_hand_detection_confidence=opt.conf_thres,
        min_hand_presence_confidence=opt.conf_thres,
        min_tracking_confidence=opt.conf_thres,
    )
    camera = cv2.VideoCapture(opt.source)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, opt.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, opt.height)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera {opt.source}. Check macOS camera permissions or try --source 1.")

    timestamp_ms = 0
    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                success, frame = camera.read()
                if not success:
                    raise RuntimeError("The camera stopped returning frames.")
                if not opt.no_mirror:
                    frame = cv2.flip(frame, 1)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                media_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = max(timestamp_ms + 1, time.monotonic_ns() // 1_000_000)
                result = landmarker.detect_for_video(media_image, timestamp_ms)

                total = 0
                for index, (image_hand, world_hand, handedness) in enumerate(
                    zip(result.hand_landmarks, result.hand_world_landmarks, result.handedness)
                ):
                    label = handedness[0].category_name if handedness else "Hand"
                    total += draw_hand(frame, image_hand, world_hand, label, HAND_COLORS[index % len(HAND_COLORS)])

                cv2.rectangle(frame, (18, 18), (330, 84), (20, 20, 20), -1)
                cv2.putText(
                    frame,
                    f"Total raised fingers: {total}",
                    (32, 61),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("YOLOv5 Hand Tracking - press q to quit", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run(parse_opt())
