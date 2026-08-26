# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Run a context-aware YOLOE and MediaPipe camera preview."""

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from ultralytics import YOLOE

from hand_track import HAND_COLORS, draw_hand, model_path

PROMPTS = ("person", "cell phone", "remote control", "over-ear headphones")
COLORS = {
    "person": (0, 0, 255),
    "cell phone": (40, 210, 40),
    "remote control": (0, 150, 255),
    "over-ear headphones": (220, 60, 220),
}
THRESHOLDS = {
    "person": 0.40,
    "cell phone": 0.15,
    "remote control": 0.30,
    "over-ear headphones": 0.20,
}


@dataclass
class Detection:
    """One filtered object detection in original-frame coordinates."""

    name: str
    confidence: float
    box: tuple
    mask: np.ndarray
    track_id: int
    key: tuple
    hand: str = ""


@dataclass
class HandObservation:
    """MediaPipe landmarks and bounds for one visible hand."""

    label: str
    box: tuple
    image_landmarks: list


@dataclass
class TrackState:
    """Short temporal history used to suppress one-frame false positives."""

    hits: int
    last_frame: int


def parse_zone(value):
    """Parse a normalized x1,y1,x2,y2 ignore zone."""
    try:
        zone = tuple(float(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("ignore zones must contain four comma-separated numbers") from error
    if len(zone) != 4 or not all(0 <= item <= 1 for item in zone):
        raise argparse.ArgumentTypeError("ignore zones must be x1,y1,x2,y2 values between 0 and 1")
    if zone[0] >= zone[2] or zone[1] >= zone[3]:
        raise argparse.ArgumentTypeError("ignore-zone x2/y2 values must be greater than x1/y1")
    return zone


def parse_opt():
    """Parse camera, inference, and display options."""
    parser = argparse.ArgumentParser(description="Context-aware object, hand, and finger tracking camera.")
    parser.add_argument("--source", default="0", help="camera index or video path")
    parser.add_argument("--model", default="yoloe-26n-seg.pt", help="YOLOE segmentation checkpoint")
    parser.add_argument("--imgsz", type=int, default=960, help="YOLO inference size")
    parser.add_argument("--device", help="inference device; defaults to mps when available")
    parser.add_argument("--width", type=int, default=1280, help="requested camera width")
    parser.add_argument("--height", type=int, default=720, help="requested camera height")
    parser.add_argument("--confirm-frames", type=int, default=2, help="frames required before showing a detection")
    parser.add_argument(
        "--ignore-zone", action="append", type=parse_zone, default=[], help="normalized x1,y1,x2,y2 zone"
    )
    parser.add_argument("--show-unheld-phones", action="store_true", help="show phones away from hands or people")
    orientation = parser.add_mutually_exclusive_group()
    orientation.add_argument("--mirror", action="store_true", help="mirror the camera like a selfie preview")
    orientation.add_argument(
        "--no-mirror", dest="mirror", action="store_false", help="use natural orientation (default)"
    )
    parser.set_defaults(mirror=False)
    parser.add_argument("--no-masks", action="store_true", help="start with segmentation masks hidden")
    parser.add_argument("--no-hands", action="store_true", help="start with hand landmarks hidden")
    parser.add_argument("--save-dir", type=Path, default=Path("runs/vision-camera"), help="screenshot directory")
    return parser.parse_args()


def pixel_box(box, width, height):
    """Clip an xyxy box to integer frame coordinates."""
    x1, y1, x2, y2 = box
    return (
        max(0, min(width - 1, int(x1))),
        max(0, min(height - 1, int(y1))),
        max(0, min(width - 1, int(x2))),
        max(0, min(height - 1, int(y2))),
    )


def box_center(box):
    """Return the center of an xyxy box."""
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def point_in_ignored_zone(point, zones, width, height):
    """Return whether a pixel point falls inside a configured ignore zone."""
    x, y = point
    return any(x1 * width <= x <= x2 * width and y1 * height <= y <= y2 * height for x1, y1, x2, y2 in zones)


def hand_observations(result, width, height, mirrored):
    """Convert MediaPipe output to labeled hand observations."""
    observations = []
    for image_hand, handedness in zip(result.hand_landmarks, result.handedness):
        points = [(landmark.x * width, landmark.y * height) for landmark in image_hand]
        x_values, y_values = zip(*points)
        label = handedness[0].category_name if handedness else "Hand"
        if mirrored and label in {"Left", "Right"}:
            label = "Right" if label == "Left" else "Left"
        observations.append(
            HandObservation(
                label,
                (min(x_values), min(y_values), max(x_values), max(y_values)),
                image_hand,
            )
        )
    return observations


def box_intersection(first, second):
    """Return the intersection area of two xyxy boxes."""
    width = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    return width * height


def nearest_hand(box, hands):
    """Return the label of a hand touching or immediately surrounding an object."""
    object_center = box_center(box)
    closest = None
    closest_distance = math.inf
    for hand in hands:
        hand_width = max(1, hand.box[2] - hand.box[0])
        hand_height = max(1, hand.box[3] - hand.box[1])
        expanded = (
            hand.box[0] - hand_width * 0.65,
            hand.box[1] - hand_height * 0.65,
            hand.box[2] + hand_width * 0.65,
            hand.box[3] + hand_height * 0.65,
        )
        hand_center = box_center(hand.box)
        distance = math.dist(object_center, hand_center)
        touching = box_intersection(box, expanded) > 0
        if touching and distance < closest_distance:
            closest, closest_distance = hand.label, distance
    return closest


def overlaps_person(det, people):
    """Return whether an object overlaps a person's segmentation mask."""
    x1, y1, x2, y2 = det.box
    area = max(1, (x2 - x1) * (y2 - y1))
    return any(
        np.count_nonzero(person.mask[y1:y2, x1:x2]) / area >= 0.05 for person in people if person.mask is not None
    )


def mask_from_result(result, index, frame_shape):
    """Rasterize one result polygon into original-frame coordinates."""
    if result.masks is None or index >= len(result.masks.xy):
        return None
    polygon = result.masks.xy[index]
    if len(polygon) < 3:
        return None
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
    return mask


def extract_detections(result, frame_shape, zones):
    """Apply class thresholds and ignore zones to raw YOLOE results."""
    height, width = frame_shape[:2]
    detections = []
    if result.boxes is None:
        return detections
    for index, box in enumerate(result.boxes):
        class_id = int(box.cls.item())
        name = result.names[class_id]
        confidence = float(box.conf.item())
        if confidence < THRESHOLDS.get(name, 0.25):
            continue
        bounds = pixel_box(box.xyxy[0].tolist(), width, height)
        if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
            continue
        if point_in_ignored_zone(box_center(bounds), zones, width, height):
            continue
        track_id = int(box.id.item()) if box.id is not None else None
        fallback = (int(box_center(bounds)[0] // 80), int(box_center(bounds)[1] // 80))
        key = (name, track_id) if track_id is not None else (name, *fallback)
        detections.append(
            Detection(name, confidence, bounds, mask_from_result(result, index, frame_shape), track_id, key)
        )
    return detections


def confirmed_detections(detections, states, frame_index, required_hits):
    """Update temporal state and return detections stable for the requested number of frames."""
    visible = []
    for detection in detections:
        previous = states.get(detection.key)
        hits = previous.hits + 1 if previous and previous.last_frame == frame_index - 1 else 1
        states[detection.key] = TrackState(hits, frame_index)
        if hits >= required_hits:
            visible.append(detection)
    for key in [key for key, state in states.items() if frame_index - state.last_frame > 12]:
        del states[key]
    return visible


def apply_context(detections, hands, show_unheld_phones):
    """Associate objects with hands and suppress implausible background phone detections."""
    people = [detection for detection in detections if detection.name == "person"]
    filtered = []
    for detection in detections:
        if detection.name in {"cell phone", "over-ear headphones"}:
            detection.hand = nearest_hand(detection.box, hands) or ""
        if (
            detection.name == "cell phone"
            and not show_unheld_phones
            and not detection.hand
            and not overlaps_person(detection, people)
        ):
            continue
        filtered.append(detection)
    return filtered


def draw_detections(frame, detections, show_masks):
    """Render fixed-color masks, boxes, track IDs, and hand relationships."""
    if show_masks:
        overlay = frame.copy()
        for detection in detections:
            if detection.mask is not None:
                overlay[detection.mask.astype(bool)] = COLORS.get(detection.name, (255, 255, 255))
        frame[:] = cv2.addWeighted(overlay, 0.38, frame, 0.62, 0)

    for detection in detections:
        color = COLORS.get(detection.name, (255, 255, 255))
        x1, y1, x2, y2 = detection.box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
        relation = f" in {detection.hand} hand" if detection.hand else ""
        track = f" #{detection.track_id}" if detection.track_id is not None else ""
        label = f"{detection.name}{relation}{track} {detection.confidence:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        label_bottom = max(y1, text_height + 12)
        label_top = label_bottom - text_height - 12
        cv2.rectangle(
            frame,
            (x1, label_top),
            (min(frame.shape[1] - 1, x1 + text_width + 10), label_bottom),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 5, label_bottom - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def draw_ignore_zones(frame, zones):
    """Draw configured background areas that do not produce detections."""
    height, width = frame.shape[:2]
    for x1, y1, x2, y2 in zones:
        start, end = (int(x1 * width), int(y1 * height)), (int(x2 * width), int(y2 * height))
        cv2.rectangle(frame, start, end, (100, 100, 100), 2)
        cv2.putText(frame, "IGNORE", (start[0] + 6, start[1] + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 2)


def draw_hud(frame, fps, device, show_masks, show_hands, phone_filter):
    """Draw a compact status panel and keyboard help."""
    lines = [
        f"Vision Camera  |  {fps:.1f} FPS  |  {device}",
        f"Masks {'ON' if show_masks else 'OFF'}   Hands {'ON' if show_hands else 'OFF'}   Held-phone filter {'ON' if phone_filter else 'OFF'}",
    ]
    overlay = frame.copy()
    cv2.rectangle(overlay, (15, 15), (640, 86), (15, 15, 15), -1)
    frame[:] = cv2.addWeighted(overlay, 0.78, frame, 0.22, 0)
    for index, line in enumerate(lines):
        cv2.putText(frame, line, (30, 43 + index * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        "q quit   f flip   m masks   h hands   p phone filter   s screenshot",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def run(opt):
    """Run the unified camera loop."""
    if opt.confirm_frames < 1:
        raise ValueError("--confirm-frames must be at least 1")
    device = opt.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    source = int(opt.source) if str(opt.source).isdigit() else opt.source

    print(f"Loading {opt.model} on {device} with classes: {', '.join(PROMPTS)}")
    object_model = YOLOE(opt.model)
    object_model.set_classes(list(PROMPTS))
    hand_options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path())),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.45,
        min_hand_presence_confidence=0.45,
        min_tracking_confidence=0.45,
    )

    camera = cv2.VideoCapture(source)
    if isinstance(source, int):
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, opt.width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, opt.height)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Could not open source {source}. Check camera permissions or try --source 1.")

    states = {}
    frame_index = 0
    timestamp_ms = 0
    fps, previous_time = 0.0, time.perf_counter()
    show_masks, show_hands = not opt.no_masks, not opt.no_hands
    phone_filter = not opt.show_unheld_phones
    mirrored = opt.mirror
    window = "Vision Camera"

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(hand_options) as hand_model:
            while True:
                success, frame = camera.read()
                if not success:
                    break
                if isinstance(source, int) and mirrored:
                    frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp_ms = max(timestamp_ms + 1, time.monotonic_ns() // 1_000_000)
                hand_result = hand_model.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms
                )
                hands = hand_observations(hand_result, frame.shape[1], frame.shape[0], mirrored)

                object_result = object_model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=0.05,
                    iou=0.55,
                    imgsz=opt.imgsz,
                    device=device,
                    max_det=40,
                    verbose=False,
                )[0]
                detections = extract_detections(object_result, frame.shape, opt.ignore_zone)
                detections = confirmed_detections(detections, states, frame_index, opt.confirm_frames)
                detections = apply_context(detections, hands, not phone_filter)

                draw_detections(frame, detections, show_masks)
                if show_hands:
                    for index, hand in enumerate(hands):
                        draw_hand(
                            frame,
                            hand.image_landmarks,
                            hand.label,
                            HAND_COLORS[index % len(HAND_COLORS)],
                        )
                draw_ignore_zones(frame, opt.ignore_zone)

                current_time = time.perf_counter()
                instantaneous_fps = 1 / max(current_time - previous_time, 1e-6)
                fps = instantaneous_fps if fps == 0 else fps * 0.9 + instantaneous_fps * 0.1
                previous_time = current_time
                draw_hud(frame, fps, device, show_masks, show_hands, phone_filter)
                cv2.imshow(window, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in {ord("q"), 27}:
                    break
                if key == ord("m"):
                    show_masks = not show_masks
                elif key == ord("h"):
                    show_hands = not show_hands
                elif key == ord("p"):
                    phone_filter = not phone_filter
                elif key == ord("f"):
                    mirrored = not mirrored
                elif key == ord("s"):
                    opt.save_dir.mkdir(parents=True, exist_ok=True)
                    output = opt.save_dir / f"vision-{time.strftime('%Y%m%d-%H%M%S')}.jpg"
                    cv2.imwrite(str(output), frame)
                    print(f"Saved {output}")
                frame_index += 1
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run(parse_opt())
