"""Evaluate the shared vision-camera pipeline against annotated recorded scenarios."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2
import psutil
import torch

from hand_track import extended_fingers
from vision_camera import PROMPTS, THRESHOLDS, PerceptionPipeline


def parse_opt():
    """Parse benchmark options."""
    parser = argparse.ArgumentParser(description="Run the vision camera against recorded annotated scenarios.")
    parser.add_argument("--manifest", type=Path, default=Path("vision_camera_scenarios.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/vision-camera-benchmark/results.json"))
    parser.add_argument("--device", help="override the manifest device")
    parser.add_argument("--validate-only", action="store_true", help="validate annotations without loading models")
    return parser.parse_args()


def validate_box(box, context):
    """Validate one xyxy annotation box."""
    if not isinstance(box, list) or len(box) != 4 or not all(isinstance(value, (int, float)) for value in box):
        raise ValueError(f"{context} box must contain four numbers")
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError(f"{context} box must have x2/y2 greater than x1/y1")


def load_manifest(path):
    """Load and validate the versioned scenario manifest."""
    with path.open(encoding="utf-8") as file:
        manifest = json.load(file)
    if manifest.get("version") != 1:
        raise ValueError("manifest version must be 1")
    scenarios = manifest.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("manifest must contain at least one scenario")
    identifiers = set()
    for scenario in scenarios:
        identifier = scenario.get("id")
        if not identifier or identifier in identifiers:
            raise ValueError(f"scenario id must be present and unique: {identifier!r}")
        identifiers.add(identifier)
        if not scenario.get("clip") or not scenario.get("tags"):
            raise ValueError(f"scenario {identifier} must define clip and tags")
        frames = set()
        for annotation in scenario.get("annotations", []):
            frame = annotation.get("frame")
            if not isinstance(frame, int) or frame < 0 or frame in frames:
                raise ValueError(f"scenario {identifier} annotation frames must be unique non-negative integers")
            frames.add(frame)
            hand_ids = set()
            for hand in annotation.get("hands", []):
                if not hand.get("id") or hand["id"] in hand_ids:
                    raise ValueError(f"scenario {identifier} hand ids must be present and unique per frame")
                hand_ids.add(hand["id"])
                validate_box(hand.get("box"), f"scenario {identifier} hand {hand['id']}")
            object_ids = set()
            for item in annotation.get("objects", []):
                if not item.get("id") or item["id"] in object_ids:
                    raise ValueError(f"scenario {identifier} object ids must be present and unique per frame")
                object_ids.add(item["id"])
                if item.get("name") not in PROMPTS:
                    raise ValueError(f"scenario {identifier} object {item['id']} has an unsupported name")
                if item.get("held_by") and item["held_by"] not in hand_ids:
                    raise ValueError(f"scenario {identifier} object {item['id']} references an unknown hand")
                validate_box(item.get("box"), f"scenario {identifier} object {item['id']}")
    return manifest


def intersection_over_union(first, second):
    """Return IoU for two xyxy boxes."""
    width = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = width * height
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    return intersection / max(first_area + second_area - intersection, 1)


def match_boxes(expected, predicted, same_class=False, threshold=0.5):
    """Greedily match expected and predicted boxes above an IoU threshold."""
    candidates = []
    for expected_index, target in enumerate(expected):
        for predicted_index, observation in enumerate(predicted):
            if same_class and target["name"] != observation["name"]:
                continue
            overlap = intersection_over_union(target["box"], observation["box"])
            if overlap >= threshold:
                candidates.append((overlap, expected_index, predicted_index))
    matches, used_expected, used_predicted = [], set(), set()
    for overlap, expected_index, predicted_index in sorted(candidates, reverse=True):
        if expected_index not in used_expected and predicted_index not in used_predicted:
            matches.append((expected_index, predicted_index, overlap))
            used_expected.add(expected_index)
            used_predicted.add(predicted_index)
    return matches


def blank_counts():
    """Return raw metric counters."""
    return {
        "object_expected": 0,
        "object_predicted": 0,
        "object_matched": 0,
        "hand_matched": 0,
        "handedness_correct": 0,
        "finger_count_correct": 0,
        "finger_count_expected": 0,
        "association_correct": 0,
        "association_expected": 0,
        "identity_switches": 0,
    }


def evaluate_frame(annotation, objects, hands, counts, track_history):
    """Accumulate current-capability metrics for one annotated frame."""
    expected_objects = annotation.get("objects", [])
    expected_hands = annotation.get("hands", [])
    object_matches = match_boxes(expected_objects, objects, same_class=True)
    hand_matches = match_boxes(expected_hands, hands)
    counts["object_expected"] += len(expected_objects)
    counts["object_predicted"] += len(objects)
    counts["object_matched"] += len(object_matches)
    counts["hand_matched"] += len(hand_matches)

    hand_labels = {hand["id"]: hand.get("label") for hand in expected_hands}
    for expected_index, predicted_index, _ in object_matches:
        expected, predicted = expected_objects[expected_index], objects[predicted_index]
        track_id = predicted.get("track_id")
        previous_track = track_history.get(expected["id"])
        if previous_track is not None and track_id is not None and track_id != previous_track:
            counts["identity_switches"] += 1
        if track_id is not None:
            track_history[expected["id"]] = track_id
        if expected.get("held_by"):
            counts["association_expected"] += 1
            if predicted.get("hand") == hand_labels[expected["held_by"]]:
                counts["association_correct"] += 1

    for expected_index, predicted_index, _ in hand_matches:
        expected, predicted = expected_hands[expected_index], hands[predicted_index]
        if expected.get("label") == predicted.get("label"):
            counts["handedness_correct"] += 1
        if "raised_fingers" in expected:
            counts["finger_count_expected"] += 1
            if expected["raised_fingers"] == predicted["raised_fingers"]:
                counts["finger_count_correct"] += 1


def ratio(numerator, denominator):
    """Return a rounded ratio or None when no annotation supports it."""
    return round(numerator / denominator, 4) if denominator else None


def summarize(counts):
    """Convert raw counters into user-facing metrics."""
    true_positives = counts["object_matched"]
    summary = {
        "object_precision_iou_0_5": ratio(true_positives, counts["object_predicted"]),
        "object_recall_iou_0_5": ratio(true_positives, counts["object_expected"]),
        "handedness_accuracy": ratio(counts["handedness_correct"], counts["hand_matched"]),
        "exact_finger_count_accuracy": ratio(counts["finger_count_correct"], counts["finger_count_expected"]),
        "hand_object_association_accuracy": ratio(counts["association_correct"], counts["association_expected"]),
        "track_identity_switches": counts["identity_switches"],
    }
    return summary


def serialize_detections(detections):
    """Convert detections to JSON-safe state."""
    return [
        {
            "name": detection.name,
            "confidence": round(detection.confidence, 6),
            "box": list(detection.box),
            "track_id": detection.track_id,
            "hand": detection.hand or None,
        }
        for detection in detections
    ]


def serialize_hands(hands):
    """Convert hands to JSON-safe state."""
    return [
        {
            "label": hand.label,
            "box": [round(value, 3) for value in hand.box],
            "raised_fingers": sum(extended_fingers(hand.image_landmarks)),
        }
        for hand in hands
    ]


def run_scenario(scenario, manifest_path, configuration, device):
    """Run one enabled recorded scenario and return predictions and metrics."""
    clip = (manifest_path.parent / scenario["clip"]).resolve()
    camera = cv2.VideoCapture(str(clip))
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Could not open scenario clip: {clip}")
    start_frame = scenario.get("start_frame", 0)
    end_frame = scenario.get("end_frame")
    camera.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    source_fps = camera.get(cv2.CAP_PROP_FPS) or 30.0
    source_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    annotations = {item["frame"]: item for item in scenario.get("annotations", [])}
    counts, track_history, frames, durations = blank_counts(), {}, [], []
    evaluated_annotations = set()
    process = psutil.Process()
    start_memory = process.memory_info().rss
    peak_memory = start_memory
    frame_index = start_frame
    timestamp_ms = max(0, round(start_frame * 1000 / source_fps))
    try:
        with PerceptionPipeline(
            configuration["model"],
            configuration["imgsz"],
            device,
            configuration["confirm_frames"],
            configuration.get("ignore_zones", []),
            configuration.get("show_unheld_phones", False),
            configuration.get("max_hands", 2),
        ) as pipeline:
            while end_frame is None or frame_index <= end_frame:
                success, frame = camera.read()
                if not success:
                    break
                mirrored = scenario.get("mirror", configuration.get("mirror", False))
                if mirrored:
                    frame = cv2.flip(frame, 1)
                started = time.perf_counter()
                detections, hands = pipeline.process(frame, frame_index, timestamp_ms, mirrored)
                durations.append((time.perf_counter() - started) * 1000)
                objects, hand_states = serialize_detections(detections), serialize_hands(hands)
                frames.append({"frame": frame_index, "objects": objects, "hands": hand_states})
                if frame_index in annotations:
                    evaluate_frame(annotations[frame_index], objects, hand_states, counts, track_history)
                    evaluated_annotations.add(frame_index)
                peak_memory = max(peak_memory, process.memory_info().rss)
                frame_index += 1
                timestamp_ms = max(timestamp_ms + 1, round(frame_index * 1000 / source_fps))
    finally:
        camera.release()
    if not frames:
        raise RuntimeError(f"Scenario {scenario['id']} did not produce any frames")
    if missing_annotations := annotations.keys() - evaluated_annotations:
        raise ValueError(
            f"Scenario {scenario['id']} did not process annotated frames: "
            f"{', '.join(str(frame) for frame in sorted(missing_annotations))}"
        )
    sorted_durations = sorted(durations)
    percentile_index = min(len(sorted_durations) - 1, int(len(sorted_durations) * 0.95)) if durations else 0
    return {
        "id": scenario["id"],
        "clip": scenario["clip"],
        "tags": scenario["tags"],
        "source": {
            "width": source_width,
            "height": source_height,
            "fps": round(source_fps, 3),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "mirror": scenario.get("mirror", configuration.get("mirror", False)),
        },
        "metrics": summarize(counts),
        "performance": {
            "frames": len(frames),
            "mean_frame_ms": round(sum(durations) / len(durations), 3) if durations else None,
            "p95_frame_ms": round(sorted_durations[percentile_index], 3) if durations else None,
            "effective_fps": round(1000 * len(durations) / sum(durations), 3) if sum(durations) else None,
            "peak_memory_increase_mb": round((peak_memory - start_memory) / (1024 * 1024), 3),
        },
        "frames": frames,
        "_counts": counts,
    }


def git_state():
    """Return the application commit and dirty state."""
    root = Path(__file__).resolve().parent
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True).stdout
    )
    return {"commit": commit, "dirty": dirty}


def run(opt):
    """Validate the manifest and run every enabled scenario."""
    manifest_path = opt.manifest.resolve()
    manifest = load_manifest(manifest_path)
    enabled = [scenario for scenario in manifest["scenarios"] if scenario.get("enabled", True)]
    configuration = manifest.get("configuration", {})
    required = {"model", "imgsz", "confirm_frames"}
    if missing_configuration := required - configuration.keys():
        raise ValueError(f"manifest configuration is missing: {', '.join(sorted(missing_configuration))}")
    if opt.validate_only:
        missing_clips = [
            scenario["clip"] for scenario in enabled if not (manifest_path.parent / scenario["clip"]).is_file()
        ]
        if missing_clips:
            raise FileNotFoundError(f"enabled scenario clips are missing: {', '.join(missing_clips)}")
        print(f"Validated {len(manifest['scenarios'])} scenarios; {len(enabled)} enabled")
        return
    if not enabled:
        raise ValueError("manifest has no enabled scenarios; record and annotate a clip, then set enabled to true")
    device = opt.device or configuration.get("device") or ("mps" if torch.backends.mps.is_available() else "cpu")
    scenarios = [run_scenario(scenario, manifest_path, configuration, device) for scenario in enabled]
    aggregate = blank_counts()
    for scenario in scenarios:
        for key, value in scenario.pop("_counts").items():
            aggregate[key] += value
    result = {
        "manifest_version": manifest["version"],
        "application": git_state(),
        "configuration": {
            **configuration,
            "device": device,
            "prompts": PROMPTS,
            "class_thresholds": THRESHOLDS,
            "tracker": "bytetrack.yaml",
        },
        "event_metrics": {"status": "not_implemented"},
        "aggregate_metrics": summarize(aggregate),
        "scenarios": scenarios,
    }
    opt.output.parent.mkdir(parents=True, exist_ok=True)
    with opt.output.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    print(f"Wrote {len(scenarios)} scenario results to {opt.output}")


if __name__ == "__main__":
    run(parse_opt())
