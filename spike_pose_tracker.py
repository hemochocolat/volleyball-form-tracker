#to start it  run, python spike_pose_tracker.py --input "video\YOUR_NEW_VIDEO.MOV" --output "outputs\(put whatever you want the file to be called here).mp4" --csv "outputs\new_video_metrics.csv"
#replace YOUR_NEW_VIDEO.MOV with whatever video you want (the file name)
#then run Start-Process "(here you put whatever is after --output on the line above)"
#if you run these, it should work.
#have fun!

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


LandmarkMap = Dict[str, Tuple[float, float, float]]


@dataclass
class FrameMetrics:
    frame: int
    time_s: float
    pose_detected: bool
    dominant_side: str
    hip_center_y_px: Optional[float] = None
    shoulder_center_y_px: Optional[float] = None
    dominant_wrist_y_px: Optional[float] = None
    wrist_above_shoulder_body_lengths: Optional[float] = None
    hip_rise_body_lengths: Optional[float] = None
    dominant_elbow_angle_deg: Optional[float] = None
    dominant_shoulder_angle_deg: Optional[float] = None
    left_knee_angle_deg: Optional[float] = None
    right_knee_angle_deg: Optional[float] = None
    torso_lean_deg: Optional[float] = None
    knee_bend_score: float = 0.0
    arm_extension_score: float = 0.0
    reach_score: float = 0.0
    jump_score: float = 0.0
    spike_score: float = 0.0
    spike_phase: str = "at_rest"
    contact_candidate: bool = False
    best_contact_candidate: bool = False


POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS
POSE_LANDMARK = mp.solutions.pose.PoseLandmark
DRAWING_UTILS = mp.solutions.drawing_utils

LANDMARK_DRAWING_SPEC = DRAWING_UTILS.DrawingSpec(
    color=(0, 255, 255),
    thickness=2,
    circle_radius=4,
)
CONNECTION_DRAWING_SPEC = DRAWING_UTILS.DrawingSpec(
    color=(20, 20, 20),
    thickness=6,
    circle_radius=2,
)


CSV_FIELDS = [
    "frame",
    "time_s",
    "pose_detected",
    "dominant_side",
    "hip_center_y_px",
    "shoulder_center_y_px",
    "dominant_wrist_y_px",
    "wrist_above_shoulder_body_lengths",
    "hip_rise_body_lengths",
    "dominant_elbow_angle_deg",
    "dominant_shoulder_angle_deg",
    "left_knee_angle_deg",
    "right_knee_angle_deg",
    "torso_lean_deg",
    "knee_bend_score",
    "arm_extension_score",
    "reach_score",
    "jump_score",
    "spike_score",
    "spike_phase",
    "contact_candidate",
    "best_contact_candidate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track volleyball spike poses in a video with MediaPipe Pose and OpenCV."
    )
    parser.add_argument("--input", required=True, help="Path to the input video.")
    parser.add_argument(
        "--output",
        default="annotated_spike.mp4",
        help="Path for the annotated output video. Use --no-video to skip writing video.",
    )
    parser.add_argument(
        "--csv",
        default="spike_metrics.csv",
        help="Path for per-frame spike metrics CSV.",
    )
    parser.add_argument(
        "--side",
        choices=["auto", "left", "right"],
        default="auto",
        help="Dominant hitting arm. Auto chooses the higher visible wrist each frame.",
    )
    parser.add_argument(
        "--model-complexity",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="MediaPipe Pose model complexity. 2 is slower but usually more accurate.",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence for initial pose detection.",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence for pose tracking between frames.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show a live preview window while processing.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Only write the CSV metrics, not an annotated video.",
    )
    return parser.parse_args()


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def safe_round(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(value, digits)


def angle_deg(
    a: Tuple[float, float],
    b: Tuple[float, float],
    c: Tuple[float, float],
) -> Optional[float]:
    ba = np.array([a[0] - b[0], a[1] - b[1]], dtype=np.float32)
    bc = np.array([c[0] - b[0], c[1] - b[1]], dtype=np.float32)
    norm = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if norm == 0.0:
        return None
    cosine = clamp(float(np.dot(ba, bc) / norm), -1.0, 1.0)
    return math.degrees(math.acos(cosine))


def point(
    landmarks: LandmarkMap,
    name: str,
    min_visibility: float = 0.45,
) -> Optional[Tuple[float, float]]:
    landmark = landmarks.get(name)
    if landmark is None:
        return None
    x, y, visibility = landmark
    if visibility < min_visibility:
        return None
    return x, y


def midpoint(
    a: Optional[Tuple[float, float]],
    b: Optional[Tuple[float, float]],
) -> Optional[Tuple[float, float]]:
    if a is None or b is None:
        return None
    return (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0


def distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def build_landmark_map(results, width: int, height: int) -> LandmarkMap:
    if not results.pose_landmarks:
        return {}

    output: LandmarkMap = {}
    for landmark in POSE_LANDMARK:
        lm = results.pose_landmarks.landmark[landmark.value]
        output[landmark.name.lower()] = (lm.x * width, lm.y * height, lm.visibility)
    return output


def choose_side(landmarks: LandmarkMap, requested_side: str) -> str:
    if requested_side != "auto":
        return requested_side

    left_wrist = point(landmarks, "left_wrist")
    right_wrist = point(landmarks, "right_wrist")
    if left_wrist is None and right_wrist is None:
        return "unknown"
    if right_wrist is None:
        return "left"
    if left_wrist is None:
        return "right"
    return "left" if left_wrist[1] < right_wrist[1] else "right"


def side_names(side: str) -> Tuple[str, str, str, str]:
    return (
        f"{side}_shoulder",
        f"{side}_elbow",
        f"{side}_wrist",
        f"{side}_hip",
    )


def compute_metrics(
    frame_index: int,
    fps: float,
    landmarks: LandmarkMap,
    requested_side: str,
    baseline_hip_y: Optional[float],
) -> Tuple[FrameMetrics, Optional[float]]:
    time_s = frame_index / fps if fps > 0 else 0.0
    if not landmarks:
        return FrameMetrics(frame_index, time_s, False, "unknown"), baseline_hip_y

    left_shoulder = point(landmarks, "left_shoulder")
    right_shoulder = point(landmarks, "right_shoulder")
    left_hip = point(landmarks, "left_hip")
    right_hip = point(landmarks, "right_hip")
    shoulder_center = midpoint(left_shoulder, right_shoulder)
    hip_center = midpoint(left_hip, right_hip)

    dominant_side = choose_side(landmarks, requested_side)
    if dominant_side == "unknown" or shoulder_center is None or hip_center is None:
        return FrameMetrics(frame_index, time_s, False, dominant_side), baseline_hip_y

    if baseline_hip_y is None:
        baseline_hip_y = hip_center[1]

    torso_length = max(distance(shoulder_center, hip_center), 1.0)
    shoulder_name, elbow_name, wrist_name, hip_name = side_names(dominant_side)
    dominant_shoulder = point(landmarks, shoulder_name)
    dominant_elbow = point(landmarks, elbow_name)
    dominant_wrist = point(landmarks, wrist_name)
    dominant_hip = point(landmarks, hip_name)

    elbow_angle = None
    shoulder_angle = None
    if dominant_shoulder and dominant_elbow and dominant_wrist:
        elbow_angle = angle_deg(dominant_shoulder, dominant_elbow, dominant_wrist)
    if dominant_elbow and dominant_shoulder and dominant_hip:
        shoulder_angle = angle_deg(dominant_elbow, dominant_shoulder, dominant_hip)

    left_knee_angle = joint_angle(landmarks, "left_hip", "left_knee", "left_ankle")
    right_knee_angle = joint_angle(landmarks, "right_hip", "right_knee", "right_ankle")
    visible_knees = [a for a in [left_knee_angle, right_knee_angle] if a is not None]
    average_knee_angle = sum(visible_knees) / len(visible_knees) if visible_knees else None

    torso_lean = torso_lean_deg(shoulder_center, hip_center)
    wrist_above_shoulder = None
    if dominant_wrist is not None:
        wrist_above_shoulder = (shoulder_center[1] - dominant_wrist[1]) / torso_length

    hip_rise = (baseline_hip_y - hip_center[1]) / torso_length
    knee_bend_score = 0.0
    if average_knee_angle is not None:
        knee_bend_score = clamp((165.0 - average_knee_angle) / 65.0)

    arm_extension_score = 0.0
    if elbow_angle is not None:
        arm_extension_score = clamp((elbow_angle - 120.0) / 60.0)

    reach_score = 0.0
    if wrist_above_shoulder is not None:
        reach_score = clamp((wrist_above_shoulder - 0.15) / 1.1)

    jump_score = clamp(hip_rise / 0.8)
    spike_score = (
        0.35 * reach_score
        + 0.25 * arm_extension_score
        + 0.25 * jump_score
        + 0.15 * knee_bend_score
    )
    contact_candidate = (
        spike_score >= 0.60
        and reach_score >= 0.45
        and arm_extension_score >= 0.45
        and jump_score >= 0.10
    )
    phase = classify_phase(
        average_knee_angle=average_knee_angle,
        wrist_above_shoulder=wrist_above_shoulder,
        hip_rise=hip_rise,
        elbow_angle=elbow_angle,
        contact_candidate=contact_candidate,
    )

    metrics = FrameMetrics(
        frame=frame_index,
        time_s=time_s,
        pose_detected=True,
        dominant_side=dominant_side,
        hip_center_y_px=hip_center[1],
        shoulder_center_y_px=shoulder_center[1],
        dominant_wrist_y_px=dominant_wrist[1] if dominant_wrist else None,
        wrist_above_shoulder_body_lengths=wrist_above_shoulder,
        hip_rise_body_lengths=hip_rise,
        dominant_elbow_angle_deg=elbow_angle,
        dominant_shoulder_angle_deg=shoulder_angle,
        left_knee_angle_deg=left_knee_angle,
        right_knee_angle_deg=right_knee_angle,
        torso_lean_deg=torso_lean,
        knee_bend_score=knee_bend_score,
        arm_extension_score=arm_extension_score,
        reach_score=reach_score,
        jump_score=jump_score,
        spike_score=spike_score,
        spike_phase=phase,
        contact_candidate=contact_candidate,
    )
    return metrics, baseline_hip_y


def joint_angle(
    landmarks: LandmarkMap,
    a_name: str,
    b_name: str,
    c_name: str,
) -> Optional[float]:
    a = point(landmarks, a_name)
    b = point(landmarks, b_name)
    c = point(landmarks, c_name)
    if a is None or b is None or c is None:
        return None
    return angle_deg(a, b, c)


def torso_lean_deg(
    shoulder_center: Tuple[float, float],
    hip_center: Tuple[float, float],
) -> float:
    dx = shoulder_center[0] - hip_center[0]
    dy = hip_center[1] - shoulder_center[1]
    return math.degrees(math.atan2(dx, dy))


def classify_phase(
    average_knee_angle: Optional[float],
    wrist_above_shoulder: Optional[float],
    hip_rise: float,
    elbow_angle: Optional[float],
    contact_candidate: bool,
) -> str:
    if contact_candidate:
        return "contact_candidate"
    if average_knee_angle is not None and average_knee_angle < 135 and hip_rise < 0.15:
        return "load"
    if hip_rise >= 0.15 and (wrist_above_shoulder or 0.0) < 0.35:
        return "takeoff"
    if (wrist_above_shoulder or 0.0) >= 0.35 and (elbow_angle or 0.0) < 150:
        return "ready-to-hit"
    if (wrist_above_shoulder or 0.0) >= 0.35 and (elbow_angle or 0.0) >= 150:
        return "reach"
    return "approach_or_reset"


def draw_overlay(
    frame,
    results,
    metrics: FrameMetrics,
    width: int,
    height: int,
) -> None:
    if results.pose_landmarks:
        DRAWING_UTILS.draw_landmarks(
            frame,
            results.pose_landmarks,
            POSE_CONNECTIONS,
            landmark_drawing_spec=LANDMARK_DRAWING_SPEC,
            connection_drawing_spec=CONNECTION_DRAWING_SPEC,
        )

    text_lines = [
        f"Frame {metrics.frame} | {metrics.time_s:.2f}s",
        f"Phase: {metrics.spike_phase}",
        f"Side: {metrics.dominant_side}",
        f"Spike score: {metrics.spike_score:.2f}",
    ]
    if metrics.best_contact_candidate:
        text_lines.append("BEST CONTACT CANDIDATE")
    elif metrics.contact_candidate:
        text_lines.append("Contact candidate")

    x = 16
    y = 32
    for line in text_lines:
        cv2.putText(
            frame,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 30

    draw_score_bar(frame, metrics.spike_score, width, height)


def draw_score_bar(frame, score: float, width: int, height: int) -> None:
    bar_width = min(360, width - 32)
    bar_height = 14
    x = 16
    y = height - 32
    fill = int(bar_width * clamp(score))
    cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (40, 40, 40), -1)
    cv2.rectangle(frame, (x, y), (x + fill, y + bar_height), (80, 220, 120), -1)
    cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (230, 230, 230), 1)


def mark_best_contact(metrics: List[FrameMetrics]) -> None:
    candidates = [m for m in metrics if m.contact_candidate]
    if not candidates:
        candidates = [m for m in metrics if m.pose_detected]
    if not candidates:
        return
    best = max(candidates, key=lambda m: m.spike_score)
    best.best_contact_candidate = True


def write_csv(path: Path, metrics: Iterable[FrameMetrics]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in metrics:
            writer.writerow(
                {
                    "frame": row.frame,
                    "time_s": safe_round(row.time_s, 4),
                    "pose_detected": row.pose_detected,
                    "dominant_side": row.dominant_side,
                    "hip_center_y_px": safe_round(row.hip_center_y_px, 3),
                    "shoulder_center_y_px": safe_round(row.shoulder_center_y_px, 3),
                    "dominant_wrist_y_px": safe_round(row.dominant_wrist_y_px, 3),
                    "wrist_above_shoulder_body_lengths": safe_round(
                        row.wrist_above_shoulder_body_lengths, 4
                    ),
                    "hip_rise_body_lengths": safe_round(row.hip_rise_body_lengths, 4),
                    "dominant_elbow_angle_deg": safe_round(
                        row.dominant_elbow_angle_deg, 3
                    ),
                    "dominant_shoulder_angle_deg": safe_round(
                        row.dominant_shoulder_angle_deg, 3
                    ),
                    "left_knee_angle_deg": safe_round(row.left_knee_angle_deg, 3),
                    "right_knee_angle_deg": safe_round(row.right_knee_angle_deg, 3),
                    "torso_lean_deg": safe_round(row.torso_lean_deg, 3),
                    "knee_bend_score": safe_round(row.knee_bend_score, 4),
                    "arm_extension_score": safe_round(row.arm_extension_score, 4),
                    "reach_score": safe_round(row.reach_score, 4),
                    "jump_score": safe_round(row.jump_score, 4),
                    "spike_score": safe_round(row.spike_score, 4),
                    "spike_phase": row.spike_phase,
                    "contact_candidate": row.contact_candidate,
                    "best_contact_candidate": row.best_contact_candidate,
                }
            )


def process_video(args: argparse.Namespace) -> List[FrameMetrics]:
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if not args.no_video:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not create output video: {output_path}")

    metrics_by_frame: List[FrameMetrics] = []
    annotated_frames: List[np.ndarray] = []
    baseline_hip_y = None

    with mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=args.model_complexity,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    ) as pose:
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = pose.process(rgb)
            landmarks = build_landmark_map(results, width, height)
            metrics, baseline_hip_y = compute_metrics(
                frame_index, fps, landmarks, args.side, baseline_hip_y
            )
            metrics_by_frame.append(metrics)

            if writer is not None or args.display:
                annotated = frame.copy()
                draw_overlay(annotated, results, metrics, width, height)
                if writer is not None:
                    annotated_frames.append(annotated)
                if args.display:
                    cv2.imshow("Volleyball Spike Pose Tracker", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            frame_index += 1

    capture.release()
    if args.display:
        cv2.destroyAllWindows()

    mark_best_contact(metrics_by_frame)

    if writer is not None:
        for annotated, metrics in zip(annotated_frames, metrics_by_frame):
            if metrics.best_contact_candidate:
                cv2.rectangle(
                    annotated,
                    (8, 8),
                    (width - 8, height - 8),
                    (0, 255, 255),
                    5,
                )
                cv2.putText(
                    annotated,
                    "BEST CONTACT CANDIDATE",
                    (16, min(height - 56, 190)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 0),
                    5,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    "BEST CONTACT CANDIDATE",
                    (16, min(height - 56, 190)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            writer.write(annotated)
        writer.release()

    write_csv(Path(args.csv), metrics_by_frame)
    return metrics_by_frame


def summarize(metrics: List[FrameMetrics]) -> str:
    detected = [row for row in metrics if row.pose_detected]
    if not metrics:
        return "No frames were processed."
    if not detected:
        return f"Processed {len(metrics)} frames, but no pose was detected."

    best = max(detected, key=lambda row: row.spike_score)
    return (
        f"Processed {len(metrics)} frames. Pose detected in {len(detected)} frames. "
        f"Best contact candidate: frame {best.frame} at {best.time_s:.2f}s "
        f"with spike score {best.spike_score:.2f}."
    )


def main() -> None:
    args = parse_args()
    metrics = process_video(args)
    print(summarize(metrics))
    print(f"Metrics CSV: {Path(args.csv).resolve()}")
    if not args.no_video:
        print(f"Annotated video: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
