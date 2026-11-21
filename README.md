# Volleyball Spike Pose Tracker

Python tool for analyzing volleyball spikes in video with OpenCV and MediaPipe Pose.

It reads a video, tracks body landmarks frame-by-frame, writes an annotated video, and exports a CSV with spike-relevant measurements:

- dominant wrist reach above the shoulders
- dominant elbow and shoulder angles
- left/right knee angles for loading mechanics
- torso lean
- hip rise as a rough jump-height signal
- spike phase labels
- contact-candidate and best-contact-candidate flags

## Python Version

This project is pinned to Python 3.11.9 in `.python-version` and `pyproject.toml`.

Install Python 3.11.9 first, then confirm:

```powershell
python --version
```

Expected output:

```text
Python 3.11.9
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you use the Windows Python launcher, create the environment directly from Python 3.11:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
pip install -r requirements.txt
```

## Run

```powershell
python spike_pose_tracker.py --input "path\to\spike_video.mp4" --output "outputs\annotated_spike.mp4" --csv "outputs\spike_metrics.csv"
```

Use a known hitting side when possible:

```powershell
python spike_pose_tracker.py --input "path\to\spike_video.mp4" --side right
```

For a more accurate but slower model:

```powershell
python spike_pose_tracker.py --input "path\to\spike_video.mp4" --model-complexity 2
```

To show a preview while processing:

```powershell
python spike_pose_tracker.py --input "path\to\spike_video.mp4" --display
```

Press `q` in the preview window to stop early.

## CSV Columns To Watch

- `best_contact_candidate`: the single frame with the strongest spike score.
- `contact_candidate`: frames that look like likely ball-contact timing.
- `wrist_above_shoulder_body_lengths`: higher usually means better reach.
- `dominant_elbow_angle_deg`: near full extension is closer to 180 degrees.
- `hip_rise_body_lengths`: rough jump height relative to torso length.
- `left_knee_angle_deg` / `right_knee_angle_deg`: smaller values during load indicate deeper knee bend.

## Filming Tips

For better tracking, use a side or diagonal angle where the hitter is visible head-to-toe. Keep the athlete large in frame, avoid heavy motion blur, and avoid having other players cross directly in front of the hitter during the approach and jump.

MediaPipe Pose is a single-person pose tracker. If multiple athletes are in frame, crop the input video around the hitter first for more reliable results.
