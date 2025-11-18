import os
import json
from typing import List, Dict, Tuple

import numpy as np
import parselmouth


ALIGN_DIR = os.path.join("static", "json", "aligned_pitch")


def _extract_pitch_series(audio_path: str) -> Tuple[List[float], float]:
    """
    Extract raw F0 series (Hz) and duration from a WAV using Parselmouth.

    Returns a list of floats (0.0 for unvoiced frames) and the total duration in seconds.
    """
    snd = parselmouth.Sound(audio_path)
    pitch = snd.to_pitch()
    freqs = pitch.selected_array["frequency"]
    # replace negatives with 0.0 for consistency
    series = [float(f) if f > 0 else 0.0 for f in freqs]
    return series, snd.get_total_duration()


def _resample(series: List[float], target_len: int) -> np.ndarray:
    """
    Resample a 1D series to a fixed length using linear interpolation.

    Zeros (unvoiced) are kept as 0.0 but still participate in interpolation.
    """
    if len(series) == 0:
        return np.zeros(target_len, dtype=float)
    if len(series) == 1:
        return np.full(target_len, series[0], dtype=float)

    x_old = np.linspace(0.0, 1.0, num=len(series))
    x_new = np.linspace(0.0, 1.0, num=target_len)
    return np.interp(x_new, x_old, np.asarray(series, dtype=float))


def _normalize_log_pitch(series: np.ndarray) -> np.ndarray:
    """
    Convert to log Hz and z-normalize voiced frames only.

    Unvoiced frames (<= 0) are ignored for statistics and set to 0 after normalization.
    """
    series = np.asarray(series, dtype=float)
    voiced_mask = series > 0
    if not np.any(voiced_mask):
        return np.zeros_like(series)

    log_series = np.zeros_like(series)
    log_series[voiced_mask] = np.log(series[voiced_mask])

    mean = log_series[voiced_mask].mean()
    std = log_series[voiced_mask].std()
    if std < 1e-6:
        std = 1e-6

    norm = np.zeros_like(series)
    norm[voiced_mask] = (log_series[voiced_mask] - mean) / std
    return norm


def _score_pair(native: List[float], learner: List[float]) -> Tuple[float, str]:
    """
    Compute a similarity score (0–100) and pitch direction hint.

    Score is based on mean absolute error between normalized log-pitch contours.
    Pitch direction is 'up', 'down', or 'ok' based on average F0 difference.
    """
    if len(native) < 3 or len(learner) < 3:
        return 0.0, "unknown"

    target_len = 50
    n_res = _normalize_log_pitch(_resample(native, target_len))
    l_res = _normalize_log_pitch(_resample(learner, target_len))

    mae = float(np.mean(np.abs(n_res - l_res)))
    # Heuristic: treat mae >= 1.5 as "very different"
    max_mae = 1.5
    raw_score = max(0.0, 1.0 - mae / max_mae)
    score = raw_score * 100.0

    # Direction hint based on average F0 (not log, in Hz)
    n_mean = np.mean([f for f in native if f > 0])
    l_mean = np.mean([f for f in learner if f > 0]) if any(f > 0 for f in learner) else 0.0
    diff = l_mean - n_mean

    # Threshold ~30 Hz as a simple heuristic
    if l_mean == 0 or n_mean == 0:
        direction = "unknown"
    elif diff > 30:
        direction = "down"  # learner too high, should go down
    elif diff < -30:
        direction = "up"    # learner too low, should go up
    else:
        direction = "ok"

    return float(score), direction

def _color_for_score(score, mode="normal"):
    """
    Map a per-segment score to a color.
    Thresholds are mode-dependent.
    """
    if mode == "lenient":
        green_thr = 70
        yellow_thr = 50
    elif mode == "strict":
        green_thr = 85
        yellow_thr = 65
    else:  # normal
        green_thr = 80
        yellow_thr = 60

    if score >= green_thr:
        return "green"
    elif score >= yellow_thr:
        return "yellow"
    else:
        return "red"


def _segment_score(native_curve, learner_curve, mode="normal"):
    """
    Compute a score [0,100] for one character/segment based on
    shape/direction and magnitude. This is just a template.
    Plug in your own DTW/spacing metrics if you already have them.
    """

    # Basic sanity
    if len(native_curve) == 0 or len(learner_curve) == 0:
        return 0.0, "unknown"

    native = np.array(native_curve, dtype=float)
    learner = np.array(learner_curve, dtype=float)

    # Filter NaNs / nulls
    mask = ~np.isnan(native) & ~np.isnan(learner)
    if mask.sum() < 3:
        return 0.0, "unknown"

    native = native[mask]
    learner = learner[mask]

    # Normalize (z-score) within segment so we compare shape
    native_z = (native - native.mean()) / (native.std() + 1e-6)
    learner_z = (learner - learner.mean()) / (learner.std() + 1e-6)

    # --- 1. Direction: rising/falling/flat ---
    native_delta = native[-1] - native[0]
    learner_delta = learner[-1] - learner[0]

    def _dir(delta, eps=0.5):
        if delta > eps:
            return "up"
        elif delta < -eps:
            return "down"
        else:
            return "level"

    native_dir = _dir(native_delta)
    learner_dir = _dir(learner_delta)

    # --- 2. Shape similarity (cosine) ---
    cos_sim = float(
        np.dot(native_z, learner_z) /
        (np.linalg.norm(native_z) * np.linalg.norm(learner_z) + 1e-6)
    )
    # map [-1,1] -> [0,100]
    shape_score = 50 * (cos_sim + 1)

    # --- 3. Magnitude deviation (how far off in semitones) ---
    mag_diff = float(np.mean(np.abs(native - learner)))
    # mode-dependent penalty strength
    if mode == "lenient":
        penalty_per_semi = 5.0
        max_penalty = 20.0
    elif mode == "strict":
        penalty_per_semi = 10.0
        max_penalty = 35.0
    else:
        penalty_per_semi = 7.5
        max_penalty = 25.0

    mag_penalty = min(mag_diff * penalty_per_semi, max_penalty)

    # --- 4. Direction correctness boost/penalty ---
    if native_dir == "level":
        dir_score = 100.0  # any stable-ish pitch is ok
    elif native_dir == learner_dir:
        dir_score = 100.0
    else:
        # mismatch is penalized less in lenient mode
        dir_penalty = 25.0 if mode == "strict" else 15.0
        dir_score = 100.0 - dir_penalty

    # combine
    if mode == "lenient":
        w_shape, w_dir, w_mag = 0.5, 0.3, 0.2
    elif mode == "strict":
        w_shape, w_dir, w_mag = 0.6, 0.3, 0.1
    else:  # normal
        w_shape, w_dir, w_mag = 0.55, 0.3, 0.15

    base_score = (
        w_shape * shape_score +
        w_dir * dir_score -
        w_mag * mag_penalty
    )

    base_score = max(0.0, min(100.0, base_score))

    # map direction for UI arrows
    if learner_dir == "up":
        pitch_direction = "up"
    elif learner_dir == "down":
        pitch_direction = "down"
    elif learner_dir == "level":
        pitch_direction = "ok"
    else:
        pitch_direction = "unknown"

    return base_score, pitch_direction


def evaluate_pronunciation(sentence_id, mode="normal"):
    """
    Top-level evaluation function used by the Flask app.
    Loads native + learner pitch for `sentence_id`,
    computes per-character scores, and returns JSON-ready dict.
    """

    # --- load your alignment & pitch here (pseudo-code) ---
    # alignment = load_alignment_for_sentence(sentence_id)
    #   -> list of segments: { "text": "你", "native_pitch": [...], "learner_pitch": [...] }
    alignment = load_alignment_for_sentence(sentence_id)  # your existing function

    segments_out = []
    segment_scores = []

    for seg in alignment:
        native_curve = seg.get("native_pitch", [])
        learner_curve = seg.get("learner_pitch", [])

        score, pitch_direction = _segment_score(native_curve, learner_curve, mode=mode)
        color = _color_for_score(score, mode=mode)

        segments_out.append({
            "text": seg.get("text", ""),
            "score": score,
            "pitch_direction": pitch_direction,
            "color": color
        })
        segment_scores.append(score)

    overall = float(np.mean(segment_scores)) if segment_scores else 0.0

    return {
        "mode": mode,
        "overall_score": overall,
        "segments": segments_out
    }



# def evaluate_pronunciation(base_id: str, learner_audio_path: str) -> Dict:
#     """
#     Evaluate learner pronunciation for a given sentence.

#     Parameters
#     ----------
#     base_id : str
#         Sentence/audio ID, e.g. 'm1' or 'f3'. Must match the prefix of the
#         aligned_pitch JSON: `<base_id>_aligned_pitch.json`.
#     learner_audio_path : str
#         Path to the learner's WAV recording, e.g. `static/learner_audio/m1_learner.wav`.

#     Returns
#     -------
#     dict
#         {
#           "overall_score": float,
#           "segments": [
#             {
#               "index": int,
#               "text": str,
#               "start": float,
#               "end": float,
#               "score": float,
#               "color": "green|yellow|red",
#               "pitch_direction": "up|down|ok|unknown",
#             }, ...
#           ]
#         }
#     """
#     align_path = os.path.join(ALIGN_DIR, f"{base_id}_aligned_pitch.json")
#     if not os.path.exists(align_path):
#         raise FileNotFoundError(f"Aligned pitch file not found: {align_path}")

#     if not os.path.exists(learner_audio_path):
#         raise FileNotFoundError(f"Learner audio not found: {learner_audio_path}")

#     with open(align_path, "r", encoding="utf-8") as f:
#         alignment = json.load(f)

#     learner_series, learner_duration = _extract_pitch_series(learner_audio_path)
#     if len(learner_series) == 0 or learner_duration <= 0:
#         raise RuntimeError("Could not extract pitch from learner audio")

#     time_step = learner_duration / len(learner_series)

#     segment_results = []
#     scores = []

#     for idx, seg in enumerate(alignment):
#         start = float(seg.get("start", 0.0))
#         end = float(seg.get("end", 0.0))
#         text = seg.get("text", "")

#         native = seg.get("pitch_segment", []) or []
#         # ensure native is list of floats
#         native = [float(hz) for hz in native if hz is not None and hz > 0]

#         if end <= start or len(native) < 3:
#             # Not enough information; mark as low confidence
#             seg_result = {
#                 "index": idx,
#                 "text": text,
#                 "start": start,
#                 "end": end,
#                 "score": 0.0,
#                 "color": "red",
#                 "pitch_direction": "unknown",
#             }
#             segment_results.append(seg_result)
#             scores.append(0.0)
#             continue

#         start_idx = max(0, int(start / time_step))
#         end_idx = min(len(learner_series), int(end / time_step))

#         learner_slice = [
#             float(hz) for hz in learner_series[start_idx:end_idx]
#             if hz is not None and hz > 0
#         ]

#         if len(learner_slice) < 3:
#             score, direction = 0.0, "unknown"
#         else:
#             score, direction = _score_pair(native, learner_slice)

#         if score >= 80:
#             color = "green"
#         elif score >= 60:
#             color = "yellow"
#         else:
#             color = "red"

#         segment_results.append(
#             {
#                 "index": idx,
#                 "text": text,
#                 "start": start,
#                 "end": end,
#                 "score": round(score, 1),
#                 "color": color,
#                 "pitch_direction": direction,
#             }
#         )
#         scores.append(score)

#     overall = float(np.mean(scores)) if scores else 0.0

#     return {
#         "overall_score": round(overall, 1),
#         "segments": segment_results,
#     }
