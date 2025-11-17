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


def evaluate_pronunciation(base_id: str, learner_audio_path: str) -> Dict:
    """
    Evaluate learner pronunciation for a given sentence.

    Parameters
    ----------
    base_id : str
        Sentence/audio ID, e.g. 'm1' or 'f3'. Must match the prefix of the
        aligned_pitch JSON: `<base_id>_aligned_pitch.json`.
    learner_audio_path : str
        Path to the learner's WAV recording, e.g. `static/learner_audio/m1_learner.wav`.

    Returns
    -------
    dict
        {
          "overall_score": float,
          "segments": [
            {
              "index": int,
              "text": str,
              "start": float,
              "end": float,
              "score": float,
              "color": "green|yellow|red",
              "pitch_direction": "up|down|ok|unknown",
            }, ...
          ]
        }
    """
    align_path = os.path.join(ALIGN_DIR, f"{base_id}_aligned_pitch.json")
    if not os.path.exists(align_path):
        raise FileNotFoundError(f"Aligned pitch file not found: {align_path}")

    if not os.path.exists(learner_audio_path):
        raise FileNotFoundError(f"Learner audio not found: {learner_audio_path}")

    with open(align_path, "r", encoding="utf-8") as f:
        alignment = json.load(f)

    learner_series, learner_duration = _extract_pitch_series(learner_audio_path)
    if len(learner_series) == 0 or learner_duration <= 0:
        raise RuntimeError("Could not extract pitch from learner audio")

    time_step = learner_duration / len(learner_series)

    segment_results = []
    scores = []

    for idx, seg in enumerate(alignment):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        text = seg.get("text", "")

        native = seg.get("pitch_segment", []) or []
        # ensure native is list of floats
        native = [float(hz) for hz in native if hz is not None and hz > 0]

        if end <= start or len(native) < 3:
            # Not enough information; mark as low confidence
            seg_result = {
                "index": idx,
                "text": text,
                "start": start,
                "end": end,
                "score": 0.0,
                "color": "red",
                "pitch_direction": "unknown",
            }
            segment_results.append(seg_result)
            scores.append(0.0)
            continue

        start_idx = max(0, int(start / time_step))
        end_idx = min(len(learner_series), int(end / time_step))

        learner_slice = [
            float(hz) for hz in learner_series[start_idx:end_idx]
            if hz is not None and hz > 0
        ]

        if len(learner_slice) < 3:
            score, direction = 0.0, "unknown"
        else:
            score, direction = _score_pair(native, learner_slice)

        if score >= 80:
            color = "green"
        elif score >= 60:
            color = "yellow"
        else:
            color = "red"

        segment_results.append(
            {
                "index": idx,
                "text": text,
                "start": start,
                "end": end,
                "score": round(score, 1),
                "color": color,
                "pitch_direction": direction,
            }
        )
        scores.append(score)

    overall = float(np.mean(scores)) if scores else 0.0

    return {
        "overall_score": round(overall, 1),
        "segments": segment_results,
    }
