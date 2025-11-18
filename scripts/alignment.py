import os
import json
from typing import List, Dict, Tuple

import numpy as np
import parselmouth
import librosa  # DTW


# Paths
ALIGN_DIR = os.path.join("static", "json", "aligned_pitch")
SENTENCES_JSON = os.path.join("static", "json", "sentences.json")
LEARNER_AUDIO_DIR = os.path.join("static", "learner_audio")


def _extract_pitch_series(audio_path: str) -> Tuple[List[float], float]:
    """
    Extract raw F0 series (Hz) and duration from a WAV using Parselmouth.
    Returns (list_of_Hz_or_0_for_unvoiced, duration_seconds).
    """
    snd = parselmouth.Sound(audio_path)
    pitch = snd.to_pitch()
    freqs = pitch.selected_array["frequency"]
    series = [float(f) if f > 0 else 0.0 for f in freqs]
    return series, snd.get_total_duration()


def _resample(series: List[float], target_len: int) -> np.ndarray:
    """
    Resample a 1D series to a fixed length using linear interpolation.
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
    Optional whole-sentence score (currently unused). Kept for future use.
    """
    if len(native) < 3 or len(learner) < 3:
        return 0.0, "unknown"

    target_len = 50
    n_res = _normalize_log_pitch(_resample(native, target_len))
    l_res = _normalize_log_pitch(_resample(learner, target_len))

    mae = float(np.mean(np.abs(n_res - l_res)))
    max_mae = 1.5
    raw_score = max(0.0, 1.0 - mae / max_mae)
    score = raw_score * 100.0

    n_mean = np.mean([f for f in native if f > 0])
    l_mean = np.mean([f for f in learner if f > 0]) if any(f > 0 for f in learner) else 0.0
    diff = l_mean - n_mean

    if l_mean == 0 or n_mean == 0:
        direction = "unknown"
    elif diff > 30:
        direction = "down"
    elif diff < -30:
        direction = "up"
    else:
        direction = "ok"

    return float(score), direction


def _color_for_score(score, mode="normal"):
    """
    Map a per-segment score to a color.
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


def _dtw_shape_score(native_z: np.ndarray, learner_z: np.ndarray) -> float:
    """
    Use librosa's DTW on z-normalized shapes to get a 0–100 shape score.
    Lower cost => higher score.
    """
    X = native_z.reshape(1, -1)
    Y = learner_z.reshape(1, -1)

    D, wp = librosa.sequence.dtw(X=X, Y=Y, metric="euclidean")
    total_cost = float(D[-1, -1])
    path_len = len(wp) if len(wp) > 0 else max(len(native_z), len(learner_z))

    avg_cost = total_cost / (path_len + 1e-6)

    max_cost = 2.0
    raw = max(0.0, 1.0 - avg_cost / max_cost)
    return raw * 100.0


def _segment_score(native_curve, learner_curve, mode="normal"):
    """
    Compute a score [0,100] for one character/segment based on
    shape/direction and magnitude.

    Handles unequal-length curves by trimming both to the same length.
    Uses z-normalized shapes and DTW for contour similarity.
    """
    # Basic sanity
    if len(native_curve) == 0 or len(learner_curve) == 0:
        return 0.0, "unknown"

    native = np.array(native_curve, dtype=float)
    learner = np.array(learner_curve, dtype=float)

    # 🔹 Make sure arrays have the same length before any elementwise ops
    min_len = min(len(native), len(learner))
    if min_len < 3:
        return 0.0, "unknown"

    native = native[:min_len]
    learner = learner[:min_len]

    # Treat non-positive as unvoiced
    native[native <= 0] = np.nan
    learner[learner <= 0] = np.nan

    # Filter out positions where either is NaN
    mask = ~np.isnan(native) & ~np.isnan(learner)
    native = native[mask]
    learner = learner[mask]

    if len(native) < 3 or len(learner) < 3:
        return 0.0, "unknown"

    # z-normalize within segment (for shape comparison)
    native_z = (native - native.mean()) / (native.std() + 1e-6)
    learner_z = (learner - learner.mean()) / (learner.std() + 1e-6)

    # --- 1. Shape similarity via DTW (assuming you have _dtw_shape_score) ---
    shape_score = _dtw_shape_score(native_z, learner_z)

    # --- 2. Direction: rising/falling/flat ---
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

    # --- 3. Magnitude deviation (how far off in semitones) ---
    # at this point native and learner are same length
    mag_diff = float(np.mean(np.abs(native - learner)))

    if mode == "lenient":
        penalty_per_semi = 5.0
        max_penalty = 20.0
    elif mode == "strict":
        penalty_per_semi = 10.0
        max_penalty = 35.0
    else:  # normal
        penalty_per_semi = 7.5
        max_penalty = 25.0

    mag_penalty = min(mag_diff * penalty_per_semi, max_penalty)

    # --- 4. Direction correctness boost/penalty ---
    if native_dir == "level":
        dir_score = 100.0  # any stable-ish pitch is okay
    elif native_dir == learner_dir:
        dir_score = 100.0
    else:
        dir_penalty = 25.0 if mode == "strict" else 15.0
        dir_score = 100.0 - dir_penalty

    # --- 5. Combine into a final score ---
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

    # Direction for UI arrows
    if learner_dir == "up":
        pitch_direction = "up"
    elif learner_dir == "down":
        pitch_direction = "down"
    elif learner_dir == "level":
        pitch_direction = "ok"
    else:
        pitch_direction = "unknown"

    return base_score, pitch_direction



def load_alignment_for_sentence(sentence_id: str) -> List[Dict]:
    """
    Build alignment objects for scoring:

    Returns a list of {
      "text": <single character>,
      "native_pitch": [...],
      "learner_pitch": [...]
    }
    """
    # 1) Native aligned segments
    align_path = os.path.join(ALIGN_DIR, f"{sentence_id}_aligned_pitch.json")
    if not os.path.exists(align_path):
        raise FileNotFoundError(f"Aligned pitch file not found: {align_path}")

    with open(align_path, "r", encoding="utf-8") as f:
        native_segments = json.load(f)

    # 2) Sentence metadata for characters
    if not os.path.exists(SENTENCES_JSON):
        raise FileNotFoundError(f"sentences.json not found at {SENTENCES_JSON}")

    with open(SENTENCES_JSON, "r", encoding="utf-8") as f:
        sentences = json.load(f)

    sentence = next(
        (s for s in sentences if s["audio_filename"].split(".")[0] == sentence_id),
        None,
    )
    if sentence is None:
        raise FileNotFoundError(f"No metadata found for sentence_id={sentence_id}")

    trad = sentence.get("chinese_traditional") or ""
    char_list = [c for c in trad if not c.isspace()]

    # 3) Learner audio & pitch
    learner_path = os.path.join(LEARNER_AUDIO_DIR, f"{sentence_id}_learner.wav")
    if not os.path.exists(learner_path):
        raise FileNotFoundError(
            f"Learner audio not found for {sentence_id}: {learner_path}"
        )

    snd = parselmouth.Sound(learner_path)
    pitch_obj = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=500)
    times = pitch_obj.xs()
    freqs = pitch_obj.selected_array["frequency"]

    alignment_out: List[Dict] = []

    for idx, seg in enumerate(native_segments):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))

        native_curve = np.array(seg.get("pitch_segment", []), dtype=float)
        native_curve[native_curve <= 0] = np.nan

        mask = (times >= start) & (times <= end)
        if not np.any(mask):
            learner_curve = np.array([], dtype=float)
        else:
            lc = np.array(freqs[mask], dtype=float)
            lc[lc <= 0] = np.nan
            learner_curve = lc

        text = char_list[idx] if idx < len(char_list) else ""

        alignment_out.append(
            {
                "text": text,
                "native_pitch": native_curve.tolist(),
                "learner_pitch": learner_curve.tolist(),
            }
        )

    return alignment_out


def evaluate_pronunciation(sentence_id, mode="normal"):
    """
    Top-level evaluation function used by the Flask app.
    """
    alignment = load_alignment_for_sentence(sentence_id)

    segments_out = []
    segment_scores = []

    for seg in alignment:
        native_curve = seg.get("native_pitch", [])
        learner_curve = seg.get("learner_pitch", [])

        score, pitch_direction = _segment_score(native_curve, learner_curve, mode=mode)
        color = _color_for_score(score, mode=mode)

        segments_out.append(
            {
                "text": seg.get("text", ""),
                "score": score,
                "pitch_direction": pitch_direction,
                "color": color,
            }
        )
        segment_scores.append(score)

    overall = float(np.mean(segment_scores)) if segment_scores else 0.0

    return {
        "mode": mode,
        "overall_score": overall,
        "segments": segments_out,
    }
