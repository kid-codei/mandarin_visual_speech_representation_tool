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

    Behavior:
      - If native/learner have different lengths, truncate both to the
        shorter length so NumPy ops are safe.
      - Treat <= 0 Hz as unvoiced (NaN) before scoring.
      - If 0 overlapping voiced frames -> 0, "unknown".
      - If 1–2 overlapping voiced frames -> simple magnitude-based fallback.
      - If >=3 overlapping voiced frames -> full shape/direction scoring.
    """

    # Basic sanity
    if not native_curve or not learner_curve:
        return 0.0, "unknown"

    native = np.asarray(native_curve, dtype=float)
    learner = np.asarray(learner_curve, dtype=float)

    # --- Make lengths match to avoid broadcasting errors ---
    min_len = min(len(native), len(learner))
    if min_len == 0:
        return 0.0, "unknown"

    if len(native) != len(learner):
        native = native[:min_len]
        learner = learner[:min_len]

    # --- Treat non-positive / NaN as unvoiced ---
    native[(native <= 0) | np.isnan(native)] = np.nan
    learner[(learner <= 0) | np.isnan(learner)] = np.nan

    # Overlapping voiced frames
    mask = ~np.isnan(native) & ~np.isnan(learner)
    n_voiced = int(mask.sum())

    # ========== Case 1: 0 voiced frames ==========
    if n_voiced == 0:
        return 0.0, "unknown"

    # ========== Case 2: 1–2 voiced frames (fallback) ==========
    if n_voiced < 3:
        n_vals = native[mask]
        l_vals = learner[mask]

        # Mean absolute difference in Hz
        diff_hz = float(np.mean(np.abs(n_vals - l_vals)))

        # Map diff_hz to [0, 100]:
        #   0 Hz diff -> 100
        #   >= 80 Hz diff -> 0   (tweakable)
        max_diff = 80.0
        raw = max(0.0, 1.0 - diff_hz / max_diff)
        score = raw * 100.0

        # Direction hint based on who is higher on average
        n_mean = float(np.mean(n_vals))
        l_mean = float(np.mean(l_vals))
        if n_mean == 0 or l_mean == 0:
            direction = "unknown"
        else:
            d = l_mean - n_mean
            if d > 20:
                direction = "down"  # learner too high overall
            elif d < -20:
                direction = "up"    # learner too low overall
            else:
                direction = "ok"

        return score, direction

    # ========== Case 3: >=3 voiced frames (full logic) ==========
    native = native[mask]
    learner = learner[mask]

    # Normalize (z-score) within segment so we compare shape
    native_z = (native - native.mean()) / (native.std() + 1e-6)
    learner_z = (learner - learner.mean()) / (learner.std() + 1e-6)

    # 1. Direction: rising/falling/flat
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

    # 2. Shape similarity (cosine)
    cos_sim = float(
        np.dot(native_z, learner_z) /
        (np.linalg.norm(native_z) * np.linalg.norm(learner_z) + 1e-6)
    )
    # map [-1,1] -> [0,100]
    shape_score = 50 * (cos_sim + 1)

    # 3. Magnitude deviation (how far off in Hz)
    mag_diff = float(np.mean(np.abs(native - learner)))

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

    # 4. Direction correctness boost/penalty
    if native_dir == "level":
        dir_score = 100.0
    elif native_dir == learner_dir:
        dir_score = 100.0
    else:
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

def _compute_full_native_series(alignment):
    """
    Concatenate native pitch segments across the whole sentence
    into a single 1D numpy array (Hz).

    Tries `native_pitch` first, then `pitch_segment`.
    """
    all_vals = []
    for seg in alignment:
        curve = seg.get("native_pitch")
        if not curve:
            curve = seg.get("pitch_segment")
        if not curve:
            continue
        all_vals.extend([float(x) for x in curve])
    return np.asarray(all_vals, dtype=float)

def _extract_full_learner_series(sentence_id, time_step=0.01, floor=75, ceiling=500):
    """
    Extract full-sentence learner pitch from static/learner_audio/<id>_learner.wav.

    Returns a 1D numpy array of Hz with np.nan for unvoiced.
    """
    wav_path = os.path.join(LEARNER_AUDIO_DIR, f"{sentence_id}_learner.wav")
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"Learner audio not found for {sentence_id}: {wav_path}")

    snd = parselmouth.Sound(wav_path)
    pitch_obj = snd.to_pitch(time_step=time_step, pitch_floor=floor, pitch_ceiling=ceiling)
    freqs = pitch_obj.selected_array["frequency"]
    series = np.array([f if f > 0 else np.nan for f in freqs], dtype=float)
    return series

def _prepare_for_dtw(series: np.ndarray):
    """
    Convert Hz array to a z-normalized contour for DTW.

    - Ignores np.nan and non-positive values as unvoiced.
    - Returns an array of same length, with unvoiced set to 0.
    """
    series = np.asarray(series, dtype=float)
    mask = ~np.isnan(series) & (series > 0)
    if not np.any(mask):
        return None

    voiced = series[mask]
    mean = voiced.mean()
    std = voiced.std()
    if std < 1e-6:
        std = 1e-6

    z = np.zeros_like(series)
    z[mask] = (series[mask] - mean) / std
    # unvoiced stays at 0
    return z

def _dtw_warp_full_sentence(native_series: np.ndarray, learner_series: np.ndarray) -> np.ndarray:
    """
    Use DTW to warp the learner pitch contour onto the native timeline.

    Both inputs are 1D Hz arrays. Returns a 1D array of length len(native_series),
    where each native frame is assigned an averaged learner Hz value along the DTW path.
    """
    native_series = np.asarray(native_series, dtype=float)
    learner_series = np.asarray(learner_series, dtype=float)

    if native_series.size == 0 or learner_series.size == 0:
        return np.full_like(native_series, np.nan)

    native_z = _prepare_for_dtw(native_series)
    learner_z = _prepare_for_dtw(learner_series)
    if native_z is None or learner_z is None:
        return np.full_like(native_series, np.nan)

    # librosa expects shape = (features, time)
    X = native_z.reshape(1, -1)
    Y = learner_z.reshape(1, -1)

    _, wp = librosa.sequence.dtw(X=X, Y=Y, metric="euclidean")
    wp = wp[::-1]  # path from start to end

    warped = np.full(native_series.shape, np.nan, dtype=float)
    counts = np.zeros(native_series.shape, dtype=int)

    for i, j in wp:
        if 0 <= i < len(native_series) and 0 <= j < len(learner_series):
            if np.isnan(warped[i]):
                warped[i] = learner_series[j]
            else:
                warped[i] += learner_series[j]
            counts[i] += 1

    mask = counts > 0
    warped[mask] = warped[mask] / counts[mask]

    return warped


def evaluate_pronunciation(sentence_id, mode="normal"):
    """
    Top-level evaluation function used by the Flask app.
    """
    alignment = load_alignment_for_sentence(sentence_id)

    segments_out = []
    segment_scores = []

    for seg in alignment:
        native_curve = seg.get("native_pitch") or seg.get("pitch_segment") or []
        learner_curve = seg.get("learner_pitch", [])

        score, pitch_direction = _segment_score(native_curve, learner_curve, mode=mode)
        color = _color_for_score(score, mode=mode)

        segments_out.append({
            "text": seg.get("text", ""),
            "score": score,
            "pitch_direction": pitch_direction,
            "color": color,
        })
        segment_scores.append(score)

    overall = float(np.mean(segment_scores)) if segment_scores else 0.0

    # --- Full-sentence DTW alignment for visualization ---
    try:
        native_full = _compute_full_native_series(alignment)
        learner_full = _extract_full_learner_series(sentence_id)
        dtw_aligned = _dtw_warp_full_sentence(native_full, learner_full)

        # Convert NaNs to 0.0 for JSON; JS treats 0 as unvoiced.
        dtw_aligned_list = [
            float(x) if not np.isnan(x) else 0.0
            for x in dtw_aligned
        ]
    except Exception as e:
        # Fail soft: still return feedback, just no DTW overlay
        print(f"[DTW] Failed for {sentence_id}: {e}")
        dtw_aligned_list = []

    return {
        "mode": mode,
        "overall_score": overall,
        "segments": segments_out,
        "dtw_aligned_learner_pitch": dtw_aligned_list,
    }

