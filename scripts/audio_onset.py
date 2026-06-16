import parselmouth
import numpy as np
from typing import Tuple


MAX_TRIM_S = 1.0  # never trim more than this many seconds of leading silence


def detect_onset_time(
    wav_path: str,
    intensity_floor_db: float = -50.0,
    rel_threshold_db: float = 35.0,
    min_speech_ms: float = 40.0,
) -> float:
    """
    Estimate when speech actually starts in an audio file (in seconds).

    Strategy:
    - Convert to Intensity (dB) via Praat/parselmouth.
    - Compute a dynamic threshold: max_intensity - rel_threshold_db,
      but never below intensity_floor_db.
    - Find the first contiguous run of frames above this threshold that
      lasts at least min_speech_ms.
    If no speech is found, returns 0.0.
    """
    snd = parselmouth.Sound(wav_path)
    intensity = snd.to_intensity()

    times = intensity.xs()
    values = intensity.values[0]

    if len(values) == 0:
        return 0.0

    max_db = float(np.max(values))
    thr = max(max_db - rel_threshold_db, intensity_floor_db)

    min_speech_s = min_speech_ms / 1000.0
    above = values > thr

    run_start_idx = None
    for i, is_above in enumerate(above):
        if is_above and run_start_idx is None:
            run_start_idx = i
        elif not is_above and run_start_idx is not None:
            duration = times[i - 1] - times[run_start_idx]
            if duration >= min_speech_s:
                return min(float(times[run_start_idx]), MAX_TRIM_S)
            run_start_idx = None

    # Handle possible run to the end
    if run_start_idx is not None:
        duration = times[-1] - times[run_start_idx]
        if duration >= min_speech_s:
            return min(float(times[run_start_idx]), MAX_TRIM_S)

    return 0.0


def trim_audio_to_onset(
    wav_path: str,
    output_path: str,
    onset_time: float,
) -> None:
    """
    Write a new WAV that starts at onset_time seconds.

    If onset_time <= 0, just copy the audio.
    """
    snd = parselmouth.Sound(wav_path)

    if onset_time <= 0.0:
        snd.save(output_path, "WAV")
        return

    trimmed = snd.extract_part(
        from_time=onset_time,
        to_time=None,
        preserve_times=False  # new file's 0.0 = onset_time
    )
    trimmed.save(output_path, "WAV")


def detect_and_trim(
    wav_path: str,
    output_path: str,
    **kwargs,
) -> Tuple[float, str]:
    """
    Detect onset and write a trimmed WAV starting at speech.

    Returns: (onset_time_in_original_file_sec, output_path)
    """
    onset = detect_onset_time(wav_path, **kwargs)
    trim_audio_to_onset(wav_path, output_path, onset)
    return onset, output_path
