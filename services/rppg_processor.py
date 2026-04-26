"""
rPPG Processor — Remote Photoplethysmography
─────────────────────────────────────────────────────────────────────────────
Measures heart rate from camera frames by tracking subtle skin color changes.

Algorithm (CHROM method, De Haan & Jeanne 2013):
  1. Face detection using OpenCV Haar cascade
  2. Extract mean R, G, B values from face ROI each frame
  3. Collect 10-second sliding window (300 frames at 30fps, or variable)
  4. Apply CHROM algorithm: orthogonal chrominance signals Xs, Ys
  5. Bandpass filter 0.67–3.0 Hz (40–180 BPM)
  6. FFT → dominant frequency → BPM

Why CHROM over plain green-channel:
  - Cancels illuminant-dependent noise (RGB fluctuations from lights)
  - More robust to skin tone differences
  - Less sensitive to subtle camera shake

References:
  - De Haan & Jeanne, IEEE Trans Bio-Med Eng, 2013
  - Wang et al., IEEE Trans Bio-Med Eng, 2017 (POS method)
  - rPPG-Toolbox (open source benchmark)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from scipy.signal import butter, filtfilt

# ─── Bandpass Filter ────────────────────────────────────────────────────────


def _bandpass(signal: np.ndarray, fs: float, low: float = 0.67, high: float = 3.0) -> np.ndarray:
    """Butterworth bandpass filter. fs = sample rate in Hz."""
    nyq = 0.5 * fs
    low_n = low / nyq
    high_n = high / nyq
    # Clamp to valid range
    low_n = max(0.01, min(low_n, 0.99))
    high_n = max(0.01, min(high_n, 0.99))
    if low_n >= high_n:
        return signal
    b, a = butter(4, [low_n, high_n], btype="band")
    padlen = min(len(signal) - 1, 3 * max(len(a), len(b)))
    if len(signal) < 15:  # too short for meaningful filtering
        return signal
    return filtfilt(b, a, signal, padlen=padlen)


# ─── CHROM rPPG ─────────────────────────────────────────────────────────────


def _chrom_bvp(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    CHROM method: compute blood volume pulse signal.
    Returns 1D BVP signal array, same length as input.
    """
    r = r.astype(float)
    g = g.astype(float)
    b = b.astype(float)

    # Normalize by mean (remove DC)
    r_n = r / (np.mean(r) + 1e-9)
    g_n = g / (np.mean(g) + 1e-9)
    b_n = b / (np.mean(b) + 1e-9)

    # CHROM chrominance signals
    xs = 3 * r_n - 2 * g_n  # Xs
    ys = 1.5 * r_n + g_n - 1.5 * b_n  # Ys

    # Standardize
    std_xs = np.std(xs) + 1e-9
    std_ys = np.std(ys) + 1e-9
    alpha = std_xs / std_ys

    bvp = xs - alpha * ys
    return bvp


# ─── RPPGProcessor ──────────────────────────────────────────────────────────


class RPPGProcessor:
    """
    Per-session stateful rPPG processor.

    Usage:
        proc = RPPGProcessor()
        proc.add_frame(base64_jpeg_string, timestamp=time.time())
        result = proc.compute()  # call any time
    """

    WINDOW_SEC = 8.0  # sliding window length (seconds)
    MIN_FRAMES = 12  # lowered — phone captures at ~5fps, so 12 frames = ~2.5s warmup
    TARGET_FPS = 30.0
    BPM_LOW = 40
    BPM_HIGH = 180

    def __init__(self):
        self._r: deque = deque()
        self._g: deque = deque()
        self._b: deque = deque()
        self._ts: deque = deque()

        self.last_bpm: float = 0.0
        self.last_hrv: float = 0.0
        self.last_quality: str = "waiting"
        self.last_waveform: list = []
        self.frames_total: int = 0

    # ── Public API ─────────────────────────────────────────────────────────

    def add_rgb(self, r_val: float, g_val: float, b_val: float, timestamp: float) -> dict[str, Any]:
        """
        Directly receive R, G, B stream from edge device. No face detection needed server-side.
        """
        self.frames_total += 1

        self._r.append(r_val)
        self._g.append(g_val)
        self._b.append(b_val)
        self._ts.append(timestamp)

        self._trim_window()

        # PF-09: Signal validation — reject invalid RGB
        if (r_val < 10 and g_val < 10 and b_val < 10) or (r_val > 245 and g_val > 245 and b_val > 245):
            return {"face_found": True, "signal_quality": "invalid", "message": "Adjust lighting", "bpm": 0}

        quality = self._signal_quality()
        return {"face_found": True, "signal_quality": quality, "r": r_val, "g": g_val, "b": b_val}

    def compute(self) -> dict[str, Any]:
        """
        Compute BPM from current window.
        Returns: { bpm, hrv_ms, signal_quality, waveform, fps, frames_in_window, status }
        """
        n = len(self._r)
        quality = self._signal_quality()

        if n < self.MIN_FRAMES:
            return {
                "bpm": self.last_bpm,
                "hrv_ms": self.last_hrv,
                "signal_quality": quality,
                "waveform": self.last_waveform,
                "frames_in_window": n,
                "status": "warmup",
                "message": f"Collecting signal… {n}/{self.MIN_FRAMES} frames",
            }

        # Estimate effective sample rate
        ts_arr = np.array(self._ts)
        elapsed = ts_arr[-1] - ts_arr[0]
        fs = (n - 1) / (elapsed + 1e-9)
        fs = float(np.clip(fs, 1.5, 60.0))

        r = np.array(self._r)
        g = np.array(self._g)
        b = np.array(self._b)

        # 1. Resample to uniform time grid to eliminate network jitter noise
        from scipy.interpolate import interp1d
        from scipy.signal import detrend

        uniform_ts = np.linspace(ts_arr[0], ts_arr[-1], n)
        r = interp1d(ts_arr, r, kind="linear")(uniform_ts)
        g = interp1d(ts_arr, g, kind="linear")(uniform_ts)
        b = interp1d(ts_arr, b, kind="linear")(uniform_ts)

        # CHROM BVP
        bvp = _chrom_bvp(r, g, b)

        # 2. Detrend to remove baseline wander (low-frequency drift)
        bvp = detrend(bvp)

        # Bandpass
        bvp_f = _bandpass(bvp, fs, 0.67, 3.0)

        # FFT with Hann window to reduce spectral leakage
        n_fft = len(bvp_f)
        window = np.hanning(n_fft)
        bvp_windowed = bvp_f * window
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
        power = np.abs(np.fft.rfft(bvp_windowed)) ** 2

        # Restrict to BPM range
        mask = (freqs >= self.BPM_LOW / 60.0) & (freqs <= self.BPM_HIGH / 60.0)
        if mask.sum() == 0:
            bpm_raw = self.last_bpm or 75.0
        else:
            peak_idx = np.argmax(power[mask])
            bpm_raw = float(freqs[mask][peak_idx] * 60.0)

        # PF-09: Variance check on last 20 RGB samples — flat signal = no skin/no pulse
        if n >= 20:
            r_var = float(np.var(np.array(self._r)[-20:]))
            g_var = float(np.var(np.array(self._g)[-20:]))
            b_var = float(np.var(np.array(self._b)[-20:]))
            if r_var < 0.5 and g_var < 0.5 and b_var < 0.5:
                return {
                    "bpm": self.last_bpm,
                    "hrv_ms": self.last_hrv,
                    "signal_quality": "no_pulse",
                    "message": "Ensure face is visible — no signal variance detected",
                    "waveform": self.last_waveform,
                    "frames_in_window": n,
                    "status": "invalid",
                }

        # PF-09: Artifact detection — reject wild jumps
        artifact_flag = False
        if self.last_bpm > 0 and abs(bpm_raw - self.last_bpm) > 25:
            bpm_raw = self.last_bpm  # Hold previous — don't trust wild jumps
            artifact_flag = True

        # Clamp to physiological range
        bpm_raw = max(self.BPM_LOW, min(self.BPM_HIGH, bpm_raw))

        # Heavy temporal smoothing — rPPG from phone camera is noisy
        # 0.85/0.15 means it takes ~5 readings to converge to a new value
        if self.last_bpm > 0:
            if abs(bpm_raw - self.last_bpm) > 12:
                bpm = self.last_bpm * 0.85 + bpm_raw * 0.15
            else:
                bpm = self.last_bpm * 0.70 + bpm_raw * 0.30
        else:
            bpm = bpm_raw

        # SNR-based quality (tuned for WebSocket RGB stream, not raw camera)
        # CHROM on a clean cardiac pulse typically gives SNR in 0.08-0.30 range
        peak_power = float(power[mask].max()) if mask.sum() > 0 else 0
        total_power = float(power.sum()) + 1e-9
        snr = peak_power / total_power

        if snr > 0.18:
            quality_str = "excellent"
        elif snr > 0.08:
            quality_str = "good"
        elif snr > 0.03:
            quality_str = "fair"
        else:
            quality_str = "poor"

        # HRV approximation: std of inter-beat intervals estimated from BVP peaks
        hrv_ms_raw = self._estimate_hrv(bvp_f, fs)

        # Smooth HRV
        if self.last_hrv > 0 and hrv_ms_raw > 0:
            hrv_ms = self.last_hrv * 0.65 + hrv_ms_raw * 0.35
        else:
            hrv_ms = hrv_ms_raw

        # Waveform: downsample to 50 points for the app
        wf_ds = bvp_f[:: max(1, len(bvp_f) // 50)].tolist()
        # Normalize to -1..1
        wf_max = max(abs(v) for v in wf_ds) or 1.0
        wf_norm = [round(v / wf_max, 3) for v in wf_ds]

        self.last_bpm = round(bpm, 1)
        self.last_hrv = hrv_ms
        self.last_quality = quality_str
        self.last_waveform = wf_norm

        return {
            "bpm": self.last_bpm,
            "hrv_ms": hrv_ms,
            "signal_quality": "artifact" if artifact_flag else quality_str,
            "waveform": wf_norm,
            "fps": round(fs, 1),
            "frames_in_window": n,
            "snr": round(snr, 4),
            "status": "ok",
            "artifact_detected": artifact_flag,
        }

    # ── Private helpers ────────────────────────────────────────────────────

    def _trim_window(self) -> None:
        """Keep only WINDOW_SEC of data, but guarantee we keep at least MIN_FRAMES."""
        while len(self._ts) > self.MIN_FRAMES:
            if self._ts[-1] - self._ts[0] <= self.WINDOW_SEC:
                break
            self._r.popleft()
            self._g.popleft()
            self._b.popleft()
            self._ts.popleft()

    def _signal_quality(self) -> str:
        n = len(self._r)
        if n < 10:
            return "waiting"
        if n < self.MIN_FRAMES:
            return "warmup"
        return "measuring"

    def _estimate_hrv(self, bvp: np.ndarray, fs: float) -> float:
        """
        Estimate RMSSD (root mean square of successive differences) from BVP peaks.
        Returns HRV in ms.
        """
        try:
            from scipy.signal import find_peaks

            # Find systolic peaks. Lower prominence for webcam BVP signals.
            min_dist = int(fs * 60.0 / 180)  # max 180 BPM
            peaks, _ = find_peaks(bvp, distance=min_dist, prominence=0.01)
            if len(peaks) < 3:
                return 0.0
            ibi_s = np.diff(peaks) / fs  # inter-beat intervals in seconds
            ibi_ms = ibi_s * 1000.0
            rmssd = float(np.sqrt(np.mean(np.diff(ibi_ms) ** 2)))
            return round(min(rmssd, 200.0), 1)
        except Exception:
            return 0.0
