"""Per-frame picture metrics for one camera view (computed on the analysed RGB view).

luma_mean          mean BT.709 luma, 0-255 (full range)
luma_p02 / p98     2nd / 98th luma percentile (shadow / highlight clipping)
sharpness          variance of the Laplacian of the luma at analysis resolution (higher = sharper)
lab_a / lab_b      mean CIELAB a* (green-red) and b* (blue-yellow) of the whole frame; 0 = neutral.
                   This is the scene colour: a blue work mat moves it without any colour cast.
lab_a_sd/lab_b_sd  their spread over the frame
neutral_a/_b       mean a*, b* of the near-neutral pixels only (chroma < NEUTRAL_CHROMA, mid
                   lightness): grey/white surfaces, which turn the colour of a cast. This is the
                   colour-cast measurement; empty when the frame has too few neutral pixels.
neutral_share      share of the frame that is near-neutral
Exposure and colour statistics are taken on a half-resolution copy (identical means,
4x faster); sharpness uses the full analysed view.
"""

from __future__ import annotations

import cv2
import numpy as np

_BT709 = np.array([[0.0722, 0.7152, 0.2126]], dtype=np.float32)   # weights for B, G, R order
NEUTRAL_CHROMA = 15.0          # a*b* chroma below this = near-neutral (grey/white) pixel
NEUTRAL_L8 = (40, 240)         # 8-bit L (0-255) range: exclude near-black and clipped pixels
MIN_NEUTRAL_SHARE = 0.02       # fewer neutral pixels than this: no colour-cast measurement for the frame


def _percentile_u8(hist: np.ndarray, q: float) -> float:
    cdf = np.cumsum(hist)
    return float(np.searchsorted(cdf, q / 100.0 * cdf[-1]))


def frame_quality(rgb: np.ndarray) -> dict:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    luma = cv2.transform(bgr, _BT709)                                 # uint8, saturating, rounded
    _, lap_sd = cv2.meanStdDev(cv2.Laplacian(luma, cv2.CV_16S))       # exact for 8-bit input (|lap| <= 2040)
    sharp = float(lap_sd[0, 0]) ** 2
    small = cv2.resize(bgr, (rgb.shape[1] // 2, rgb.shape[0] // 2), interpolation=cv2.INTER_AREA)
    small_luma = cv2.transform(small, _BT709)
    hist = np.bincount(small_luma.ravel(), minlength=256)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)                      # 8-bit: a*, b* offset by 128
    (_, a_mean, b_mean), (_, a_sd, b_sd) = (x.ravel() for x in cv2.meanStdDev(lab))
    a = lab[..., 1].astype(np.float32) - 128.0
    b = lab[..., 2].astype(np.float32) - 128.0
    lightness = lab[..., 0]
    neutral = (a * a + b * b < NEUTRAL_CHROMA ** 2) & (lightness >= NEUTRAL_L8[0]) & (lightness <= NEUTRAL_L8[1])
    share = float(neutral.mean())
    if share >= MIN_NEUTRAL_SHARE:
        neutral_a, neutral_b = round(float(a[neutral].mean()), 2), round(float(b[neutral].mean()), 2)
    else:
        neutral_a = neutral_b = ""
    return {
        "luma_mean": round(float(small_luma.mean()), 2),
        "luma_p02": _percentile_u8(hist, 2),
        "luma_p98": _percentile_u8(hist, 98),
        "sharpness": round(sharp, 1),
        "lab_a": round(float(a_mean) - 128.0, 2),
        "lab_b": round(float(b_mean) - 128.0, 2),
        "lab_a_sd": round(float(a_sd), 2),
        "lab_b_sd": round(float(b_sd), 2),
        "neutral_a": neutral_a,
        "neutral_b": neutral_b,
        "neutral_share": round(share, 4),
    }
