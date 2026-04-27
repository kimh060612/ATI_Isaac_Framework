import cv2
import numpy as np


class HistogramAutoExposure:
    def __init__(
        self,
        target=0.45,
        low_percentile=5.0,
        high_percentile=95.0,
        min_exposure=1e-4,
        max_exposure=0.032,
        smoothing=0.2,
        max_ev_step=0.5,
        eps=1e-6,
    ):
        """
        target:
            목표 luminance. 0~1 normalized scale.
            일반적으로 0.4~0.5 정도가 무난함.

        low_percentile, high_percentile:
            histogram clipping 범위.
            예: 5~95 percentile만 사용.

        min_exposure, max_exposure:
            exposure time 제한. 단위는 seconds.

        smoothing:
            0~1. 클수록 빠르게 반응.
            0.1~0.3 정도 추천.

        max_ev_step:
            한 프레임에서 바뀔 수 있는 최대 exposure 변화량.
            EV 기준. 0.5면 한 번에 최대 약 sqrt(2)배 변화.
        """
        self.target = target
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile
        self.min_exposure = min_exposure
        self.max_exposure = max_exposure
        self.smoothing = smoothing
        self.max_ev_step = max_ev_step
        self.eps = eps

    def rgb_to_luminance(self, image):
        """
        image: uint8 RGB or BGR image, shape (H, W, 3)
        return: luminance in [0, 1]
        """
        image = image.astype(np.float32) / 255.0

        # OpenCV를 쓰는 경우 image가 BGR일 가능성이 크므로 BGR 기준.
        b = image[..., 0]
        g = image[..., 1]
        r = image[..., 2]

        # Rec.709 luminance
        y = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return y

    def measure_luminance(self, image, mask=None):
        y = self.rgb_to_luminance(image)

        if mask is not None:
            y = y[mask > 0]
        else:
            y = y.reshape(-1)

        if y.size == 0:
            return self.target

        low = np.percentile(y, self.low_percentile)
        high = np.percentile(y, self.high_percentile)

        clipped = y[(y >= low) & (y <= high)]

        if clipped.size == 0:
            clipped = y

        # log-mean은 bright outlier에 덜 민감해서 AE에 자주 잘 맞음.
        measured = np.exp(np.mean(np.log(clipped + self.eps)))

        return float(measured)

    def update_exposure(self, image, current_exposure, mask=None):
        measured = self.measure_luminance(image, mask=mask)

        # 목표 밝기 / 현재 밝기
        raw_ratio = self.target / max(measured, self.eps)

        # ratio를 EV step으로 제한
        # ratio = 2^EV
        ev = np.log2(raw_ratio)
        ev = np.clip(ev, -self.max_ev_step, self.max_ev_step)
        limited_ratio = 2.0 ** ev

        desired_exposure = current_exposure * limited_ratio

        # log-domain smoothing
        log_current = np.log(max(current_exposure, self.eps))
        log_desired = np.log(max(desired_exposure, self.eps))

        log_next = (1.0 - self.smoothing) * log_current + self.smoothing * log_desired
        next_exposure = np.exp(log_next)

        next_exposure = np.clip(
            next_exposure,
            self.min_exposure,
            self.max_exposure,
        )

        debug = {
            "measured_luminance": measured,
            "raw_ratio": raw_ratio,
            "limited_ratio": limited_ratio,
            "ev_step": ev,
            "next_exposure": float(next_exposure),
        }

        return float(next_exposure), debug


class HistogramAEExposureGain:
    def __init__(
        self,
        target=0.45,
        low_percentile=5.0,
        high_percentile=95.0,
        min_exposure=0.001,
        max_exposure=0.016,
        min_gain=1.0,
        max_gain=8.0,
        smoothing=0.25,
        max_ev_step=0.5,
        eps=1e-6,
    ):
        self.hist_ae = HistogramAutoExposure(
            target=target,
            low_percentile=low_percentile,
            high_percentile=high_percentile,
            min_exposure=min_exposure,
            max_exposure=max_exposure,
            smoothing=smoothing,
            max_ev_step=max_ev_step,
            eps=eps,
        )

        self.target = target
        self.min_exposure = min_exposure
        self.max_exposure = max_exposure
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.smoothing = smoothing
        self.max_ev_step = max_ev_step
        self.eps = eps

    def update(self, image, current_exposure, current_gain, mask=None):
        measured = self.hist_ae.measure_luminance(image, mask=mask)

        current_total = current_exposure * current_gain
        raw_ratio = self.target / max(measured, self.eps)

        ev = np.log2(raw_ratio)
        ev = np.clip(ev, -self.max_ev_step, self.max_ev_step)
        limited_ratio = 2.0 ** ev

        desired_total = current_total * limited_ratio

        # log-domain smoothing
        log_current = np.log(max(current_total, self.eps))
        log_desired = np.log(max(desired_total, self.eps))
        log_next = (1 - self.smoothing) * log_current + self.smoothing * log_desired
        next_total = np.exp(log_next)

        # exposure 우선, 부족하면 gain 사용
        next_exposure = np.clip(next_total / self.min_gain,
                                self.min_exposure,
                                self.max_exposure)

        next_gain = next_total / next_exposure
        next_gain = np.clip(next_gain, self.min_gain, self.max_gain)

        debug = {
            "measured_luminance": measured,
            "raw_ratio": raw_ratio,
            "limited_ratio": limited_ratio,
            "ev_step": ev,
            "next_total_exposure_gain": float(next_total),
            "next_exposure": float(next_exposure),
            "next_gain": float(next_gain),
        }

        return float(next_exposure), float(next_gain), debug
    
class HighlightProtectedHistogramAE(HistogramAEExposureGain):
    def __init__(
        self,
        *args,
        saturation_threshold=0.98,
        max_saturation_ratio=0.01,
        highlight_penalty_strength=0.7,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.saturation_threshold = saturation_threshold
        self.max_saturation_ratio = max_saturation_ratio
        self.highlight_penalty_strength = highlight_penalty_strength

    def update(self, image, current_exposure, current_gain, mask=None):
        y = self.hist_ae.rgb_to_luminance(image)

        if mask is not None:
            yy = y[mask > 0]
        else:
            yy = y.reshape(-1)

        saturation_ratio = np.mean(yy >= self.saturation_threshold)

        measured = self.hist_ae.measure_luminance(image, mask=mask)

        raw_ratio = self.target / max(measured, self.eps)

        # saturation이 많으면 exposure를 강제로 낮춤
        if saturation_ratio > self.max_saturation_ratio:
            excess = saturation_ratio / self.max_saturation_ratio
            penalty_ev = self.highlight_penalty_strength * np.log2(excess)
            raw_ratio *= 2.0 ** (-penalty_ev)

        ev = np.log2(raw_ratio)
        ev = np.clip(ev, -self.max_ev_step, self.max_ev_step)
        limited_ratio = 2.0 ** ev

        current_total = current_exposure * current_gain
        desired_total = current_total * limited_ratio

        log_current = np.log(max(current_total, self.eps))
        log_desired = np.log(max(desired_total, self.eps))
        log_next = (1 - self.smoothing) * log_current + self.smoothing * log_desired
        next_total = np.exp(log_next)

        next_exposure = np.clip(
            next_total / self.min_gain,
            self.min_exposure,
            self.max_exposure,
        )

        next_gain = np.clip(
            next_total / next_exposure,
            self.min_gain,
            self.max_gain,
        )

        debug = {
            "measured_luminance": float(measured),
            "saturation_ratio": float(saturation_ratio),
            "raw_ratio_after_highlight_penalty": float(raw_ratio),
            "ev_step": float(ev),
            "next_exposure": float(next_exposure),
            "next_gain": float(next_gain),
        }

        return float(next_exposure), float(next_gain), debug