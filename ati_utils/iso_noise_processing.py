import numpy as np

def add_d455_noise(
    clean_uint8: np.array, 
    iso: float, 
    read_noise_e: float = 2.0, 
    photon_scale: float = 40.0
):
    """
    Physically motivated noise model for OV9782-like sensor.

    The renderer produces:  pixel_value = photons * iso_gain  (clipped to [0, 255])

    We undo the gain, apply noise in the photon domain, then re-apply gain.
    This naturally produces:
      - More noise at high ISO (read noise amplified by gain)
      - Less noise with more light or longer shutter (more photons)
      - Reduced dynamic range at high ISO (highlights clip + noise floor rises)

    Parameters
    ----------
    clean_uint8  : ndarray [H,W,3] uint8, clean linear render
    iso          : float, ISO setting (100 = base)
    read_noise_e : float, read noise std in electrons (OV9782: 1.5-3e estimated)
    photon_scale : float, photons per DN at ISO 100 (calibration knob)

    Returns
    -------
    noisy_uint8  : ndarray [H,W,3] uint8
    """
    iso_gain = iso / 100.0
    signal = clean_uint8.astype(np.float64)

    # 1) Undo ISO gain to recover photon-proportional signal (must use linear rendering)
    signal_base = signal / iso_gain

    # 2) Convert to photon counts
    photons = signal_base * photon_scale

    # 3) Photon shot noise (Poisson)
    photons_noisy = np.random.poisson(
        np.clip(photons, 0, None).astype(np.float64)
    ).astype(np.float64)

    # 4) Back to DN and re-apply ISO gain
    signal_noisy = (photons_noisy / photon_scale) * iso_gain

    # 5) Read noise: constant in electrons, amplified by ISO gain
    read_noise_dn = read_noise_e / photon_scale
    signal_noisy += np.random.normal(0, read_noise_dn * iso_gain, signal.shape)

    return np.clip(signal_noisy, 0, 255).astype(np.uint8)