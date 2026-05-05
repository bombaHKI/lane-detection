import numpy as np

def kapur_threshold(intensities: np.ndarray) -> float | None:
    """Return the max-entropy intensity threshold (Kapur's method) or None."""
    if intensities.size == 0:
        return None
    val, counts = np.unique(intensities, return_counts=True)
    if val.size < 2:
        return None

    probs = counts.astype(np.float64) / counts.sum()
    P_cum = np.cumsum(probs)
    log_probs = np.log(probs, out=np.zeros_like(probs), where=probs != 0)
    H_cum = np.cumsum(probs * -log_probs)
    H_total = H_cum[-1]

    eps = 1e-10
    P_omega     = P_cum
    P_omega_bar = 1.0 - P_cum

    term1 = np.log(P_omega + eps)     + (H_cum / (P_omega + eps))
    term2 = np.log(P_omega_bar + eps) + ((H_total - H_cum) / (P_omega_bar + eps))
    Phi = term1 + term2
    Phi[0]  = -np.inf
    Phi[-1] = -np.inf

    return float(val[int(np.argmax(Phi))])