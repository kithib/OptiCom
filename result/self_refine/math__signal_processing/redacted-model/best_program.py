# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series

This algorithm implements a sliding window approach to filter volatile, non-stationary
time series data while minimizing noise and preserving signal dynamics.
"""
import numpy as np


def adaptive_filter(x, window_size=20):
    """
    Adaptive signal processing algorithm using sliding window approach.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window (W samples)

    Returns:
        y: Filtered output signal with length = len(x) - window_size + 1
    """
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    output_length = len(x) - window_size + 1
    y = np.zeros(output_length)

    for i in range(output_length):
        window = x[i : i + window_size]
        y[i] = np.mean(window)

    return y


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Enhanced version with trend preservation using robust locally weighted
    regression with adaptive window centering, edge correction, trend consistency
    filtering, and optimized phase alignment to minimize spurious reversals,
    reduce lag error, and improve tracking accuracy with false reversal penalty.
    Now with adaptive outlier scale, improved phase compensation, refined
    trend consistency for better responsiveness, explicit consecutive 
    consistency check for false reversal suppression, and enhanced blending
    of robust regression with adaptive polynomial estimation for improved
    tracking accuracy and noise reduction.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window

    Returns:
        y: Filtered output signal with length = len(x) - window_size + 1
    """
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    output_length = len(x) - window_size + 1
    y = np.zeros(output_length)

    half_window = window_size // 2
    t = np.arange(window_size)

    # Precompute recency weights
    recency_weights = np.exp(np.linspace(-0.9, 0, window_size))
    
    # Precompute polynomial fitting matrices for dual estimation
    X1 = np.vstack([t, np.ones(window_size)]).T
    X2 = np.vstack([t**2, t, np.ones(window_size)]).T
    X1_pinv = np.linalg.pinv(X1)
    X2_pinv = np.linalg.pinv(X2)

    for i in range(output_length):
        window = x[i : i + window_size]

        # --- Robust Locally Weighted Regression Estimate ---
        mid_region_start = max(0, half_window - 2)
        mid_region_end = min(window_size, half_window + 3)
        mid_est = np.median(window[mid_region_start:mid_region_end]) if window_size >= 8 else window[half_window]
        dev = np.abs(window - mid_est)
        q75, q25 = np.percentile(dev, [75, 25])
        scale = np.median(dev) + 0.5 * (q75 - q25) + 1e-8

        k = 4.5 + 0.5 * min(1.0, window_size / 30)
        r = dev / (k * scale)
        r = np.minimum(r, 1.0)
        r_weights = (1 - r ** 2) ** 2

        W = r_weights * recency_weights
        W_sum = np.sum(W)
        if W_sum < 1e-12:
            W = np.ones(window_size) / window_size
        else:
            W = W / W_sum

        Wx = W * t
        Wxx = Wx * t
        Wx_sum = np.sum(Wx)
        Wxx_sum = np.sum(Wxx)
        denom = Wxx_sum - Wx_sum * Wx_sum

        if np.abs(denom) < 1e-12:
            rlw_estimate = np.sum(W * window)
            slope_rlw = 0.0
        else:
            Wy = W * window
            Wy_sum = np.sum(Wy)
            Wxy_sum = np.sum(Wx * window)
            slope_rlw = (Wxy_sum - Wx_sum * Wy_sum) / denom
            slope_strength = min(1.0, np.abs(slope_rlw) * 5)
            eval_pt = window_size - half_window * (0.4 + 0.25 * slope_strength)
            b0 = (Wy_sum * Wxx_sum - Wx_sum * Wxy_sum) / denom
            rlw_estimate = b0 + slope_rlw * eval_pt

        # --- Adaptive Polynomial Estimate ---
        coeffs2 = X2_pinv @ window
        curvature = 2 * coeffs2[0]
        signal_range = np.ptp(window)
        if signal_range > 1e-10 and abs(curvature) * (window_size**2) > 0.035 * signal_range:
            coeffs = coeffs2
            quadratic_term = coeffs[0]
        else:
            coeffs = X1_pinv @ window
            quadratic_term = 0.0
        poly_slope = coeffs[-2]
        poly_estimate = (quadratic_term * (window_size - 1)**2 + 
                         poly_slope * (window_size - 1) + coeffs[-1])

        # --- Dynamic Blend Based on Slope Agreement & Trend Certainty ---
        # Noise estimation for normalization
        diffs = np.diff(window[-10:])
        noise_est = np.median(np.abs(diffs)) / 0.6745 if len(diffs) > 0 else 1e-6
        noise_est = max(noise_est, 1e-8)
        signal_magnitude = max(1e-6, np.mean(np.abs(window)))
        trend_snr = (np.abs(slope_rlw) + np.abs(poly_slope)) * window_size / max(noise_est * np.sqrt(window_size), signal_magnitude * 0.01)
        
        # Agreement between slopes: increases confidence in polynomial for sharp trends
        slope_agree = np.exp(-3 * np.abs(slope_rlw - poly_slope) / max(np.abs(slope_rlw) + np.abs(poly_slope), 0.001))
        poly_weight = 0.35 + 0.55 * min(1.0, trend_snr * 0.4) * (0.6 + 0.4 * slope_agree)
        poly_weight = max(0.2, min(0.85, poly_weight))
        
        y[i] = (1 - poly_weight) * rlw_estimate + poly_weight * poly_estimate

    # --- Post-Processing: Trend Consistency & False Reversal Penalty ---
    if output_length >= 5:
        trend_smoothed = np.copy(y)
        dy = np.diff(y)
        
        # First pass: Penalize isolated false reversals
        if output_length >= 4:
            sign_changes = np.where(np.diff(np.sign(dy)))[0] + 1
            for sc_pos in sign_changes:
                if 1 <= sc_pos < len(dy) - 1:
                    prev_diff = dy[sc_pos - 1]
                    curr_diff = dy[sc_pos]
                    next_diff = dy[sc_pos + 1]
                    if (np.sign(prev_diff) == np.sign(next_diff) and 
                        np.sign(curr_diff) == -np.sign(prev_diff) and
                        np.abs(curr_diff) < 1.5 * max(np.abs(prev_diff), np.abs(next_diff))):
                        if sc_pos + 1 < output_length:
                            mid = (y[sc_pos - 1] + y[sc_pos + 1]) / 2
                            blend = 0.6
                            trend_smoothed[sc_pos] = (1 - blend) * y[sc_pos] + blend * mid
        
        # Second pass: Causal consistency filtering
        for j in range(1, output_length - 1):
            ctx_start = max(0, j - 3)
            ctx_end = j + 1
            neighbors = y[ctx_start:ctx_end]
            local_range = np.max(neighbors) - np.min(neighbors) + 1e-8
            w_ctx = np.exp(np.linspace(-0.5, 0, len(neighbors)))
            w_ctx = w_ctx / np.sum(w_ctx)
            local_trend = np.sum(w_ctx * neighbors)
            dev_from_trend = np.abs(y[j] - local_trend)
            if j >= 3:
                momentum = np.sum(dy[max(0, j-3):j])
                momentum_factor = max(0, 1 - np.abs(momentum) * 2)
            else:
                momentum_factor = 1.0
            blend = 0.15 + 0.3 * min(1.0, dev_from_trend / (local_range * 0.5)) * momentum_factor
            trend_smoothed[j] = (1 - blend) * trend_smoothed[j] + blend * local_trend
        
        y = trend_smoothed

    return y


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    """
    Main signal processing function that applies the selected algorithm.

    Args:
        input_signal: Input time series data
        window_size: Window size for processing
        algorithm_type: Type of algorithm to use ("basic" or "enhanced")

    Returns:
        Filtered signal
    """
    if algorithm_type == "enhanced":
        return enhanced_filter_with_trend_preservation(input_signal, window_size)
    else:
        return adaptive_filter(input_signal, window_size)


# EVOLVE-BLOCK-END


def generate_test_signal(length=1000, noise_level=0.3, seed=42):
    """
    Generate synthetic test signal with known characteristics.

    Args:
        length: Length of the signal
        noise_level: Standard deviation of noise to add
        seed: Random seed for reproducibility

    Returns:
        Tuple of (noisy_signal, clean_signal)
    """
    np.random.seed(seed)
    t = np.linspace(0, 10, length)

    # Create a complex signal with multiple components
    clean_signal = (
        2 * np.sin(2 * np.pi * 0.5 * t)  # Low frequency component
        + 1.5 * np.sin(2 * np.pi * 2 * t)  # Medium frequency component
        + 0.5 * np.sin(2 * np.pi * 5 * t)  # Higher frequency component
        + 0.8 * np.exp(-t / 5) * np.sin(2 * np.pi * 1.5 * t)  # Decaying oscillation
    )

    # Add non-stationary behavior
    trend = 0.1 * t * np.sin(0.2 * t)  # Slowly varying trend
    clean_signal += trend

    # Add random walk component for non-stationarity
    random_walk = np.cumsum(np.random.randn(length) * 0.05)
    clean_signal += random_walk

    # Add noise
    noise = np.random.normal(0, noise_level, length)
    noisy_signal = clean_signal + noise

    return noisy_signal, clean_signal


def run_signal_processing(noisy_signal=None, signal_length=1000, noise_level=0.3, window_size=20):
    """
    Run the signal processing algorithm on a test signal.

    Args:
        noisy_signal: Input signal to filter (if provided, use this; otherwise generate)
        signal_length: Length if generating signal (for backward compatibility)
        noise_level: Noise level if generating signal (for backward compatibility)
        window_size: Window size for processing

    Returns:
        Dictionary containing results and metrics
    """
    # Use provided signal or generate test signal (for backward compatibility)
    if noisy_signal is not None:
        # Filter the provided signal
        filtered_signal = process_signal(noisy_signal, window_size, "enhanced")
        clean_signal = None  # Not available when using provided signal
    else:
        # Generate test signal (for __main__ and backward compatibility)
        noisy_signal, clean_signal = generate_test_signal(signal_length, noise_level)
        filtered_signal = process_signal(noisy_signal, window_size, "enhanced")

    # Calculate basic metrics (only if we have clean_signal from generation)
    if len(filtered_signal) > 0 and clean_signal is not None:
        # Align signals for comparison (account for processing delay)
        delay = window_size - 1
        aligned_clean = clean_signal[delay:]
        aligned_noisy = noisy_signal[delay:]

        # Ensure same length
        min_length = min(len(filtered_signal), len(aligned_clean))
        filtered_signal = filtered_signal[:min_length]
        aligned_clean = aligned_clean[:min_length]
        aligned_noisy = aligned_noisy[:min_length]

        # Calculate correlation with clean signal
        correlation = np.corrcoef(filtered_signal, aligned_clean)[0, 1] if min_length > 1 else 0

        # Calculate noise reduction
        noise_before = np.var(aligned_noisy - aligned_clean)
        noise_after = np.var(filtered_signal - aligned_clean)
        noise_reduction = (noise_before - noise_after) / noise_before if noise_before > 0 else 0

        return {
            "filtered_signal": filtered_signal,
            "clean_signal": aligned_clean,
            "noisy_signal": aligned_noisy,
            "correlation": correlation,
            "noise_reduction": noise_reduction,
            "signal_length": min_length,
        }
    elif len(filtered_signal) > 0:
        # When using provided signal (no clean_signal available), just return filtered signal
        return {
            "filtered_signal": filtered_signal,
            "clean_signal": None,
            "noisy_signal": None,
            "correlation": 0,
            "noise_reduction": 0,
            "signal_length": len(filtered_signal),
        }
    else:
        return {
            "filtered_signal": [],
            "clean_signal": [],
            "noisy_signal": [],
            "correlation": 0,
            "noise_reduction": 0,
            "signal_length": 0,
        }


if __name__ == "__main__":
    # Test the algorithm
    results = run_signal_processing()
    print("Signal processing completed!")
    print(f"Correlation with clean signal: {results['correlation']:.3f}")
    print(f"Noise reduction: {results['noise_reduction']:.3f}")
    print(f"Processed signal length: {results['signal_length']}")