# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series

This algorithm implements a sliding window approach to filter volatile, non-stationary
time series data while minimizing noise and preserving signal dynamics. It incorporates
adaptive window sizing, robust outlier handling, polynomial fitting, trend reversal detection,
Kalman-inspired smoothing, and multi-component fusion to optimize slope change minimization,
lag error, tracking accuracy, and false reversal penalties.
"""
import numpy as np


def adaptive_filter(x, window_size=20):
    """
    Adaptive signal processing algorithm using sliding window approach with
    multiple adaptive enhancements including noise-based window sizing and
    polynomial predictive filtering.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Base size of the sliding window (W samples)

    Returns:
        y: Filtered output signal with length = len(x) - window_size + 1
    """
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    output_length = len(x) - window_size + 1
    y = np.zeros(output_length)

    # Estimate global noise level for adaptive decisions
    if len(x) > 1:
        noise_est = np.median(np.abs(np.diff(x))) / 0.6745
    else:
        noise_est = 1.0

    # Kalman-like state for recursive smoothing
    kalman_gain = 0.1
    prev_filtered = x[0] if len(x) > 0 else 0

    for i in range(output_length):
        # Base window
        window = x[i : i + window_size]

        # Adaptive window sizing based on recent signal volatility
        recent_window = x[max(0, i) : min(len(x), i + window_size)]
        local_vol = np.std(recent_window) if len(recent_window) > 1 else noise_est
        vol_ratio = local_vol / (noise_est + 1e-8)
        adaptive_w = int(window_size * max(0.5, min(1.5, 1.0 / (vol_ratio + 0.5))))
        adaptive_w = max(5, min(window_size * 2, adaptive_w))

        # Extract adaptively sized window centered on current position
        center_idx = min(i + window_size // 2, len(x) - 1)
        half_w = adaptive_w // 2
        start_idx = max(0, center_idx - half_w)
        end_idx = min(len(x), center_idx + half_w + (adaptive_w % 2))
        adaptive_window = x[start_idx:end_idx]

        # Exponential weights emphasizing recent samples
        w_len = len(adaptive_window)
        exp_weights = np.exp(np.linspace(-2, 0, w_len))
        exp_weights = exp_weights / np.sum(exp_weights)
        weighted_val = np.sum(adaptive_window * exp_weights)

        # 1st-order predictive component on recent slopes
        slope_weights = np.exp(np.linspace(-1, 0, window_size))
        slope_weights = slope_weights / np.sum(slope_weights)
        recent_x = np.arange(window_size)
        recent_y_vals = window
        weighted_slope = np.sum(slope_weights * (recent_y_vals - np.mean(recent_y_vals)) * (recent_x - np.mean(recent_x))) / (np.sum(slope_weights * (recent_x - np.mean(recent_x))**2) + 1e-8)
        predicted = weighted_val + 0.1 * weighted_slope

        # Kalman-like fusion with previous output (reduces spurious reversals)
        kalman_gain = min(0.5, max(0.05, vol_ratio * 0.2))
        fused = kalman_gain * predicted + (1 - kalman_gain) * prev_filtered
        y[i] = fused
        prev_filtered = fused

    return y


def _detect_trend_reversal(window, recent_slope, noise_est):
    """
    Detect potential false trend reversals using cumulative deviation and slope consistency.
    """
    w_len = len(window)
    if w_len < 4:
        return False, recent_slope

    # Compute slopes over sub-windows
    half = w_len // 2
    slope1 = np.polyfit(np.arange(half), window[:half], 1)[0]
    slope2 = np.polyfit(np.arange(w_len - half), window[-half:], 1)[0]

    # Slope sign change with magnitude check vs noise
    sign_change = (slope1 * slope2) < -1e-8
    rel_magnitude = abs(slope2 - slope1) / (noise_est + 1e-8)

    return sign_change and rel_magnitude > 0.3, slope2


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Enhanced version with comprehensive trend preservation featuring:
    adaptive exponential weighting, robust MAD-based outlier handling,
    trend reversal detection and penalties, multi-order polynomial fitting,
    and fused predictions for low-latency tracking with Kalman-inspired smoothing.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window

    Returns:
        y: Filtered output signal
    """
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    x = np.asarray(x, dtype=np.float64)
    output_length = len(x) - window_size + 1
    y = np.zeros(output_length)

    # Robust noise estimate
    if len(x) > 1:
        noise_est = np.median(np.abs(np.diff(x))) / 0.6745
    else:
        noise_est = 1.0

    # State tracking for reversal penalty and smoothing
    prev_slope = 0.0
    prev_filtered = x[0] if len(x) > 0 else 0
    reversal_count = 0

    # Precompute base weights that emphasize recent samples
    base_weights = np.exp(np.linspace(-3, 0, window_size))
    base_weights = base_weights / np.sum(base_weights)
    t_poly = np.arange(window_size, dtype=np.float64)

    for i in range(output_length):
        window = x[i : i + window_size]

        # Calculate window statistics
        win_var = np.var(window)

        # Detect outliers using MAD (Median Absolute Deviation) - from Exemplar 2
        win_median = np.median(window)
        mad = np.median(np.abs(window - win_median))
        mad = mad if mad > 1e-10 else 1e-10

        # Compute variance-adaptive outlier robust weights - from Exemplar 2
        devs = np.abs(window - win_median)
        outlier_mask = devs < (3.0 * mad)  # 3σ equivalent for MAD

        # Compute adaptive combination weights
        adaptive_weights = base_weights.copy()
        adaptive_weights[~outlier_mask] *= 0.1  # Downweight outliers
        adaptive_weights = adaptive_weights / np.sum(adaptive_weights)

        # 1. Adaptive exponential weighted base - from Exemplar 1
        exp_weights = np.exp(np.linspace(-2.5, 0, window_size))
        exp_weights = exp_weights / np.sum(exp_weights)
        exp_filtered = np.sum(window * exp_weights)

        # 2. Weighted polynomial prediction for trend tracking - from Exemplar 1, enhanced with Exemplar 2's robust weights
        coefs = np.polyfit(t_poly, window, 2, w=adaptive_weights)
        poly_pred = coefs[0] * (window_size - 1)**2 + coefs[1] * (window_size - 1) + coefs[2]
        current_slope = coefs[1]  # Linear term as slope

        # 3. Trend reversal detection and penalty - from Exemplar 1
        is_reversal, _ = _detect_trend_reversal(window, prev_slope, noise_est)

        reversal_penalty = 1.0
        if is_reversal:
            reversal_penalty = 0.3
            reversal_count += 1
        else:
            reversal_count = max(0, reversal_count - 1)

        # Fuse components: weight by reversal penalty and signal quality
        vol = np.sqrt(win_var) if win_var > 0 else noise_est
        vol_ratio = vol / (noise_est + 1e-8)
        blend = min(0.8, max(0.4, 0.65 + (vol_ratio - 1) * 0.05)) * reversal_penalty
        fused = blend * poly_pred + (1 - blend) * exp_filtered

        # Kalman-like fusion with previous output for smooth transitions - from Exemplar 1 adaptive_filter
        kalman_gain = min(0.5, max(0.05, vol_ratio * 0.2))
        fused = kalman_gain * fused + (1 - kalman_gain) * prev_filtered

        # Final recent sample blending for low latency - from Exemplar 2
        recent_weight = min(0.6, 0.3 + 0.3 * win_var / (win_var + 1.0))
        final_val = (1.0 - recent_weight) * fused + recent_weight * window[-1]

        y[i] = final_val
        prev_filtered = final_val
        prev_slope = current_slope

    return y


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    """
    Main signal processing function that applies the selected algorithm
    with adaptive parameter selection based on input signal characteristics.

    Args:
        input_signal: Input time series data
        window_size: Window size for processing
        algorithm_type: Type of algorithm to use ("basic" or "enhanced")

    Returns:
        Filtered signal
    """
    # Adaptive window size based on signal length if default
    if window_size <= 0:
        window_size = min(30, max(5, len(input_signal) // 50)) if len(input_signal) > 0 else 20

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