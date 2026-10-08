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

    # Vectorized moving average using sliding window view for efficiency
    output_length = len(x) - window_size + 1
    y = np.lib.stride_tricks.sliding_window_view(x, window_size).mean(axis=1)

    return y


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Enhanced hybrid version combining best traits from top exemplars:
    - 2nd order polynomial predictive enhancement for trend curvature capture
    - Safer sliding_window_view for efficiency and readability
    - Recent slope history for false reversal penalty (5-point history with sign consensus)
    - Incremental estimate blending and lag correction for reduced lag error
    - Post-processing reversal smoothing to minimize spurious directional reversals
    - SNR-based adaptive weighting for consistent trend tracking
    - False reversal penalty using consensus and slope difference metrics

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window

    Returns:
        y: Filtered output signal
    """
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    output_length = len(x) - window_size + 1
    y = np.zeros(output_length)
    x_arr = np.asarray(x, dtype=np.float64)

    # Use sliding_window_view which is safer and more readable than as_strided
    windows = np.lib.stride_tricks.sliding_window_view(x_arr, window_size)

    # Base exponential weights that emphasize recent samples (steeper for better lag reduction)
    base_weights = np.exp(np.linspace(-2.5, 0, window_size))
    base_weights = base_weights / np.sum(base_weights)

    # Precompute polynomial fitting matrix for 2nd order predictive enhancement
    t_poly = np.arange(window_size)
    # Use 2nd order polynomial for better trend curvature capture
    X_poly = np.vstack([t_poly**2, t_poly, np.ones(window_size)]).T
    XtX_inv = np.linalg.pinv(X_poly, rcond=1e-10)

    # Precompute half window for efficient trend comparison
    half = max(2, window_size // 2)
    t_half = np.arange(half)
    X_half = np.vstack([t_half, np.ones(half)]).T
    X_half_inv = np.linalg.pinv(X_half, rcond=1e-10)

    # Store recent slopes for better consistency checking
    recent_slopes_history = np.zeros(5)
    history_idx = 0

    # Previous slope for Kalman-like filtering (from Exemplar 1)
    prev_filtered_slope = 0.0

    for i in range(output_length):
        window = windows[i]

        # 2nd order polynomial trend estimation
        poly_coeffs = XtX_inv @ window
        trend_slope = poly_coeffs[1] + 2 * poly_coeffs[0] * (window_size - 1)  # Derivative at end point
        pred_quad = poly_coeffs[0] * (window_size - 1)**2 + poly_coeffs[1] * (window_size - 1) + poly_coeffs[2]

        # Efficient trend strength comparison
        recent_coeffs = X_half_inv @ window[-half:]
        older_coeffs = X_half_inv @ window[:half]
        recent_slope = recent_coeffs[0]
        older_slope = older_coeffs[0]

        # Estimate local volatility using robust statistics
        fit_vals = poly_coeffs[0] * t_poly**2 + poly_coeffs[1] * t_poly + poly_coeffs[2]
        detrended = window - fit_vals
        local_vol = np.median(np.abs(detrended)) + 1e-8
        noise_level = np.median(np.abs(np.diff(window))) + 1e-8
        window_var = np.var(window)

        # Calculate trend consistency with history
        slope_sign_consistency = 1.0 - 0.5 * abs(np.sign(recent_slope) - np.sign(older_slope))
        slope_magnitude = abs(recent_slope)

        # Enhanced false reversal penalty using recent slope history (combined approach)
        false_reversal_penalty = 1.0
        if i > 0:
            # Update slope history
            recent_slopes_history[history_idx] = recent_slope
            history_idx = (history_idx + 1) % 5

            if i >= 4:
                # Check for consistent slope sign in history
                history_signs = np.sign(recent_slopes_history)
                sign_consensus = np.abs(np.mean(history_signs))
                consensus_penalty = 0.4 + 0.6 * sign_consensus  # Stronger penalty for sign changes
                false_reversal_penalty *= consensus_penalty

            # Original slope difference penalty (from Exemplar 2 with factor of 3)
            prev_estimate = y[i-2] if i > 1 else x_arr[i]
            prev_slope = y[i-1] - prev_estimate
            slope_diff = np.abs(trend_slope - prev_slope) / local_vol
            false_reversal_penalty *= np.exp(-slope_diff / 3)  # Slightly stronger penalty

        # Adaptive weighting with normalized penalty (tighter clipping range 0.2-1.0)
        adaptive_weights = base_weights * np.clip(false_reversal_penalty, 0.2, 1.0)
        weights_sum = np.sum(adaptive_weights)
        if weights_sum > 1e-12:
            adaptive_weights = adaptive_weights / weights_sum
        else:
            adaptive_weights = base_weights

        # Predictive enhancement combining weighted average with enhanced prediction
        weighted_avg = np.sum(window * adaptive_weights)

        # Blend prediction and weighted average based on trend strength and consistency
        # Also incorporate SNR-based blending from Exemplar 2
        snr_proxy = np.abs(recent_slope) * window_size / (np.sqrt(window_var) + 1e-8)
        snr_blend = np.clip(0.3 * snr_proxy, 0.0, 0.7)

        if slope_magnitude > noise_level * 0.5 and slope_sign_consistency > 0.5:
            # Strong consistent trend - give more weight to quadratic prediction
            trend_weight = min(0.7, slope_magnitude / noise_level * 0.55)  # Higher max weight
            trend_weight = (trend_weight + snr_blend) / 2  # Average with SNR blend
            base_estimate = weighted_avg * (1 - trend_weight) + pred_quad * trend_weight
        else:
            # Weaker or inconsistent trend - stick closer to weighted average
            window_std = np.std(window)
            if window_std > 1e-12:
                vol_factor = np.clip(local_vol / window_std, 0.15, 0.35)  # Narrower range
            else:
                vol_factor = 0.25
            # Incorporate some SNR blend here too
            final_vol = (vol_factor * 2 + snr_blend) / 3
            base_estimate = final_vol * pred_quad + (1 - final_vol) * weighted_avg

        # First-order hold to enforce slope continuity and reduce lag (from Exemplar 1)
        if i > 0:
            # Incremental estimate based on previous filtered slope
            incremental_estimate = y[i-1] + prev_filtered_slope

            # Estimate current expected slope
            if i > 1:
                current_expected_slope = y[i-1] - y[i-2]
            else:
                current_expected_slope = trend_slope

            # Update filtered slope with continuity bias
            filtered_slope = 0.6 * current_expected_slope + 0.4 * trend_slope
            prev_filtered_slope = filtered_slope

            # Blend incremental prediction with base estimate
            # This reduces lag by extrapolating previous trend
            lag_correction = 0.0
            if abs(filtered_slope) > 1e-10:
                # Estimate lag from window phase
                lag_correction = filtered_slope * (window_size * 0.35)

            # Combined final estimate using incremental blending (0.7/0.3 split from Exemplar 1)
            y[i] = 0.7 * base_estimate + 0.3 * (incremental_estimate + lag_correction * 0.1)
        else:
            y[i] = base_estimate
            prev_filtered_slope = trend_slope

    # Post-process for slope change minimization (from Exemplar 2)
    if output_length >= 2:
        dy = np.diff(y)
        dy_mean = np.mean(np.abs(dy))
        for i in range(1, output_length - 1):
            # Detect spurious reversals: sign changes in consecutive derivatives with small magnitude
            prev_slope = dy[i-1]
            curr_slope = dy[i]
            if prev_slope * curr_slope < 0:  # Sign change
                total_mag = np.abs(prev_slope) + np.abs(curr_slope)
                if total_mag < 0.5 * dy_mean + 1e-8:
                    # Penalize false reversal by smoothing over 3 points
                    y[i] = 0.5 * y[i-1] + 0.5 * y[i+1]

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