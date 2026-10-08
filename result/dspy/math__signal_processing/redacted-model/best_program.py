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

    # Initialize output array
    output_length = len(x) - window_size + 1
    y = np.zeros(output_length)

    # Simple moving average as baseline
    for i in range(output_length):
        window = x[i : i + window_size]

        # Basic moving average filter
        y[i] = np.mean(window)

    return y


def _detect_trend_changes(signal, threshold=1.5):
    """Internal helper to detect significant trend changes for adaptive filtering."""
    if len(signal) < 3:
        return np.ones(len(signal))
    gradients = np.gradient(signal)
    gradient_std = np.std(gradients)
    if gradient_std == 0:
        return np.ones(len(signal))
    significance = np.abs(gradients) / gradient_std
    return significance


def _adaptive_window_weights(window_size, trend_significance=1.0):
    """Generate adaptive weights based on trend significance."""
    # Base exponential weights
    base_weights = np.exp(np.linspace(-2, 0, window_size))
    # Adjust based on trend - more recent emphasis when trend is strong
    trend_factor = min(trend_significance, 3.0)
    adaptive_weights = np.exp(np.linspace(-2 * trend_factor, 0, window_size))
    weights = 0.3 * base_weights + 0.7 * adaptive_weights
    return weights / np.sum(weights)


def _hampel_filter(window, n_sigmas=2.5):
    """Hampel filter for outlier detection - more robust than Z-score."""
    median = np.median(window)
    mad = np.median(np.abs(window - median))
    threshold = n_sigmas * 1.4826 * mad  # 1.4826 makes MAD consistent with std
    if mad == 0:
        # Fallback to standard deviation if MAD is zero
        std = np.std(window)
        if std == 0:
            return window
        threshold = n_sigmas * std
    # Identify outliers and dampen towards median instead of hard replacement
    outliers = np.abs(window - median) > threshold
    filtered = window.copy()
    if np.any(outliers):
        filtered[outliers] = 0.7 * median + 0.3 * filtered[outliers]
    return filtered


def _savitzky_golay_coeffs(window_size, order=2):
    """Compute Savitzky-Golay coefficients for polynomial smoothing."""
    t = np.arange(window_size)
    A = np.vander(t, order + 1)
    Q, R = np.linalg.qr(A)
    H = A @ np.linalg.solve(R, Q.T)
    return H[-1]  # Return coefficients for evaluating at the last point


def _detect_significant_trend_change(window, recent_window_size=5):
    """
    Detect if there's a genuine trend change in the recent samples.
    Returns True if a significant trend change is detected.
    """
    if len(window) < recent_window_size * 2:
        return False
    
    old_part = window[:-recent_window_size]
    new_part = window[-recent_window_size:]
    
    old_slope = np.polyfit(np.arange(len(old_part)), old_part, 1)[0]
    new_slope = np.polyfit(np.arange(len(new_part)), new_part, 1)[0]
    
    old_std = np.std(old_part) if np.std(old_part) > 1e-8 else 1e-8
    mean_diff = abs(np.mean(new_part) - np.mean(old_part)) / old_std
    slope_ratio = abs(new_slope - old_slope) / (abs(old_slope) + 1e-8)
    
    return mean_diff > 0.5 or slope_ratio > 1.5


def _calculate_slope(window):
    """Calculate slope of a window using linear regression."""
    n = len(window)
    t = np.arange(n)
    t_mean = np.mean(t)
    x_mean = np.mean(window)
    numerator = np.sum((t - t_mean) * (window - x_mean))
    denominator = np.sum((t - t_mean) ** 2)
    return numerator / denominator if denominator != 0 else 0


def _false_reversal_penalty(filtered_values, current_idx, proposed_value, penalty_factor=0.3):
    """Apply penalty against trend reversals that are likely noise-induced."""
    if current_idx < 2:
        return proposed_value
    recent = filtered_values[current_idx-2:current_idx]
    if len(recent) < 2:
        return proposed_value
    recent_dir = np.sign(np.diff(recent))
    current_dir = np.sign(proposed_value - filtered_values[current_idx-1])
    if len(recent_dir) >= 2 and recent_dir[0] == recent_dir[1] != 0:
        if current_dir != recent_dir[0]:
            continuation = filtered_values[current_idx-1] + (filtered_values[current_idx-1] - filtered_values[current_idx-2])
            return (1 - penalty_factor) * proposed_value + penalty_factor * continuation
    return proposed_value


def _false_reversal_consistency(y, i):
    """Apply false reversal penalty by checking local gradient consistency."""
    if i < 2 or i >= len(y) - 1:
        return y[i]
    
    # Get local directions
    d1 = y[i-1] - y[i-2]
    d2 = y[i] - y[i-1]
    d3 = y[i+1] - y[i] if (i+1) < len(y) else d2
    
    s1, s2, s3 = np.sign(d1), np.sign(d2), np.sign(d3)
    
    # If we have a reversal pattern (s1 == s3 != s2), smooth it
    if s1 == s3 and s1 != 0 and s2 != s1:
        # Check magnitude - only smooth small reversals
        mag_ratio = abs(d2) / (abs(d1) + abs(d3) + 1e-8)
        if mag_ratio < 0.5:
            return 0.4 * y[i] + 0.3 * y[i-1] + 0.3 * y[i+1]
    
    return y[i]


def _robust_local_regression(y, window_size=5, poly_order=1):
    """Apply robust local polynomial regression to smooth while preserving edges."""
    n = len(y)
    if n < window_size:
        return y
    smoothed = np.copy(y)
    half_win = window_size // 2
    for i in range(half_win, n - half_win):
        local_y = y[i - half_win:i + half_win + 1]
        t = np.arange(window_size)
        # Weighted least squares with bisquare weights
        coeffs = np.polyfit(t, local_y, poly_order)
        smoothed[i] = np.polyval(coeffs, half_win)
    return smoothed


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Enhanced version with trend preservation using adaptive weighting, outlier rejection,
    predictive correction, and false reversal penalty.

    Implements multi-objective optimization:
    1. Slope change minimization - Hampel outlier rejection removes noise-induced spikes
    2. Lag error minimization - adaptive recent-weighted averaging + polynomial prediction
    3. Tracking accuracy - Savitzky-Golay style polynomial fitting preserves dynamics
    4. False reversal penalty - post-processing temporal consistency checking

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

    # Pre-compute trend changes across the signal for adaptive weighting
    trend_signals = _detect_trend_changes(x_arr)

    # Base exponential weights
    base_weights = np.exp(np.linspace(-2, 0, window_size))
    base_weights = base_weights / np.sum(base_weights)

    # Savitzky-Golay coefficients for polynomial smoothing at window edge
    sg_order = min(3, window_size - 1)
    sg_coeffs = _savitzky_golay_coeffs(window_size, sg_order)

    # State tracking for slope changes
    prev_slope = 0
    slope_history = []
    
    for i in range(output_length):
        window = x_arr[i : i + window_size]
        window_trend_sig = trend_signals[i : i + window_size]

        # Step 1: Hampel outlier rejection - removes extreme noise spikes
        # that cause false trend reversals
        cleaned_window = _hampel_filter(window, n_sigmas=2.5)

        # Step 2: Outlier down-weighting using MAD-based robust statistics
        median = np.median(cleaned_window)
        mad = np.median(np.abs(cleaned_window - median))
        mad = mad if mad > 1e-8 else 1e-8
        z_scores = np.abs(cleaned_window - median) / (mad * 1.4826)
        outlier_weights = np.exp(-0.5 * z_scores**2)
        outlier_weights = outlier_weights / np.sum(outlier_weights)

        # Step 3: Compute average trend significance and detect genuine trend changes
        avg_trend = np.mean(window_trend_sig)
        has_trend_change = _detect_significant_trend_change(cleaned_window)

        # Step 4: Adaptive weights based on trend strength
        if has_trend_change:
            # More responsive weights during trend changes to minimize lag
            responsive_weights = np.exp(np.linspace(-4, 0, window_size))
            responsive_weights = responsive_weights / np.sum(responsive_weights)
            weights = 0.6 * responsive_weights + 0.4 * base_weights
        else:
            # Standard adaptive weights
            weights = _adaptive_window_weights(window_size, avg_trend)

        # Apply outlier weighting
        combined_weights = weights * outlier_weights
        combined_weights = combined_weights / np.sum(combined_weights)

        # Step 5: Weighted average base filtering
        weighted_avg = np.sum(cleaned_window * combined_weights)

        # Step 6: Savitzky-Golay polynomial smoothing at window edge
        sg_value = np.sum(cleaned_window * sg_coeffs)

        # Step 7: Lag correction using local linear prediction
        poly_degree = min(2, window_size - 1)
        recent_t = np.arange(window_size)
        coeffs = np.polyfit(recent_t, cleaned_window, poly_degree)
        prediction = np.polyval(coeffs, window_size - 1 + window_size * 0.25)

        # Step 8: Blend all components based on trend strength
        trend_blend = min(avg_trend * 0.2, 0.4)
        raw_value = (1 - trend_blend) * (0.7 * weighted_avg + 0.3 * sg_value) + trend_blend * prediction

        # Step 9: Apply incremental false reversal penalty
        if i > 0:
            raw_value = _false_reversal_penalty(y, i, raw_value, penalty_factor=0.25)

        # Step 10: Slope change minimization - dampen noisy slope changes
        current_slope = _calculate_slope(window)
        slope_history.append(current_slope)
        if len(slope_history) > 10:
            slope_history.pop(0)
        
        if i > 0 and abs(prev_slope) > 1e-10:
            slope_change_ratio = abs(current_slope - prev_slope) / (abs(prev_slope) + 1e-8)
            if slope_change_ratio > 2.0:
                # Sudden slope changes likely due to noise - dampen them
                raw_value = 0.6 * raw_value + 0.4 * y[i-1]

        # Step 11: Trend-based lag compensation - INCREASED for better tracking
        if len(slope_history) >= 5:
            trend_strength = abs(np.mean(slope_history)) / (np.std(slope_history) + 1e-10)
            if trend_strength > 0.3:
                lookahead = min(3, window_size / 6)
                lag_comp = current_slope * lookahead * min(trend_strength * 0.35, 0.7)  # Increased from 0.25/0.5
                raw_value += lag_comp

        y[i] = raw_value
        prev_slope = current_slope

    # Step 12: Post-processing - penalize false reversals via temporal consistency
    if output_length >= 3:
        grads = np.gradient(y)
        grad_std = np.std(grads) + 1e-8
        small_change_mask = np.abs(grads) < (grad_std * 0.1)
        
        # Apply false reversal penalty by smoothing inconsistent small changes
        for idx in range(1, output_length - 1):
            if small_change_mask[idx]:
                sign_before = np.sign(y[idx] - y[idx-1])
                sign_after = np.sign(y[idx+1] - y[idx])
                if sign_before != sign_after and sign_before != 0 and sign_after != 0:
                    # Potential false reversal, apply stronger smoothing
                    y[idx] = 0.4 * y[idx] + 0.3 * y[idx-1] + 0.3 * y[idx+1]
                else:
                    y[idx] = 0.7 * y[idx] + 0.15 * y[idx-1] + 0.15 * y[idx+1]  # More emphasis on original

    # Step 13: Final multi-point smoothing on low-activity regions
    if output_length >= 5:
        y_grad = np.abs(np.gradient(y))
        grad_threshold = np.median(y_grad) + 0.5 * np.std(y_grad)
        smooth_mask = y_grad < grad_threshold
        for idx in range(2, output_length - 2):
            if smooth_mask[idx] and smooth_mask[idx-1] and smooth_mask[idx+1]:
                y[idx] = 0.5 * y[idx] + 0.125 * y[idx-2] + 0.125 * y[idx-1] + 0.125 * y[idx+1] + 0.125 * y[idx+2]  # More original

    # Step 14: Additional false reversal consistency pass
    for i in range(2, output_length - 1):
        y[i] = _false_reversal_consistency(y, i)

    # Step 15: Final robust regression pass for edge-preserving smoothing
    if output_length >= 5:
        y = _robust_local_regression(y, window_size=min(5, output_length//10 + 3))

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