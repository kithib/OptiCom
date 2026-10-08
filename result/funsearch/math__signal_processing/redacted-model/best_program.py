"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series

This algorithm implements a sliding window approach to filter volatile, non-stationary
time series data while minimizing noise and preserving signal dynamics. Hybrid version
combining best traits from top exemplars: outlier detection, adaptive weighting, 
polynomial regression, Kalman filtering, and false reversal protection.
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


def _detect_outliers(window, threshold=3.5):
    """
    Detect outliers in a window using Modified Z-score method.
    
    Args:
        window: Input window array
        threshold: Modified Z-score threshold for outlier detection
        
    Returns:
        Boolean array where True indicates an outlier
    """
    median = np.median(window)
    mad = np.median(np.abs(window - median))
    if mad == 0:
        mad = np.std(window)
    if mad == 0:
        return np.zeros_like(window, dtype=bool)
    modified_z = 0.6745 * (window - median) / mad
    return np.abs(modified_z) > threshold


def _estimate_noise_level(window):
    """Estimate noise level using median absolute deviation."""
    return np.median(np.abs(np.diff(window))) / 0.6745


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Enhanced hybrid version combining: outlier detection, adaptive weighted polynomial 
    regression, Kalman filtering, and trend reversal protection. Optimized for:
    slope change minimization, lag error minimization, tracking accuracy, and false reversal penalty.

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
    
    # Polynomial order constraint
    poly_order = min(2, max(1, window_size // 7))

    # Precompute base position matrix for polynomial fitting
    t = np.arange(window_size)
    A_base = np.vstack([t**k for k in range(poly_order + 1)]).T
    
    # Initial noise estimate
    global_noise_est = _estimate_noise_level(x[:min(200, len(x))])

    # Kalman filter parameters (from Exemplar 2)
    k_estimate = x[0] if len(x) > 0 else 0
    k_error = 1.0
    q = 0.001  # Process noise
    r = 0.1    # Measurement noise

    # Previous values for slope reversal protection
    prev_slope = 0.0

    for i in range(output_length):
        window = x[i : i + window_size]

        # --- Step 1: Outlier detection (Exemplar 1) ---
        outliers = _detect_outliers(window)
        clean_window = window[~outliers]
        if len(clean_window) < 3:
            clean_window = window

        # --- Step 2: Local trend for adaptive weighting (Exemplar 1) ---
        recent_window = window[max(0, window_size - 10):]
        recent_trend = np.polyfit(np.arange(len(recent_window)), recent_window, 1)[0]
        
        # Adaptive decay based on trend magnitude
        trend_mag = abs(recent_trend)
        decay_param = min(4.0, 2.0 + min(2.0, trend_mag))
        weights = np.exp(np.linspace(-decay_param, 0, window_size))
        weights[outliers] *= 0.1
        weights = weights / np.sum(weights)

        # --- Step 3: Weighted polynomial regression (Exemplar 1) ---
        W = np.diag(weights)
        A = A_base
        AtW = A.T @ W
        coeffs = np.linalg.lstsq(AtW @ A + 1e-8 * np.eye(poly_order+1), AtW @ window, rcond=None)[0]
        poly_pred = np.sum(coeffs * (window_size - 1)**np.arange(poly_order+1))

        # --- Step 4: Calculate current slope with false reversal penalty (Exemplar 2) ---
        trend_coeffs = np.polyfit(t, window, 1)
        current_slope = trend_coeffs[0]
        
        if i > 0:
            # Combined slope reversal protection (both exemplars' techniques)
            slope_change = current_slope - prev_slope
            if prev_slope * current_slope < -0.1:
                current_slope = prev_slope * 0.6 + current_slope * 0.4
            
            # Additional noise-based reversal protection
            prev_slope_val = y[i-1] - y[i-2] if i >= 2 else 0
            new_slope_val = poly_pred - y[i-1]
            if abs(prev_slope_val) < 2 * global_noise_est and abs(new_slope_val) < 2 * global_noise_est:
                if prev_slope_val * new_slope_val < 0:
                    new_slope_val = 0.2 * new_slope_val
                    poly_pred = y[i-1] + new_slope_val
        
        prev_slope = current_slope

        # Trend-adjusted estimate
        weighted_est = np.sum(window * weights)
        trend_est = weighted_est + current_slope * (window_size / 4)

        # --- Step 5: Kalman filtering (Exemplar 2) ---
        k_gain = k_error / (k_error + r)
        hybrid_meas = poly_pred * 0.6 + trend_est * 0.4
        k_estimate = k_estimate + k_gain * (hybrid_meas - k_estimate)
        k_error = (1 - k_gain) * k_error + q
        
        # Adaptive process noise
        if i > window_size // 2:
            recent_volatility = np.std(x[i - window_size//2 : i + 1])
            q = np.clip(recent_volatility * 0.01, 0.0001, 0.01)

        # --- Step 6: Final hybrid output ---
        y[i] = k_estimate * 0.5 + poly_pred * 0.5

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