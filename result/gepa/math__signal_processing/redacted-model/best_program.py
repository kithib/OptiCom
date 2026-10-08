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
    x = np.asarray(x, dtype=np.float64)
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    # Simple moving average as baseline - use vectorized convolution for O(n log n) performance
    kernel = np.ones(window_size) / window_size
    y = np.convolve(x, kernel, mode='valid')

    return y


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Enhanced version with trend preservation using adaptive weighted filtering
    with polynomial fitting for slope change minimization and false reversal penalty.
    Combines 2nd-degree polynomial fitting in high-volatility regions with
    two-pass false reversal detection and adaptive polynomial prediction for optimal performance.
    Adds adaptive window size adjustment, refined volatility thresholding, improved
    polynomial prediction blending for lag error minimization, forward-looking
    polynomial extrapolation bias correction, and Kalman-inspired recursive
    blending to adaptively balance predicted and measured values.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window

    Returns:
        y: Filtered output signal
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    output_length = len(x) - window_size + 1
    if output_length == 0:
        return np.array([], dtype=np.float64)
    
    y = np.zeros(output_length)

    # Base exponential weights that emphasize recent samples - steeper decay (-3.2 range for balance)
    exp_weights = np.exp(np.linspace(-3.2, 0, window_size))
    exp_weights = exp_weights / np.sum(exp_weights)

    # Precompute design matrix for 2nd-degree polynomial fitting
    t_poly = np.linspace(0, 1, window_size)
    X_poly = np.vstack([np.ones(window_size), t_poly, t_poly**2]).T

    # Create array for rolling slopes to detect false reversals
    slopes = np.zeros(output_length)
    
    # Compute global gradient threshold for volatility detection (robust median absolute deviation)
    global_gradients = np.abs(np.diff(x))
    if len(global_gradients) > 0:
        median_grad = np.median(global_gradients)
        mad = np.median(np.abs(global_gradients - median_grad))
        global_threshold = median_grad + 0.45 * mad
    else:
        global_threshold = 0.1
    
    # Adaptive effective window size based on signal-to-noise proxy
    if len(x) > 0:
        signal_range = np.max(x) - np.min(x)
        noise_est = np.median(np.abs(np.diff(x)))
        snr_proxy = signal_range / (noise_est + 1e-8)
        # Adjust effective polynomial window: smaller window for higher SNR (crisper response)
        eff_window_mult = max(0.6, min(1.0, 0.8 + 0.2 / (1.0 + 0.1 * snr_proxy)))
    else:
        eff_window_mult = 0.8

    # First pass: compute filtered values and slopes using adaptive polynomial fitting
    # Initialize Kalman-inspired state variables for predicted/measured blending
    prev_prediction = 0.0
    process_noise = 0.02
    measurement_noise = 0.08
    
    for i in range(output_length):
        window = x[i : i + window_size]
        
        # Calculate local volatility for adaptive filtering (window std for robustness)
        window_std = np.std(window) if np.std(window) > 1e-8 else 1.0
        window_grad = np.abs(np.diff(window))
        volatility = np.mean(window_grad) if len(window_grad) > 0 else 0

        # Adaptive weighting based on volatility strength
        vol_ratio = volatility / global_threshold if global_threshold > 1e-8 else 1.0

        # Detect trend in window using linear regression slope
        t = np.arange(window_size)
        p = np.polyfit(t, window, 1)
        slope, intercept = p[0], p[1]
        slope_magnitude = np.abs(slope)

        # Calculate adaptive volatility factor (higher in more volatile windows)
        volatility_factor = np.mean(np.abs(np.diff(window))) / window_std if window_std > 1e-8 else 0.5

        # 1st order polynomial prediction for current point (reduces lag)
        polynomial_pred = slope * (window_size - 1) + intercept

        # Adaptive blend: more prediction weight when strong trend/high volatility
        # Enhanced blend with optimized coefficients for lag vs noise balance
        pred_blend = min(0.88, 0.38 + 0.34 * slope_magnitude + 0.25 * volatility_factor)

        if vol_ratio > 1.18:
            # High volatility: use 2nd-degree polynomial fit to capture dynamics
            coeffs, _, _, _ = np.linalg.lstsq(X_poly, window, rcond=None)
            # Evaluate polynomial at window end-point to minimize lag
            poly_estimate = coeffs[0] + coeffs[1] * 1.0 + coeffs[2] * 1.0
            # Weighted average base estimate
            weighted_est = np.sum(window * exp_weights)
            # Blended estimate with polynomial prediction (reduces lag error)
            blended_est = weighted_est * (1 - pred_blend) + poly_estimate * pred_blend
            
            # Kalman-inspired blending of prediction vs current estimate
            if i > 0:
                # Prediction from previous state's slope
                innovation = blended_est - prev_prediction
                kalman_gain = process_noise / (process_noise + measurement_noise)
                blended_est = prev_prediction + kalman_gain * innovation
            
            prev_prediction = blended_est + slope * 0.1  # Predict next value
            y[i] = blended_est
            # Extract slope (linear + 2*quadratic at t=1) for false reversal detection
            slopes[i] = coeffs[1] + 2 * coeffs[2]
        elif vol_ratio > 0.72:
            # Medium volatility: weighted blend for balanced performance - lowered threshold from 0.75
            coeffs, _, _, _ = np.linalg.lstsq(X_poly, window, rcond=None)
            poly_estimate = coeffs[0] + coeffs[1] * 1.0 + coeffs[2] * 1.0
            weighted_avg = np.sum(window * exp_weights)
            blend_factor = min(0.88, 0.58 + 0.22 * (vol_ratio - 1.0))
            # Weighted average base estimate
            weighted_est = blend_factor * poly_estimate + (1 - blend_factor) * weighted_avg
            # Blended estimate with polynomial prediction (reduces lag error)
            blended_est = weighted_est * (1 - pred_blend) + poly_estimate * pred_blend
            
            # Kalman-inspired blending
            if i > 0:
                innovation = blended_est - prev_prediction
                kalman_gain = process_noise / (process_noise + measurement_noise * 1.2)
                blended_est = prev_prediction + kalman_gain * innovation
            
            prev_prediction = blended_est + slope * 0.1
            y[i] = blended_est
            slopes[i] = coeffs[1] + 2 * coeffs[2]
        else:
            # Low volatility: use weighted least squares linear fit for robustness
            t_lin = np.arange(window_size)
            W = np.diag(exp_weights)
            X = np.vstack([t_lin, np.ones(window_size)]).T
            XtW = X.T @ W
            coeffs = np.linalg.solve(XtW @ X + 1e-8 * np.eye(2), XtW @ window)
            slope_coeff, intercept_coeff = coeffs
            
            slopes[i] = slope_coeff
            
            # Blend polynomial end-point estimate with weighted average
            poly_estimate = intercept_coeff + slope_coeff * (window_size - 1)
            weighted_avg = np.sum(window * exp_weights)
            # Adjusted blend with higher polynomial weight for reduced lag (0.48 vs 0.45)
            base_estimate = (eff_window_mult * poly_estimate + (1 - eff_window_mult) * 0.48 * poly_estimate 
                            + (1 - eff_window_mult) * 0.52 * weighted_avg)
            # Blended estimate with polynomial prediction (reduces lag error)
            blended_est = base_estimate * (1 - pred_blend) + poly_estimate * pred_blend
            
            # Kalman-inspired blending with heavier smoothing in low volatility
            if i > 0:
                innovation = blended_est - prev_prediction
                kalman_gain = process_noise / (process_noise + measurement_noise * 1.5)
                blended_est = prev_prediction + kalman_gain * innovation
            
            prev_prediction = blended_est + slope * 0.1
            y[i] = blended_est

    # Correction pass: reduce polynomial overshoot by regressing filtered values to recent weighted averages
    if output_length > 5:
        # Precompute trailing exponential weights for bias correction
        correction_weights = np.exp(np.linspace(-1.8, 0, 5))
        correction_weights = correction_weights / np.sum(correction_weights)
        y_corrected = y.copy()
        
        for i in range(3, output_length):
            # Compute robust trailing reference as bias anchor
            trail_start = max(0, i - 5)
            trail_end = i
            trail_y = y[trail_start:trail_end]
            if len(trail_y) >= 3:
                use_weights = correction_weights[-len(trail_y):]
                use_weights = use_weights / np.sum(use_weights)
                trail_ref = np.sum(trail_y * use_weights)
                # Apply mild regression to reference to tame polynomial overshoot
                y_corrected[i] = 0.86 * y[i] + 0.14 * trail_ref
        y = y_corrected

    # Second pass: apply false reversal penalty and slope change minimization
    if output_length > 1:
        slope_changes = np.abs(np.diff(slopes))
        max_slope_change = np.max(slope_changes)
        median_slope_change = np.median(slope_changes)
    else:
        max_slope_change = 0.0
        median_slope_change = 0.0
    
    # Adaptive penalty threshold based on both median and max slope changes
    penalty_base = median_slope_change if median_slope_change > 1e-8 else (0.18 * max_slope_change if max_slope_change > 0 else 0.01)
    penalty_threshold = 0.52 * penalty_base
    
    for i in range(1, output_length):
        # Detect potential false reversal (slope sign change with small magnitude)
        prev_slope = slopes[i-1]
        curr_slope = slopes[i]
        
        if prev_slope * curr_slope < -1e-12:  # Slope sign changed with meaningful magnitude
            slope_magnitude = min(abs(prev_slope), abs(curr_slope))
            if slope_magnitude < penalty_threshold:
                # Apply penalty: adaptively blend previous filtered value to reduce false reversal
                penalty_strength = min(0.68, 0.82 * (1.0 - slope_magnitude / (penalty_threshold + 1e-8)))
                y[i] = (1 - penalty_strength) * y[i] + penalty_strength * y[i-1]
    
    # Third pass: slope continuity smoothing for small, spurious variations with refined bounds
    if output_length > 3:
        y_smoothed = y.copy()
        # Robust std estimate for thresholding
        y_mad = np.median(np.abs(y - np.median(y)))
        y_std = 1.4826 * y_mad if y_mad > 1e-8 else np.std(y)
        
        for i in range(2, output_length - 1):
            # Local second derivative for curvature check
            d2 = y[i+1] - 2 * y[i] + y[i-1]
            prev_d2 = y[i] - 2 * y[i-1] + y[i-2]
            
            # Apply smoothing with refined threshold for better fine-grained control
            if abs(d2) < 0.07 * y_std and prev_d2 * d2 < -1e-6:
                y_smoothed[i] = 0.66 * y[i] + 0.17 * y[i-1] + 0.17 * y[i+1]
        y = y_smoothed

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