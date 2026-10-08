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


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Enhanced version with trend preservation and multi-objective optimization.
    Uses adaptive exponential weighting, polynomial fitting with predictive extrapolation,
    statistical reversal detection, and adaptive blending. Optimizes for:
    1. Slope change minimization (reducing spurious directional reversals)
    2. Lag error minimization (maintaining responsiveness via prediction)
    3. Tracking accuracy (preserving genuine signal trends with polynomial fitting)
    4. False reversal penalty (avoiding noise-induced trend changes via thresholds)

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window

    Returns:
        y: Filtered output signal
    """
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    x = np.asarray(x)
    output_length = len(x) - window_size + 1
    y = np.zeros(output_length)
    
    # Exponential weights: emphasize recent samples but weighted more heavily for responsiveness
    # Base weight distribution: exponential rolloff centered to reduce lag vs variance tradeoff
    base_weights = np.exp(np.linspace(-2.0, 0, window_size))
    base_weights = base_weights / np.sum(base_weights)
    
    # Precompute linear regression matrix (for slope/intercept) to avoid recomputing each iteration
    t = np.arange(window_size)
    A_lin = np.vstack([t, np.ones(window_size)]).T
    A_lin_pinv = np.linalg.pinv(A_lin)  # Precomputed pseudo-inverse for linear fits
    
    # Precompute polynomial regression matrices (2nd order for curvature estimation)
    A_poly2 = np.vstack([t**2, t, np.ones(window_size)]).T
    A_poly2_pinv = np.linalg.pinv(A_poly2)
    
    # State variables for continuity and trend tracking
    prev_slope = 0.0
    prev_filtered = None
    prev_deriv = 0.0
    
    # Noise scale estimation from MAD of differences (robust to spikes/outliers)
    # Use first portion of signal for noise floor estimate
    est_len = min(window_size * 2, len(x))
    noise_mad = np.median(np.abs(np.diff(x[:est_len]))) / 0.6745  # Convert MAD to std approx
    noise_mad = max(noise_mad, 1e-10)  # Guard against degenerate cases
    
    # Additional state for advanced reversal penalty tracking
    slope_history = []
    val_history = []
    max_history = 10
    trend_confidence = 0.5
    
    for i in range(output_length):
        window = x[i : i + window_size]
        
        # 1. Local noise estimation (adapt per window)
        lin_coeffs = A_lin_pinv @ window
        slope, intercept = lin_coeffs
        fit_residual = window - (slope * t + intercept)
        local_std = np.std(fit_residual)
        current_noise = max(local_std, noise_mad)
        
        # Update histories for advanced consistency checking
        slope_history.append(slope)
        val_history.append(window[-1])
        if len(slope_history) > max_history:
            slope_history.pop(0)
            val_history.pop(0)
        
        # 2. Quadratic fit for better trend tracking around inflections/curvature
        quad_coeffs = A_poly2_pinv @ window
        c2, c1, c0 = quad_coeffs
        # Predict at window endpoint using fit for reduced lag (aligns filter output with data end)
        pred_end_quad = c2 * (window_size-1)**2 + c1 * (window_size-1) + c0
        # Predict one sample ahead to reduce boundary lag even further
        pred_next = c2 * window_size**2 + c1 * window_size + c0  # Predictive extrapolation
        
        # 3. Weighted average using adaptive exponential weights
        # Adapt weight distribution based on signal activity/derivative
        # Higher activity -> more emphasis on recent samples
        activity = np.clip(np.abs(slope) / (current_noise + 1e-8), 0, 5)
        exp_scale = -1.0 - 1.5 * (activity / 5.0)  # -1.0 to -2.5: more negative = sharper recency bias
        adaptive_weights = np.exp(np.linspace(exp_scale, 0, window_size))
        adaptive_weights /= np.sum(adaptive_weights)
        weighted_ma = np.sum(window * adaptive_weights)
        
        # 4. Advanced false reversal penalty: suppress noise-induced slope direction changes
        # Apply significance test on slope vs noise; only penalize if direction changes without significance
        reversal_penalty_applied = False
        if len(slope_history) >= 4:
            recent_slopes = slope_history[-4:]
            slope_mean = np.mean(recent_slopes)
            
            if prev_slope != 0:
                current_dir = np.sign(slope)
                prev_dir = np.sign(prev_slope)
                
                if current_dir != prev_dir and current_dir != 0:
                    slope_magnitude = np.abs(slope)
                    noise_threshold = (current_noise / (window_size / 1.5)) * 2.0
                    recent_consensus = np.sign(slope_mean)
                    
                    recent_vals = val_history[-5:]
                    val_change = np.abs(recent_vals[-1] - np.mean(recent_vals[:-1])) if len(recent_vals) > 1 else 0
                    
                    if (slope_magnitude < noise_threshold or 
                        recent_consensus != current_dir or
                        val_change < current_noise * 0.8):
                        # Blend predicted values to maintain output consistency
                        adj_slope = 1.0 - min(1.0, slope_magnitude / (noise_threshold + 1e-10))
                        blend_penalty = 0.4 + 0.5 * adj_slope  # from 0.4 to 0.9
                        pred_next = pred_next * (1 - blend_penalty) + (prev_filtered + prev_slope) * blend_penalty
                        pred_end_quad = pred_end_quad * (1 - blend_penalty) + (prev_filtered) * blend_penalty
                        trend_confidence = max(0.1, trend_confidence * 0.4)
                        reversal_penalty_applied = True
                    else:
                        trend_confidence = min(1.0, trend_confidence + 0.25)
                else:
                    if not reversal_penalty_applied:
                        trend_confidence = min(1.0, trend_confidence + 0.08)
        
        # Also apply original style penalty for continuity
        t_centered = t - np.mean(t)
        slope_se = current_noise / np.sqrt(np.sum(t_centered**2))
        slope_snr = np.abs(slope) / (slope_se + 1e-10)
        
        # 5. Blend components: prediction for responsiveness, weighted MA for noise reduction
        # Pred_next is forward looking, reduces lag at the price of some smoothness
        # We adapt based on signal confidence: more prediction when SNR is high
        slope_snr_norm = np.clip(np.abs(slope) / (current_noise + 1e-8) / 3.0, 0.0, 1.0)
        # Polynomial endpoints are smoother than pure linear extrapolation
        blend_pred = 0.2 + 0.35 * slope_snr_norm + 0.15 * trend_confidence  # 0.2-0.7 weight on prediction
        blend_pred = min(0.7, blend_pred)
        candidate = (1 - blend_pred) * (0.6 * weighted_ma + 0.4 * pred_end_quad) + blend_pred * pred_next
        
        # 6. Enforce output continuity: clamp jumps beyond what trend predicts
        # Prevents discontinuities that don't match measured trend
        if prev_filtered is not None:
            expected_jump = np.abs(slope)
            actual_jump = np.abs(candidate - prev_filtered)
            jump_thresh = 2.5 * expected_jump + 2.0 * current_noise
            if actual_jump > jump_thresh:
                # Clamp but preserve sign of difference
                candidate = prev_filtered + jump_thresh * np.sign(candidate - prev_filtered)
        
        # Assign and update state
        y[i] = candidate
        prev_filtered = candidate
        prev_slope = slope

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