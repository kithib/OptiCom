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

    x = np.asarray(x, dtype=np.float64)
    output_length = len(x) - window_size + 1
    y = np.zeros(output_length, dtype=np.float64)

    for i in range(output_length):
        window = x[i : i + window_size]
        y[i] = np.mean(window)

    return y


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Enhanced version with trend preservation using adaptive filtering with
    polynomial fitting, robust trend detection, predictive lag compensation,
    and advanced hysteresis reversal suppression for multi-objective optimization.

    Implements:
    1. Slope change minimization - reduces spurious directional reversals with adaptive hysteresis
    2. Lag error minimization - maintains responsiveness with optimized predictive polynomial extrapolation
    3. Tracking accuracy - preserves genuine signal trends with SNR-adaptive weighted robust fitting
    4. False reversal penalty - avoids noise-induced trend changes with multi-level statistical gating

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
    y = np.zeros(output_length, dtype=np.float64)
    
    half_window = window_size // 2
    
    # Optimized exponential weights - balances recency and smoothness
    exp_weights = np.exp(np.linspace(-7.5, 0, window_size))
    exp_weights = exp_weights / np.sum(exp_weights)
    
    # Uniform weights for maximum smoothing in noise-dominated regions
    uniform_weights = np.ones(window_size) / window_size
    
    # Pre-compute polynomial basis
    if window_size >= 6:
        t_window = np.arange(window_size)

    for i in range(output_length):
        window = x[i : i + window_size]
        
        # 1. Advanced multi-scale trend detection with optimized windows
        if window_size >= 7:
            # Short-term trend detection (optimized 6-sample for sensitivity)
            recent = window[-6:]
            recent_grads = np.diff(recent)
            short_trend = np.mean(recent_grads)
            short_trend_mag = np.abs(short_trend)
            short_noise = np.median(np.abs(recent_grads - np.median(recent_grads))) + 1e-10
            
            # Long-term trend consistency analysis
            long_recent = window[-half_window:] if half_window > 0 else window
            long_grads = np.diff(long_recent)
            long_trend = np.mean(long_grads)
            trend_consistency = 1.0 - np.abs(short_trend - long_trend) / (short_trend_mag + np.abs(long_trend) + 1e-10)
            trend_consistency = max(0.0, min(1.0, trend_consistency))
        else:
            recent = window
            recent_grads = np.diff(recent)
            short_trend = np.mean(recent_grads)
            short_trend_mag = np.abs(short_trend)
            short_noise = np.std(recent_grads) + 1e-10
            trend_consistency = 1.0
        
        # 2. Robust edge detection with MAD statistics
        recent_median = np.median(window[-half_window:]) if half_window > 0 else window[-1]
        old_median = np.median(window[:half_window]) if half_window > 0 else window[0]
        
        signal_change = np.abs(recent_median - old_median)
        noise_level = np.median(np.abs(np.diff(window))) + 1e-10
        
        # Combined SNR measures with trend consistency weighting
        snr_change = signal_change / noise_level
        snr_trend = short_trend_mag / short_noise
        snr_combined = max(snr_change, snr_trend * trend_consistency)
        
        # 3. Precision adaptive weight blending (further calibrated)
        if snr_combined > 0.80:
            # Signal-dominant: strongly emphasize recent samples for minimal lag
            alpha = min(0.98, 0.81 + 0.10 * min(snr_combined, 2.4))
            base_weights = alpha * exp_weights + (1 - alpha) * uniform_weights
        else:
            # Noise-dominant: optimized smoothing blend
            alpha = max(0.23, 0.71 - 0.28 * (1.0 - snr_combined))
            base_weights = alpha * exp_weights + (1 - alpha) * uniform_weights
        
        # 4. Advanced robust polynomial fitting with refined outlier rejection
        pred_correction = 0.0
        pred_gain = 0.0
        
        if window_size >= 6:
            try:
                order = 2 if window_size >= 12 else 1
                coeffs = np.polyfit(t_window, window, order)
                poly_vals = np.polyval(coeffs, t_window)
                
                # Residual-based robust weighting with optimized exponent
                residual = np.abs(window - poly_vals)
                med_resid = np.median(residual) + 1e-10
                robust_weights = np.exp(-(residual / med_resid) ** 1.7)
                final_weights = base_weights * robust_weights
                sum_fw = np.sum(final_weights)
                if sum_fw > 1e-12:
                    final_weights = final_weights / sum_fw
                else:
                    final_weights = base_weights / np.sum(base_weights)
                
                # Precision lag compensation with greatly refined parameters
                if order >= 1 and snr_combined > 0.35:
                    slope = coeffs[0]
                    pred_correction = slope * (window_size * 0.58)
                    pred_gain = 0.32 * min(1.0, snr_combined / 2.0)
            except (np.linalg.LinAlgError, ValueError, TypeError, FloatingPointError):
                sum_bw = np.sum(base_weights)
                if sum_bw > 1e-12:
                    final_weights = base_weights / sum_bw
                else:
                    final_weights = uniform_weights
        else:
            sum_bw = np.sum(base_weights)
            if sum_bw > 1e-12:
                final_weights = base_weights / sum_bw
            else:
                final_weights = uniform_weights
        
        # 5. Weighted filter output with predictive lag compensation
        filtered_val = np.sum(window * final_weights)
        filtered_val = filtered_val + pred_gain * pred_correction
        
        # 6. Highly advanced directional consistency with precision hysteresis gating
        if i >= 2:
            prev_slope = y[i-1] - y[i-2]
            current_slope = filtered_val - y[i-1]
            
            # Detect and penalize spurious directional reversals
            if prev_slope * current_slope < -1e-12:
                reversal_mag = abs(current_slope)
                trend_mag = abs(prev_slope)
                
                # Reversal confidence calculation
                reversal_confidence = min(2.0, reversal_mag / (trend_mag + 1e-8))
                
                # Precision-calibrated penalty strengths across SNR regions
                if snr_combined < 0.55:
                    # Very strong penalty for low-SNR region reversals (noise likely)
                    penalty_strength = 0.58 + 0.41 * (1.0 - reversal_confidence * 0.42)
                elif snr_combined < 0.95:
                    # Advanced moderate penalty for medium-SNR region reversals
                    penalty_strength = 0.38 + 0.21 * (1.0 - reversal_confidence * 0.52)
                else:
                    # Near-zero penalty in high-SNR regions (genuine reversals almost certain)
                    penalty_strength = 0.008 + 0.038 * (1.0 - reversal_confidence * 0.86)
                
                penalty_strength = max(0.0, min(1.0, penalty_strength))
                filtered_val = y[i-1] + (1.0 - penalty_strength) * current_slope
        
        y[i] = filtered_val

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

    clean_signal = (
        2 * np.sin(2 * np.pi * 0.5 * t)
        + 1.5 * np.sin(2 * np.pi * 2 * t)
        + 0.5 * np.sin(2 * np.pi * 5 * t)
        + 0.8 * np.exp(-t / 5) * np.sin(2 * np.pi * 1.5 * t)
    )

    trend = 0.1 * t * np.sin(0.2 * t)
    clean_signal += trend

    random_walk = np.cumsum(np.random.randn(length) * 0.05)
    clean_signal += random_walk

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
    if noisy_signal is not None:
        filtered_signal = process_signal(noisy_signal, window_size, "enhanced")
        clean_signal = None
    else:
        noisy_signal, clean_signal = generate_test_signal(signal_length, noise_level)
        filtered_signal = process_signal(noisy_signal, window_size, "enhanced")

    if len(filtered_signal) > 0 and clean_signal is not None:
        delay = window_size - 1
        aligned_clean = clean_signal[delay:]
        aligned_noisy = noisy_signal[delay:]

        min_length = min(len(filtered_signal), len(aligned_clean))
        filtered_signal = filtered_signal[:min_length]
        aligned_clean = aligned_clean[:min_length]
        aligned_noisy = aligned_noisy[:min_length]

        correlation = np.corrcoef(filtered_signal, aligned_clean)[0, 1] if min_length > 1 else 0

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
    results = run_signal_processing()
    print("Signal processing completed!")
    print(f"Correlation with clean signal: {results['correlation']:.3f}")
    print(f"Noise reduction: {results['noise_reduction']:.3f}")
    print(f"Processed signal length: {results['signal_length']}")