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

    # Efficient vectorized moving average
    cumsum = np.cumsum(np.insert(x, 0, 0))
    y = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    return y


def enhanced_filter_with_trend_preservation(x, window_size=20, alpha=0.148, penalty_weight=3.14, derivative_momentum=0.906):
    """
    Enhanced version with adaptive trend preservation, false reversal penalty,
    and polynomial predictive filtering to minimize lag and spurious reversals.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window
        alpha: Regularization parameter for trend fitting (0 < alpha < 1)
        penalty_weight: Weight for false reversal penalty (higher=more penalization)
        derivative_momentum: Smoothing factor for derivative (0-1, higher=more smoothing)

    Returns:
        y: Filtered output signal
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1")

    output_length = len(x) - window_size + 1
    if output_length <= 0:
        return np.array([], dtype=np.float64)
    y = np.zeros(output_length, dtype=np.float64)

    # Base exponential weights - optimized for responsiveness vs noise balance
    base_weights = np.exp(np.linspace(-4.14, 0, window_size))
    base_weights = base_weights / np.sum(base_weights)

    # Precompute Vandermonde matrix for 2nd order polynomial fitting
    t = np.arange(window_size)
    A = np.vander(t, 3, increasing=True)
    W_sqrt = np.sqrt(base_weights)
    A_weighted = A * W_sqrt[:, np.newaxis]

    # Precompute weighted least squares components
    AtA = A_weighted.T @ A_weighted + alpha * np.eye(3)
    AtA_inv = np.linalg.inv(AtA)
    AtW = AtA_inv @ (A_weighted.T * W_sqrt)

    # State variables
    prev_filtered = 0.0
    prev_derivative = 0.0

    # Consistency tracking
    consistency_avg = 0.5
    consistency_momentum = 0.954

    # Multi-window consistency parameters
    short_window = min(5, window_size // 2)
    consistency_long = 0.5

    # Volatility estimation window
    vol_window = min(4, window_size)

    first_window = True
    eps = np.finfo(np.float64).eps

    # Adaptation ramp for smoother initial response
    adaptation_counter = 0
    adaptation_ramp = min(25, output_length // 12) if output_length > 10 else 5

    # Lag compensation for polynomial prediction - improved tracking
    lag_compensation = 0.73

    # Volatility adaptive lag compensation
    min_lag_compensation = 0.38
    max_lag_compensation = 0.96

    # Derivative history for historical reversal validation
    derivative_history = []
    derivative_history_length = 6
    avg_derivative_magnitude = 0.01

    for i in range(output_length):
        window = x[i : i + window_size]

        # Step 1: Weighted least squares polynomial fit for trend estimation
        coeffs = AtW @ window

        # Step 2: Volatility-adaptive lag compensation
        window_diffs = np.diff(window)
        if len(window_diffs) >= vol_window:
            volatility = np.mean(np.abs(window_diffs[-vol_window:]))
        else:
            volatility = np.mean(np.abs(window_diffs)) if len(window_diffs) > 0 else eps
        
        # Normalize volatility to 0-1 range for adaptive compensation
        vol_normalized = np.clip(volatility / 0.46, 0, 1)
        # More lag compensation in high volatility to track fast changes
        adaptive_lag = min_lag_compensation + vol_normalized * (max_lag_compensation - min_lag_compensation)
        effective_lag = lag_compensation * 0.56 + adaptive_lag * 0.44
        
        pred_point = window_size - 1 + effective_lag
        trend_value = coeffs[0] + coeffs[1] * pred_point + coeffs[2] * pred_point ** 2

        # Step 3: Weighted moving average for noise reduction
        smoothed_value = np.sum(window * base_weights)

        # Step 4: Trend strength and volatility detection (volatility already computed)
        recent_trend = np.mean(window_diffs[-min(4, window_size) :]) if window_size >= 2 else 0.0
        trend_consistency = np.abs(recent_trend) / (volatility + eps)

        # Multi-window consistency: short vs medium trend agreement
        if len(window_diffs) >= short_window:
            short_trend = np.mean(window_diffs[-short_window:])
            medium_trend = np.mean(window_diffs[-2 * short_window :]) if len(window_diffs) >= 2 * short_window else recent_trend
            short_sign = np.sign(short_trend)
            medium_sign = np.sign(medium_trend)
            cross_consistency = 1.0 if short_sign == medium_sign else (1.0 - 0.758 * abs(short_sign - medium_sign) / 2.0)
        else:
            cross_consistency = 1.0

        # Combined consistency measure with optimized weights
        combined_consistency = trend_consistency * (0.623 + 0.377 * cross_consistency)

        # Update running consistency averages with bounded stabilization
        consistency_avg = consistency_momentum * consistency_avg + (1 - consistency_momentum) * combined_consistency
        consistency_long = 0.948 * consistency_long + 0.052 * combined_consistency
        consistency_avg = 0.860 * consistency_avg + 0.070
        consistency_long = 0.860 * consistency_long + 0.070

        # Step 5: Adaptive blend with volatility-adaptive coefficients for lag/noise balance
        # Reduce prediction weight in low volatility, increase in high volatility
        vol_weight = np.clip(volatility / 0.3, 0, 0.16)
        base_pred_weight = 0.566 * consistency_avg + 0.268 * consistency_long + vol_weight
        prediction_weight = min(0.934, max(0.134, base_pred_weight))
        smoothing_weight = 1.0 - prediction_weight
        candidate_value = smoothing_weight * smoothed_value + prediction_weight * trend_value

        # Step 6: False reversal penalty with tuned threshold and volatility adaptation
        if not first_window:
            current_derivative = candidate_value - prev_filtered
            
            # Update derivative history for historical validation
            derivative_history.append(current_derivative)
            if len(derivative_history) > derivative_history_length:
                derivative_history.pop(0)
            avg_derivative_magnitude = 0.92 * avg_derivative_magnitude + 0.08 * abs(current_derivative)
            
            # Historical consensus check - helps distinguish genuine reversals from noise
            if len(derivative_history) >= 4:
                recent_signs = np.sign(derivative_history[-4:])
                sign_consensus = np.sum(recent_signs == recent_signs[-1])
                historical_consensus = sign_consensus / 4.0
            else:
                historical_consensus = 0.75
            
            # Higher consistency threshold in high volatility for more aggressive reversal protection
            threshold_consistency = 1.76 - 0.12 * vol_normalized
            if (
                prev_derivative != 0
                and np.sign(current_derivative) != np.sign(prev_derivative)
                and combined_consistency < threshold_consistency
                and historical_consensus < 0.62
            ):
                reversal_penalty = (1.39 - min(1.0, combined_consistency)) / penalty_weight
                # Make penalty inversely proportional to how big the reversal is
                deriv_ratio = abs(current_derivative) / (avg_derivative_magnitude + eps)
                reversal_penalty *= min(1.0, max(0.28, 1.0 - deriv_ratio * 0.45))
                if adaptation_counter < adaptation_ramp:
                    ramp_factor = adaptation_counter / adaptation_ramp
                    reversal_penalty *= ramp_factor * ramp_factor
                candidate_value = prev_filtered + current_derivative * (1.0 - reversal_penalty)
                current_derivative = candidate_value - prev_filtered
                if derivative_history:
                    derivative_history[-1] = current_derivative

            # Derivative momentum with ramp-up and volatility adaptation
            if adaptation_counter < adaptation_ramp:
                ramp_factor = adaptation_counter / adaptation_ramp
                effective_momentum = 0.540 + (derivative_momentum - 0.540) * ramp_factor
            else:
                # Lower momentum in high volatility for faster response to real changes
                effective_momentum = derivative_momentum * (1 - 0.12 * vol_normalized)
            prev_derivative = effective_momentum * prev_derivative + (1 - effective_momentum) * current_derivative
        else:
            prev_derivative = candidate_value - prev_filtered
            derivative_history.append(prev_derivative)
            first_window = False

        y[i] = candidate_value
        prev_filtered = candidate_value
        adaptation_counter += 1

    return y


def process_signal(input_signal, window_size=20, algorithm_type="enhanced", alpha=0.148, penalty_weight=3.14, derivative_momentum=0.906):
    """
    Main signal processing function that applies the selected algorithm.

    Args:
        input_signal: Input time series data
        window_size: Window size for processing
        algorithm_type: Type of algorithm to use ("basic" or "enhanced")
        alpha: Regularization parameter for enhanced algorithm
        penalty_weight: Weight for false reversal penalty
        derivative_momentum: Smoothing factor for derivative

    Returns:
        Filtered signal
    """
    if algorithm_type == "enhanced":
        return enhanced_filter_with_trend_preservation(input_signal, window_size, alpha, penalty_weight, derivative_momentum)
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


def run_signal_processing(noisy_signal=None, signal_length=1000, noise_level=0.3, window_size=20, alpha=0.148, penalty_weight=3.14, derivative_momentum=0.906):
    """
    Run the signal processing algorithm on a test signal.

    Args:
        noisy_signal: Input signal to filter (if provided, use this; otherwise generate)
        signal_length: Length if generating signal (for backward compatibility)
        noise_level: Noise level if generating signal (for backward compatibility)
        window_size: Window size for processing
        alpha: Regularization parameter for enhanced algorithm
        penalty_weight: Weight for false reversal penalty
        derivative_momentum: Smoothing factor for derivative

    Returns:
        Dictionary containing results and metrics
    """
    if noisy_signal is not None:
        filtered_signal = process_signal(noisy_signal, window_size, "enhanced", alpha, penalty_weight, derivative_momentum)
        clean_signal = None
    else:
        noisy_signal, clean_signal = generate_test_signal(signal_length, noise_level)
        filtered_signal = process_signal(noisy_signal, window_size, "enhanced", alpha, penalty_weight, derivative_momentum)

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

        # Lag error estimate (cross-correlation peak offset)
        if min_length > window_size:
            xcorr = np.correlate(filtered_signal - np.mean(filtered_signal), aligned_clean - np.mean(aligned_clean), mode="full")
            lag_peak = np.argmax(xcorr) - (len(filtered_signal) - 1)
            lag_error = abs(lag_peak)
        else:
            lag_error = 0

        # Spurious reversals metric (filtered sign changes minus clean sign changes)
        filtered_slopes = np.diff(filtered_signal)
        clean_slopes = np.diff(aligned_clean)
        if len(filtered_slopes) > 2:
            filtered_sign_changes = np.sum(np.abs(np.diff(np.sign(filtered_slopes)))) // 2
            clean_sign_changes = np.sum(np.abs(np.diff(np.sign(clean_slopes)))) // 2
            spurious_reversals = max(0, filtered_sign_changes - clean_sign_changes)
        else:
            spurious_reversals = 0

        return {
            "filtered_signal": filtered_signal,
            "clean_signal": aligned_clean,
            "noisy_signal": aligned_noisy,
            "correlation": correlation,
            "noise_reduction": noise_reduction,
            "signal_length": min_length,
            "lag_error": lag_error,
            "spurious_reversals": spurious_reversals,
        }
    elif len(filtered_signal) > 0:
        return {
            "filtered_signal": filtered_signal,
            "clean_signal": None,
            "noisy_signal": None,
            "correlation": 0,
            "noise_reduction": 0,
            "signal_length": len(filtered_signal),
            "lag_error": 0,
            "spurious_reversals": 0,
        }
    else:
        return {
            "filtered_signal": [],
            "clean_signal": [],
            "noisy_signal": [],
            "correlation": 0,
            "noise_reduction": 0,
            "signal_length": 0,
            "lag_error": 0,
            "spurious_reversals": 0,
        }


if __name__ == "__main__":
    results = run_signal_processing()
    print("Signal processing completed!")
    print(f"Correlation with clean signal: {results['correlation']:.3f}")
    print(f"Noise reduction: {results['noise_reduction']:.3f}")
    print(f"Processed signal length: {results['signal_length']}")
    print(f"Lag error estimate: {results['lag_error']}")
    print(f"Spurious reversals estimate: {results['spurious_reversals']}")