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

    # Vectorized moving average using stride tricks for better performance
    output_length = len(x) - window_size + 1
    y = np.convolve(x, np.ones(window_size) / window_size, mode='valid')
    return y


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Enhanced version with trend preservation and adaptive filtering.
    Combines Savitzky-Golay polynomial fitting with adaptive noise estimation
    and false reversal penalty to preserve genuine trends while minimizing noise.

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

    # Precompute Savitzky-Golay polynomial fitting weights (2nd order polynomial)
    # This preserves slope and curvature while denoising
    order = min(2, window_size - 1)
    t = np.arange(window_size) - (window_size - 1) / 2
    X = np.vander(t, order + 1)
    sg_weights = X @ np.linalg.inv(X.T @ X) @ X.T
    sg_weights = sg_weights[window_size // 2]  # Central point weight for smoothing
    sg_weights = sg_weights / np.sum(sg_weights)  # Normalize

    # Initial noise estimation from early samples
    early_diff = np.diff(x[:min(window_size * 3, len(x))])
    noise_est = np.median(np.abs(early_diff)) / 0.6745 if len(early_diff) > 0 else 0.1
    noise_est = max(noise_est, 1e-8)  # Avoid division by zero

    for i in range(output_length):
        window = x[i : i + window_size]

        # Calculate window statistics for adaptation
        window_mean = np.mean(window)
        window_std = np.std(window)
        signal_to_noise = window_std / noise_est if noise_est > 0 else 10

        # Detect trend changes using normalized slope
        t_window = np.arange(window_size)
        window_slope = np.corrcoef(t_window, window)[0, 1] if window_std > 0 else 0

        # Trend strength determines adaptation (higher = more responsive)
        trend_strength = min(abs(window_slope) * 3, 1.0)

        # Combine SG (trend-preserving) with exponential (responsive) weights
        exp_weights = np.exp(np.linspace(-2 * (1 - trend_strength) - 1, 0, window_size))

        # Adapt based on SNR: high SNR = trust recent, low SNR = trust polynomial fit
        if signal_to_noise < 1.5:
            # Low SNR: favor Savitzky-Golay polynomial fit for noise reduction
            combined_weights = 0.7 * sg_weights + 0.3 * exp_weights
        else:
            # High SNR: favor responsiveness with exponential weights
            combined_weights = 0.3 * sg_weights + 0.7 * exp_weights

        combined_weights = combined_weights / np.sum(combined_weights)

        # Apply filter
        raw_value = np.sum(window * combined_weights)

        # False reversal penalty: avoid changing direction from noise alone
        if i > 0:
            prev_slope = y[i-1] - y[max(0, i-5)]
            curr_slope = raw_value - y[i-1]

            # Penalize direction reversals smaller than noise level (likely spurious)
            if (prev_slope * curr_slope < 0) and (abs(curr_slope) < noise_est * 0.5):
                y[i] = y[i-1] + np.sign(prev_slope) * 0.1 * abs(curr_slope)
            else:
                y[i] = raw_value
        else:
            y[i] = raw_value

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