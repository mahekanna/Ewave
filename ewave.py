import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class WaveType(Enum):
    """Types of Elliott Waves."""
    IMPULSE = "impulse"
    CORRECTIVE = "corrective"

class WaveDegree(Enum):
    """Degrees of Elliott Waves from largest to smallest."""
    GRAND_SUPERCYCLE = "Grand Supercycle"
    SUPERCYCLE = "Supercycle"
    CYCLE = "Cycle"
    PRIMARY = "Primary"
    INTERMEDIATE = "Intermediate"
    MINOR = "Minor"
    MINUTE = "Minute"
    MINUETTE = "Minuette"
    SUBMINUETTE = "Subminuette"

@dataclass
class WavePoint:
    """Data structure for a single point in a wave."""
    index: int              # Position in the dataset
    date: datetime          # Timestamp of the point
    price: float            # Price at this point
    wave_degree: WaveDegree = WaveDegree.INTERMEDIATE  # Wave degree

@dataclass
class Wave:
    """Data structure for a complete wave."""
    start: WavePoint                 # Starting point of the wave
    end: WavePoint                   # Ending point of the wave
    wave_num: str                    # Wave number (1,2,3,4,5,A,B,C,etc.)
    wave_type: WaveType              # 'impulse' or 'corrective'
    subwaves: List['Wave'] = None    # Subwaves (waves of lower degree)

    @property
    def length(self) -> float:
        """Length of the wave in price terms."""
        return abs(self.end.price - self.start.price)

    @property
    def duration(self) -> int:
        """Duration of the wave in bars."""
        return self.end.index - self.start.index

    @property
    def is_up(self) -> bool:
        """Whether the wave is an upward or downward wave."""
        return self.end.price > self.start.price

    @property
    def slope(self) -> float:
        """Slope/rate of change of the wave."""
        if self.duration == 0:
            return 0
        return (self.end.price - self.start.price) / self.duration

    @property
    def is_extended(self) -> bool:
        """Whether this wave is extended compared to typical waves."""
        # To be implemented based on comparison to related waves
        return False


class CycleAnalyzer:
    """
    Analyzes price data to detect dominant cycles and patterns using FFT.
    """

    def __init__(self):
        """Initialize the cycle analyzer."""
        self.cycles = {}
        self.dominant_cycles = []

    def detect_cycles(self, data: pd.Series, n_cycles: int = 5, min_period: int = 5, max_period: Optional[int] = None):
        """
        Detect dominant cycles in price data using Fast Fourier Transform.

        Parameters:
            data: Price time series
            n_cycles: Number of dominant cycles to identify
            min_period: Minimum cycle period to consider (in data points)
            max_period: Maximum cycle period to consider (in data points)

        Returns:
            List of dominant cycle periods
        """
        # Detrend the data to better isolate cycles
        detrended_data = signal.detrend(data)

        # Calculate FFT
        N = len(detrended_data)
        yf = fft(detrended_data)
        xf = fftfreq(N, 1)

        # Get positive frequencies up to Nyquist frequency
        positive_freq_idx = np.arange(1, N // 2)

        # Get amplitude spectrum
        amplitude = 2.0/N * np.abs(yf[positive_freq_idx])

        # Calculate periods from frequencies
        periods = 1.0 / xf[positive_freq_idx]

        # Apply period limits
        if max_period is None:
            max_period = N // 2

        period_mask = (periods >= min_period) & (periods <= max_period)
        periods = periods[period_mask]
        amplitude = amplitude[period_mask]

        # Find dominant cycles
        dominant_indices = np.argsort(amplitude)[-n_cycles:][::-1]  # Get indices of n_cycles highest amplitudes
        dominant_periods = periods[dominant_indices]
        dominant_amplitudes = amplitude[dominant_indices]

        # Store results
        self.cycles = {
            'periods': periods,
            'amplitude': amplitude,
            'dominant_periods': dominant_periods,
            'dominant_amplitudes': dominant_amplitudes,
        }

        # Round periods to nearest integer for practical use
        self.dominant_cycles = [int(round(p)) for p in dominant_periods]

        return self.dominant_cycles

    def plot_cycles(self, data: pd.Series, title: str = "Cycle Analysis"):
        """
        Plot the original data with detected cycles.

        Parameters:
            data: Original price data
            title: Plot title

        Returns:
            matplotlib.figure.Figure
        """
        if not self.cycles:
            raise ValueError("No cycles detected yet. Call detect_cycles first.")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # Plot original data
        ax1.plot(data.index, data.values, label='Original Data')
        ax1.set_title(f"{title} - Original Data")
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Price')
        ax1.grid(True)

        # Plot amplitude spectrum
        ax2.bar(self.cycles['periods'], self.cycles['amplitude'], alpha=0.7)
        for period, amplitude in zip(self.cycles['dominant_periods'], self.cycles['dominant_amplitudes']):
            ax2.annotate(f'{int(round(period))}',
                         xy=(period, amplitude),
                         xytext=(0, 10),
                         textcoords='offset points',
                         ha='center')

        ax2.set_title('Cycle Amplitude Spectrum')
        ax2.set_xlabel('Period (bars)')
        ax2.set_ylabel('Amplitude')
        ax2.grid(True)

        plt.tight_layout()
        return fig

    def reconstruct_cycle(self, data: pd.Series, period: int):
        """
        Reconstruct a specific cycle component from the data.

        Parameters:
            data: Original price data
            period: Cycle period to reconstruct

        Returns:
            np.ndarray: Reconstructed cycle component
        """
        # Detrend the data
        detrended_data = signal.detrend(data)

        # Calculate FFT
        N = len(detrended_data)
        yf = fft(detrended_data)
        xf = fftfreq(N, 1)

        # Get periods
        periods = 1.0 / np.abs(xf[1:N//2])

        # Find the closest period in FFT result
        closest_idx = np.argmin(np.abs(periods - period)) + 1  # +1 because we excluded the 0th element

        # Create a mask that isolates just this frequency and its conjugate
        mask = np.zeros(N, dtype=complex)
        mask[closest_idx] = yf[closest_idx]
        mask[N - closest_idx] = yf[N - closest_idx]  # Include the conjugate frequency

        # Reconstruct the cycle
        reconstructed = ifft(mask).real

        # Add back the trend
        trend = data.values - detrended_data
        reconstructed_with_trend = reconstructed + trend

        return reconstructed_with_trend


class WaveDetector:
    """
    Detects Elliott Wave patterns using cyclic analysis for improved accuracy.
    """

    def __init__(self):
        """Initialize the wave detector."""
        self.cycle_analyzer = CycleAnalyzer()
        self.fibonacci_ratios = {
            "0": 0.0,
            "0.236": 0.236,
            "0.382": 0.382,
            "0.5": 0.5,
            "0.618": 0.618,
            "0.786": 0.786,
            "0.886": 0.886,
            "1.0": 1.0,
            "1.272": 1.272,
            "1.414": 1.414,
            "1.618": 1.618,
            "2.0": 2.0,
            "2.618": 2.618,
            "3.618": 3.618,
            "4.236": 4.236
        }

    def detect_pivots_with_fft(self, data: pd.Series, timeframe: str = 'daily',
                              method: str = 'cycle_extrema', **kwargs):
        """
        Detect market pivots (highs and lows) using FFT cycle analysis.

        Parameters:
            data: Price series
            timeframe: Chart timeframe ('daily', 'hourly', '15min', etc.)
            method: Method to use ('cycle_extrema', 'adaptive_threshold', 'cycle_projection')
            **kwargs: Additional parameters for specific methods

        Returns:
            Tuple of (pivot_highs, pivot_lows) indices
        """
        # Detect dominant cycles in the data
        dominant_cycles = self.cycle_analyzer.detect_cycles(
            data,
            n_cycles=kwargs.get('n_cycles', 3),
            min_period=kwargs.get('min_period', 5),
            max_period=kwargs.get('max_period', None)
        )

        if method == 'cycle_extrema':
            return self._detect_pivots_cycle_extrema(data, dominant_cycles, **kwargs)
        elif method == 'adaptive_threshold':
            return self._detect_pivots_adaptive_threshold(data, dominant_cycles, **kwargs)
        elif method == 'cycle_projection':
            return self._detect_pivots_cycle_projection(data, dominant_cycles, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _detect_pivots_cycle_extrema(self, data: pd.Series, cycles: List[int],
                                   window_factor: float = 0.5, min_strength: float = 0.5, **kwargs):
        """
        Detect pivots by finding extrema in reconstructed cycles.

        Parameters:
            data: Price series
            cycles: List of dominant cycle periods
            window_factor: Window size as a factor of cycle period
            min_strength: Minimum relative strength to consider a pivot significant

        Returns:
            Tuple of (pivot_highs, pivot_lows) indices
        """
        pivot_highs = []
        pivot_lows = []
        cycle_signals = {}

        # Analyze each dominant cycle
        for cycle_period in cycles:
            # Reconstruct the cycle component
            cycle_data = self.cycle_analyzer.reconstruct_cycle(data, cycle_period)
            cycle_signals[cycle_period] = cycle_data

            # Define window size based on cycle period
            window = max(2, int(cycle_period * window_factor))

            # Find local maxima and minima in the cycle
            for i in range(window, len(data) - window):
                # Check for local maximum (potential pivot high)
                is_max = all(cycle_data[i] > cycle_data[i-j] for j in range(1, window+1)) and \
                         all(cycle_data[i] > cycle_data[i+j] for j in range(1, window+1))

                # Check for local minimum (potential pivot low)
                is_min = all(cycle_data[i] < cycle_data[i-j] for j in range(1, window+1)) and \
                         all(cycle_data[i] < cycle_data[i+j] for j in range(1, window+1))

                # Calculate pivot strength (how pronounced the peak/trough is)
                if is_max:
                    strength = min(
                        min(cycle_data[i] - cycle_data[i-j] for j in range(1, window+1)),
                        min(cycle_data[i] - cycle_data[i+j] for j in range(1, window+1))
                    ) / cycle_data[i]

                    if strength >= min_strength and i not in pivot_highs:
                        pivot_highs.append(i)

                elif is_min:
                    strength = min(
                        min(cycle_data[i-j] - cycle_data[i] for j in range(1, window+1)),
                        min(cycle_data[i+j] - cycle_data[i] for j in range(1, window+1))
                    ) / abs(cycle_data[i]) if cycle_data[i] != 0 else 0

                    if strength >= min_strength and i not in pivot_lows:
                        pivot_lows.append(i)

        # Deduplicate close pivots
        pivot_highs = self._deduplicate_pivots(data, pivot_highs, is_high=True)
        pivot_lows = self._deduplicate_pivots(data, pivot_lows, is_high=False)

        # Sort pivots
        pivot_highs.sort()
        pivot_lows.sort()

        return pivot_highs, pivot_lows

    def _detect_pivots_adaptive_threshold(self, data: pd.Series, cycles: List[int],
                                       threshold_factor: float = 0.5, **kwargs):
        """
        Detect pivots using an adaptive threshold based on cycle analysis.

        Parameters:
            data: Price series
            cycles: List of dominant cycle periods
            threshold_factor: Threshold factor for pivot detection

        Returns:
            Tuple of (pivot_highs, pivot_lows) indices
        """
        pivot_highs = []
        pivot_lows = []

        # Determine the most appropriate window size based on dominant cycles
        if cycles:
            window = int(np.median(cycles) / 2)
        else:
            window = 10  # Default window

        # Calculate volatility for adaptive threshold
        volatility = data.rolling(window=window).std().fillna(0)

        # Normalize price data
        normalized_data = (data - data.mean()) / data.std()

        # Detect pivots based on adaptive threshold
        for i in range(window, len(data) - window):
            # Adaptive threshold based on local volatility
            local_threshold = volatility.iloc[i] * threshold_factor

            # Check for pivot high
            if all(normalized_data.iloc[i] > normalized_data.iloc[i-j] + local_threshold for j in range(1, window+1)) and \
               all(normalized_data.iloc[i] > normalized_data.iloc[i+j] + local_threshold for j in range(1, window+1)):
                pivot_highs.append(i)

            # Check for pivot low
            elif all(normalized_data.iloc[i] < normalized_data.iloc[i-j] - local_threshold for j in range(1, window+1)) and \
                 all(normalized_data.iloc[i] < normalized_data.iloc[i+j] - local_threshold for j in range(1, window+1)):
                pivot_lows.append(i)

        return pivot_highs, pivot_lows

    def _detect_pivots_cycle_projection(self, data: pd.Series, cycles: List[int], **kwargs):
        """
        Detect pivots by projecting cycle turning points.

        Parameters:
            data: Price series
            cycles: List of dominant cycle periods

        Returns:
            Tuple of (pivot_highs, pivot_lows) indices
        """
        pivot_highs = []
        pivot_lows = []

        # Use the most dominant cycle
        if not cycles:
            return pivot_highs, pivot_lows

        primary_cycle = cycles[0]

        # Reconstruct the primary cycle
        cycle_data = self.cycle_analyzer.reconstruct_cycle(data, primary_cycle)

        # Find zero-crossings to identify potential turning points
        zero_crossings = np.where(np.diff(np.signbit(np.diff(cycle_data))))[0]

        # Classify as highs or lows
        for i in zero_crossings:
            if i > 0 and i < len(data) - 1:
                if data.iloc[i] > data.iloc[i-1] and data.iloc[i] > data.iloc[i+1]:
                    pivot_highs.append(i)
                elif data.iloc[i] < data.iloc[i-1] and data.iloc[i] < data.iloc[i+1]:
                    pivot_lows.append(i)

        return pivot_highs, pivot_lows

    def _deduplicate_pivots(self, data: pd.Series, pivots: List[int], is_high: bool,
                          min_distance: int = None):
        """
        Deduplicate pivots that are too close to each other.

        Parameters:
            data: Price series
            pivots: List of pivot indices
            is_high: Whether these are pivot highs (True) or lows (False)
            min_distance: Minimum distance between pivots

        Returns:
            List of deduplicated pivot indices
        """
        if not pivots:
            return []

        # If min_distance not specified, use average of dominant cycles / 4
        if min_distance is None and self.cycle_analyzer.dominant_cycles:
            min_distance = max(3, int(np.mean(self.cycle_analyzer.dominant_cycles) / 4))
        elif min_distance is None:
            min_distance = 5  # Default

        # Sort pivots by index
        sorted_pivots = sorted(pivots)

        # Group pivots that are close to each other
        groups = []
        current_group = [sorted_pivots[0]]

        for i in range(1, len(sorted_pivots)):
            if sorted_pivots[i] - current_group[-1] <= min_distance:
                current_group.append(sorted_pivots[i])
            else:
                groups.append(current_group)
                current_group = [sorted_pivots[i]]

        if current_group:
            groups.append(current_group)

        # For each group, select the best pivot
        deduped_pivots = []
        for group in groups:
            if is_high:
                # For pivot highs, select the highest price
                best_pivot = max(group, key=lambda i: data.iloc[i])
            else:
                # For pivot lows, select the lowest price
                best_pivot = min(group, key=lambda i: data.iloc[i])

            deduped_pivots.append(best_pivot)

        return deduped_pivots

    def create_wave_points(self, data: pd.DataFrame, highs: List[int], lows: List[int],
                         degree: WaveDegree = WaveDegree.INTERMEDIATE) -> List[WavePoint]:
        """
        Create WavePoint objects from pivot highs and lows.

        Parameters:
            data: Price data DataFrame
            highs: List of pivot high indices
            lows: List of pivot low indices
            degree: Wave degree

        Returns:
            List of WavePoint objects sorted by index
        """
        wave_points = []

        # Create WavePoint objects for highs
        for idx in highs:
            wave_points.append(WavePoint(
                index=idx,
                date=data.index[idx],
                price=data['close'].iloc[idx],
                wave_degree=degree
            ))

        # Create WavePoint objects for lows
        for idx in lows:
            wave_points.append(WavePoint(
                index=idx,
                date=data.index[idx],
                price=data['close'].iloc[idx],
                wave_degree=degree
            ))

        # Sort by index
        wave_points.sort(key=lambda x: x.index)

        return wave_points


class ElliottWaveCounter:
    """
    Core class for Elliott Wave pattern recognition and counting waves.
    Uses cycle analysis for improved accuracy.
    """

    def __init__(self, wave_detector: WaveDetector = None):
        """Initialize the Elliott Wave counter."""
        self.wave_detector = wave_detector or WaveDetector()

        # Default wave degree to use
        self.current_degree = WaveDegree.INTERMEDIATE

        # Elliott Wave rules and guidelines
        self.rules = {
            "wave2_retracement": "Wave 2 cannot retrace more than 100% of wave 1",
            "wave3_not_shortest": "Wave 3 is never the shortest among waves 1, 3, and 5",
            "wave4_no_overlap_wave1": "Wave 4 doesn't overlap with wave 1 territory (except in diagonals)",
            "wave5_beyond_wave3": "Wave 5 typically moves beyond the end of wave 3"
        }

        self.guidelines = {
            "wave3_extension": "Wave 3 is typically the longest and often extends (1.618 or more)",
            "wave2_wave4_alternation": "If wave 2 is sharp, wave 4 is usually flat, and vice versa",
            "wave5_equals_wave1": "Wave 5 often equals wave 1 when wave 3 is extended"
        }

    def identify_elliott_waves(self, data: pd.DataFrame, timeframe: str = 'daily', **kwargs):
        """
        Identify Elliott Wave patterns in price data.

        Parameters:
            data: Price data DataFrame with OHLC columns
            timeframe: Chart timeframe ('daily', 'hourly', '15min', etc.)
            **kwargs: Additional parameters for wave detection

        Returns:
            Tuple of (impulse_waves, corrective_waves, wave_points)
        """
        # Extract price series (closing prices)
        price = data['close']

        # Detect pivot points using FFT cycle analysis
        pivot_highs, pivot_lows = self.wave_detector.detect_pivots_with_fft(
            price,
            timeframe=timeframe,
            **kwargs
        )

        # Create wave points from pivots
        wave_points = self.wave_detector.create_wave_points(
            data,
            pivot_highs,
            pivot_lows,
            degree=self.current_degree
        )

        # Identify potential impulse and corrective waves
        impulse_waves = self._identify_impulse_waves(wave_points)
        corrective_waves = self._identify_corrective_waves(wave_points)

        # Apply Elliott Wave rules to validate and filter waves
        validated_impulse_waves = self._validate_impulse_waves(impulse_waves)
        validated_corrective_waves = self._validate_corrective_waves(corrective_waves)

        # Combine waves into complete wave sequences
        complete_waves = self._identify_complete_wave_sequences(
            validated_impulse_waves,
            validated_corrective_waves
        )

        return validated_impulse_waves, validated_corrective_waves, wave_points

    def _identify_impulse_waves(self, wave_points: List[WavePoint]) -> List[Wave]:
        """
        Identify potential 5-wave impulse patterns.

        Parameters:
            wave_points: List of potential wave points

        Returns:
            List of potential impulse waves
        """
        impulse_waves = []

        # Need at least 6 points to form a 5-wave sequence (start + 5 ends)
        if len(wave_points) < 6:
            return impulse_waves

        # Try different combinations of points to form a valid impulse
        for i in range(len(wave_points) - 5):
            # Skip if not enough points ahead
            if i + 5 >= len(wave_points):
                continue

            # Get potential wave points for a 5-wave sequence
            w1_start = wave_points[i]
            w1_end = wave_points[i+1]
            w2_end = wave_points[i+2]
            w3_end = wave_points[i+3]
            w4_end = wave_points[i+4]
            w5_end = wave_points[i+5]

            # Create candidate waves
            wave1 = Wave(start=w1_start, end=w1_end, wave_num="1", wave_type=WaveType.IMPULSE)
            wave2 = Wave(start=w1_end, end=w2_end, wave_num="2", wave_type=WaveType.CORRECTIVE)
            wave3 = Wave(start=w2_end, end=w3_end, wave_num="3", wave_type=WaveType.IMPULSE)
            wave4 = Wave(start=w3_end, end=w4_end, wave_num="4", wave_type=WaveType.CORRECTIVE)
            wave5 = Wave(start=w4_end, end=w5_end, wave_num="5", wave_type=WaveType.IMPULSE)

            # Check basic direction requirements before detailed validation
            if self._check_basic_impulse_directions(wave1, wave2, wave3, wave4, wave5):
                # Create the complete impulse wave
                impulse = Wave(
                    start=w1_start,
                    end=w5_end,
                    wave_num="12345",
                    wave_type=WaveType.IMPULSE,
                    subwaves=[wave1, wave2, wave3, wave4, wave5]
                )

                impulse_waves.append(impulse)

        return impulse_waves

    def _check_basic_impulse_directions(self, wave1, wave2, wave3, wave4, wave5):
        """
        Check if the directions of waves in an impulse are valid.

        Parameters:
            wave1, wave2, wave3, wave4, wave5: The five subwaves

        Returns:
            bool: True if directions are valid for an impulse
        """
        # Waves 1, 3, 5 should be in the same direction
        if not ((wave1.is_up and wave3.is_up and wave5.is_up) or
                (not wave1.is_up and not wave3.is_up and not wave5.is_up)):
            return False

        # Waves 2, 4 should be in the opposite direction to 1, 3, 5
        if not ((wave1.is_up and not wave2.is_up and not wave4.is_up) or
                (not wave1.is_up and wave2.is_up and wave4.is_up)):
            return False

        return True

    def _identify_corrective_waves(self, wave_points: List[WavePoint]) -> List[Wave]:
        """
        Identify potential 3-wave corrective patterns (A-B-C).

        Parameters:
            wave_points: List of potential wave points

        Returns:
            List of potential corrective waves
        """
        corrective_waves = []

        # Need at least 4 points to form a 3-wave sequence (start + 3 ends)
        if len(wave_points) < 4:
            return corrective_waves

        # Try different combinations of points
        for i in range(len(wave_points) - 3):
            # Skip if not enough points ahead
            if i + 3 >= len(wave_points):
                continue

            # Get potential wave points for a 3-wave sequence
            wA_start = wave_points[i]
            wA_end = wave_points[i+1]
            wB_end = wave_points[i+2]
            wC_end = wave_points[i+3]

            # Create candidate waves
            waveA = Wave(start=wA_start, end=wA_end, wave_num="A", wave_type=WaveType.CORRECTIVE)
            waveB = Wave(start=wA_end, end=wB_end, wave_num="B", wave_type=WaveType.CORRECTIVE)
            waveC = Wave(start=wB_end, end=wC_end, wave_num="C", wave_type=WaveType.CORRECTIVE)

            # Check basic direction requirements before detailed validation
            if self._check_basic_corrective_directions(waveA, waveB, waveC):
                # Create the complete corrective wave
                corrective = Wave(
                    start=wA_start,
                    end=wC_end,
                    wave_num="ABC",
                    wave_type=WaveType.CORRECTIVE,
                    subwaves=[waveA, waveB, waveC]
                )

                corrective_waves.append(corrective)

        return corrective_waves

    def _check_basic_corrective_directions(self, waveA, waveB, waveC):
        """
        Check if the directions of waves in a corrective pattern are valid.

        Parameters:
            waveA, waveB, waveC: The three subwaves

        Returns:
            bool: True if directions are valid for a corrective pattern
        """
        # Waves A and C should be in the same direction
        if (waveA.is_up != waveC.is_up):
            return False

        # Wave B should be in the opposite direction to A and C
        if (waveA.is_up == waveB.is_up):
            return False

        return True

def _validate_impulse_waves(self, impulse_waves: List[Wave]) -> List[Wave]:
    """
    Apply Elliott Wave rules to validate impulse waves.

    Parameters:
        impulse_waves: List of candidate impulse waves

    Returns:
        List of validated impulse waves
    """
    validated_waves = []

    for wave in impulse_waves:
        if not wave.subwaves or len(wave.subwaves) != 5:
            continue

        wave1, wave2, wave3, wave4, wave5 = wave.subwaves

        # Rule: Wave 2 cannot retrace more than 100% of wave 1
        if wave1.is_up:
            if wave2.end.price <= wave1.start.price:  # Wave 2 retraced too much
                continue
        else:  # Downward impulse
            if wave2.end.price >= wave1.start.price:  # Wave 2 retraced too much
                continue

        # Rule: Wave 3 is never the shortest among waves 1, 3, and 5
        wave1_length = wave1.length
        wave3_length = wave3.length
        wave5_length = wave5.length

        if wave3_length < wave1_length and wave3_length < wave5_length:
            continue

        # Rule: Wave 4 cannot overlap with wave 1's territory (except in diagonals)
        # Allow for diagonal patterns where wave 4 can overlap with wave 1
        is_diagonal = self._check_if_diagonal(wave)

        if not is_diagonal:
            if wave1.is_up:
                if wave4.end.price < wave1.end.price:  # Overlap in upward impulse
                    continue
            else:
                if wave4.end.price > wave1.end.price:  # Overlap in downward impulse
                    continue

        # Calculate Fibonacci relationships
        self._calculate_wave_relationships(wave)

        # Wave passes all rules
        validated_waves.append(wave)

    return validated_waves

def _check_if_diagonal(self, wave: Wave) -> bool:
    """
    Check if an impulse wave has the characteristics of a diagonal pattern.

    Parameters:
        wave: The impulse wave to check

    Returns:
        bool: True if the wave appears to be a diagonal
    """
    if not wave.subwaves or len(wave.subwaves) != 5:
        return False

    wave1, wave2, wave3, wave4, wave5 = wave.subwaves

    # In a diagonal:
    # 1. Wave lengths typically decrease
    decreasing_length = (wave3.length <= wave1.length and
                        wave5.length <= wave3.length)

    # 2. Wave 4 often overlaps with wave 1
    wave4_overlaps = False
    if wave1.is_up:
        wave4_overlaps = wave4.end.price < wave1.end.price
    else:
        wave4_overlaps = wave4.end.price > wave1.end.price

    # 3. Wedge shape with converging trendlines
    wedge_shape = self._check_wedge_shape(wave)

    # Classify as diagonal if at least two of the three conditions are met
    diagonal_score = sum([decreasing_length, wave4_overlaps, wedge_shape])
    return diagonal_score >= 2

def _check_wedge_shape(self, wave: Wave) -> bool:
    """
    Check if wave forms a wedge shape with converging trendlines.

    Parameters:
        wave: The impulse wave to check

    Returns:
        bool: True if the wave forms a wedge shape
    """
    if not wave.subwaves or len(wave.subwaves) != 5:
        return False

    wave1, wave2, wave3, wave4, wave5 = wave.subwaves

    # Calculate slopes of lines connecting wave 1 to wave 3 to wave 5
    # and wave 2 to wave 4

    # Only check for convergence if we have enough points
    try:
        slope1 = (wave5.end.price - wave1.start.price) / (wave5.end.index - wave1.start.index)
        slope2 = (wave4.end.price - wave2.end.price) / (wave4.end.index - wave2.end.index)

        # In a wedge, the slopes have the same sign but are converging
        same_sign = (slope1 > 0 and slope2 > 0) or (slope1 < 0 and slope2 < 0)

        if not same_sign:
            return False

        # Check for convergence
        if wave1.is_up:
            # In an upward diagonal, upper trendline has lower slope than lower trendline
            return slope1 < slope2
        else:
            # In a downward diagonal, upper trendline has higher slope than lower trendline
            return slope1 > slope2

    except (ZeroDivisionError, TypeError):
        return False

def _validate_corrective_waves(self, corrective_waves: List[Wave]) -> List[Wave]:
    """
    Apply Elliott Wave rules to validate corrective waves.

    Parameters:
        corrective_waves: List of candidate corrective waves

    Returns:
        List of validated corrective waves with pattern classifications
    """
    validated_waves = []

    for wave in corrective_waves:
        if not wave.subwaves or len(wave.subwaves) != 3:
            continue

        waveA, waveB, waveC = wave.subwaves

        # Calculate Fibonacci relationships for retracements
        self._calculate_corrective_relationships(wave)

        # Classify corrective pattern
        pattern = self._classify_corrective_pattern(wave)

        # Set the pattern attribute
        wave.pattern = pattern

        # Wave passes basic validation
        validated_waves.append(wave)

    return validated_waves

def _classify_corrective_pattern(self, wave: Wave) -> str:
    """
    Classify the type of corrective pattern.

    Parameters:
        wave: The corrective wave to classify

    Returns:
        str: Pattern classification ('zigzag', 'flat', 'triangle', etc.)
    """
    if not wave.subwaves or len(wave.subwaves) != 3:
        return "unknown"

    waveA, waveB, waveC = wave.subwaves

    # Calculate B wave retracement of A
    b_retracement = waveB.length / waveA.length

    # Calculate relative lengths
    c_to_a_ratio = waveC.length / waveA.length

    # Zigzag: Wave B retraces less than 61.8% of Wave A,
    # Wave C extends beyond the end of Wave A
    if b_retracement <= 0.618:
        if (waveA.is_up and waveC.end.price > waveA.end.price) or \
           (not waveA.is_up and waveC.end.price < waveA.end.price):
            return "zigzag"

    # Flat: Wave B retraces close to 100% of Wave A,
    # Wave C approximately equals Wave A in length
    if 0.9 <= b_retracement <= 1.1 and 0.9 <= c_to_a_ratio <= 1.1:
        return "flat"

    # Expanded Flat: Wave B goes beyond the start of Wave A,
    # Wave C goes beyond the end of Wave A
    if b_retracement > 1.0:
        if (waveA.is_up and waveC.end.price > waveA.end.price) or \
           (not waveA.is_up and waveC.end.price < waveA.end.price):
            return "expanded_flat"

    # Running Flat: Wave B goes beyond the start of Wave A,
    # Wave C fails to reach the end of Wave A
    if b_retracement > 1.0:
        if (waveA.is_up and waveC.end.price <= waveA.end.price) or \
           (not waveA.is_up and waveC.end.price >= waveA.end.price):
            return "running_flat"

    # Triangle: Waves become progressively smaller
    if b_retracement < 0.9 and c_to_a_ratio < 0.9:
        return "triangle"

    # Default
    return "complex"

def _calculate_wave_relationships(self, wave: Wave):
    """
    Calculate Fibonacci relationships between waves in an impulse pattern.

    Parameters:
        wave: The impulse wave to analyze
    """
    if not wave.subwaves or len(wave.subwaves) != 5:
        return

    wave1, wave2, wave3, wave4, wave5 = wave.subwaves

    # Calculate wave lengths
    wave1_length = wave1.length
    wave3_length = wave3.length
    wave5_length = wave5.length

    # Wave 2 retracement of Wave 1
    wave2_retracement = wave2.length / wave1_length

    # Wave 3 extension relative to Wave 1
    wave3_extension = wave3_length / wave1_length

    # Wave 4 retracement of Wave 3
    wave4_retracement = wave4.length / wave3_length

    # Wave 5 relative to Wave 1
    wave5_to_wave1_ratio = wave5_length / wave1_length

    # Store relationships as attributes
    wave.wave2_retracement = wave2_retracement
    wave.wave3_extension = wave3_extension
    wave.wave4_retracement = wave4_retracement
    wave.wave5_to_wave1_ratio = wave5_to_wave1_ratio

    # Identify extended waves
    extended_wave = None
    if wave3_length > 1.618 * wave1_length and wave3_length > 1.618 * wave5_length:
        extended_wave = 3
    elif wave1_length > 1.618 * wave3_length and wave1_length > 1.618 * wave5_length:
        extended_wave = 1
    elif wave5_length > 1.618 * wave1_length and wave5_length > 1.618 * wave3_length:
        extended_wave = 5

    wave.extended_wave = extended_wave

def _calculate_corrective_relationships(self, wave: Wave):
    """
    Calculate Fibonacci relationships in a corrective pattern.

    Parameters:
        wave: The corrective wave to analyze
    """
    if not wave.subwaves or len(wave.subwaves) != 3:
        return

    waveA, waveB, waveC = wave.subwaves

    # Wave B retracement of Wave A
    waveB_retracement = waveB.length / waveA.length if waveA.length > 0 else 0

    # Wave C relative to Wave A
    waveC_to_waveA_ratio = waveC.length / waveA.length if waveA.length > 0 else 0

    # Store relationships as attributes
    wave.waveB_retracement = waveB_retracement
    wave.waveC_to_waveA_ratio = waveC_to_waveA_ratio

def _identify_complete_wave_sequences(self, impulse_waves: List[Wave],
                                    corrective_waves: List[Wave]) -> List[Wave]:
    """
    Identify complete Elliott Wave sequences by combining impulse and corrective patterns.

    Parameters:
        impulse_waves: List of validated impulse waves
        corrective_waves: List of validated corrective waves

    Returns:
        List of waves that form higher-degree Elliott Wave sequences
    """
    # Sort waves by start index
    all_waves = impulse_waves + corrective_waves
    all_waves.sort(key=lambda w: w.start.index)

    complete_sequences = []

    # Look for impulse followed by correction (5-3 sequence)
    for i in range(len(all_waves) - 1):
        first_wave = all_waves[i]
        second_wave = all_waves[i + 1]

        # Check if we have an impulse followed by a correction
        if (first_wave.wave_type == WaveType.IMPULSE and
            second_wave.wave_type == WaveType.CORRECTIVE):

            # Check if the correction starts approximately where the impulse ends
            if abs(second_wave.start.index - first_wave.end.index) <= 3:
                # Create a higher degree wave
                higher_degree_wave = Wave(
                    start=first_wave.start,
                    end=second_wave.end,
                    wave_num="12345-ABC",
                    wave_type=WaveType.IMPULSE,  # This would be the wave type of the next higher degree
                    subwaves=[first_wave, second_wave]
                )

                complete_sequences.append(higher_degree_wave)

    return complete_sequences

def get_current_wave_position(self, impulse_waves: List[Wave],
                            corrective_waves: List[Wave]) -> str:
    """
    Determine the current position in the Elliott Wave sequence.

    Parameters:
        impulse_waves: List of identified impulse waves
        corrective_waves: List of identified corrective waves

    Returns:
        str: Description of current wave position
    """
    if not impulse_waves and not corrective_waves:
        return "No clear wave structure identified"

    # Combine and sort all waves by end index
    all_waves = impulse_waves + corrective_waves
    all_waves.sort(key=lambda x: x.end.index)

    # Get the most recent wave
    last_wave = all_waves[-1]

    # Check complete patterns first
    if last_wave.wave_num == "12345":
        return "After complete 5-wave impulse, expecting A-B-C correction"
    elif last_wave.wave_num == "ABC":
        return "After complete A-B-C correction, expecting new impulse wave"

    # Check for subwave position
    if last_wave.wave_type == WaveType.IMPULSE:
        if last_wave.wave_num == "1":
            return "After Wave 1, expecting Wave 2 correction"
        elif last_wave.wave_num == "3":
            return "After Wave 3, expecting Wave 4 correction"
        elif last_wave.wave_num == "5":
            return "After Wave 5, expecting A-B-C correction"
    elif last_wave.wave_type == WaveType.CORRECTIVE:
        if last_wave.wave_num == "2":
            return "After Wave 2, expecting Wave 3 impulse"
        elif last_wave.wave_num == "4":
            return "After Wave 4, expecting Wave 5 impulse"
        elif last_wave.wave_num == "A":
            return "After Wave A, expecting Wave B correction"
        elif last_wave.wave_num == "B":
            return "After Wave B, expecting Wave C impulse"

    return "Current wave position unclear"

def calculate_fibonacci_projections(self, impulse_waves: List[Wave],
                                  corrective_waves: List[Wave],
                                  current_price: float) -> Dict[str, float]:
    """
    Calculate Fibonacci projections for potential future price movements.

    Parameters:
        impulse_waves: List of identified impulse waves
        corrective_waves: List of identified corrective waves
        current_price: Current price level

    Returns:
        Dict[str, float]: Projected price levels
    """
    projections = {}

    # Combine and sort all waves by end index
    all_waves = impulse_waves + corrective_waves
    all_waves.sort(key=lambda x: x.end.index)

    if not all_waves:
        return projections

    # Get the most recent wave and its type
    last_wave = all_waves[-1]
    current_position = self.get_current_wave_position(impulse_waves, corrective_waves)

    # Create wave lookup dictionary
    wave_dict = {wave.wave_num: wave for wave in all_waves}

    # Calculate projections based on current wave position
    if "expecting Wave 2 correction" in current_position:
        # After wave 1
        if "1" in wave_dict:
            wave1 = wave_dict["1"]
            wave1_length = wave1.length

            # Common wave 2 retracements (38.2%, 50%, 61.8%)
            for name, ratio in [("0.382", 0.382), ("0.5", 0.5), ("0.618", 0.618)]:
                if wave1.is_up:
                    level = wave1.end.price - (wave1_length * float(name))
                    projections[f"Wave 2 ({name} retracement)"] = level
                else:
                    level = wave1.end.price + (wave1_length * float(name))
                    projections[f"Wave 2 ({name} retracement)"] = level

    elif "expecting Wave 3 impulse" in current_position:
        # After wave 2
        if "1" in wave_dict and "2" in wave_dict:
            wave1 = wave_dict["1"]
            wave2 = wave_dict["2"]
            wave1_length = wave1.length

            # Common wave 3 extensions (1.618, 2.618, 4.236 of wave 1)
            for name, ratio in [("1.618", 1.618), ("2.618", 2.618), ("4.236", 4.236)]:
                if wave1.is_up:
                    level = wave2.end.price + (wave1_length * float(name))
                    projections[f"Wave 3 ({name} extension)"] = level
                else:
                    level = wave2.end.price - (wave1_length * float(name))
                    projections[f"Wave 3 ({name} extension)"] = level

    elif "expecting Wave 4 correction" in current_position:
        # After wave 3
        if "1" in wave_dict and "3" in wave_dict:
            wave1 = wave_dict["1"]
            wave3 = wave_dict["3"]
            wave3_length = wave3.length

            # Common wave 4 retracements (23.6%, 38.2%)
            for name, ratio in [("0.236", 0.236), ("0.382", 0.382)]:
                if wave3.is_up:
                    level = wave3.end.price - (wave3_length * float(name))
                    projections[f"Wave 4 ({name} retracement)"] = level
                else:
                    level = wave3.end.price + (wave3_length * float(name))
                    projections[f"Wave 4 ({name} retracement)"] = level

            # Wave 4 shouldn't go beyond wave 1 end
            if wave1.is_up:
                projections["Wave 4 (wave 1 top support)"] = wave1.end.price
            else:
                projections["Wave 4 (wave 1 bottom support)"] = wave1.end.price

    elif "expecting Wave 5 impulse" in current_position:
        # After wave 4
        if all(w in wave_dict for w in ["1", "3", "4"]):
            wave1 = wave_dict["1"]
            wave3 = wave_dict["3"]
            wave4 = wave_dict["4"]
            wave1_length = wave1.length

            # Equality with wave 1
            if wave1.is_up:
                projections["Wave 5 (equal to wave 1)"] = wave4.end.price + wave1_length
            else:
                projections["Wave 5 (equal to wave 1)"] = wave4.end.price - wave1_length

            # 61.8% of waves 1+3 length
            waves1_3_length = wave1.length + wave3.length
            if wave1.is_up:
                projections["Wave 5 (0.618 of waves 1+3)"] = wave4.end.price + (waves1_3_length * 0.618)
            else:
                projections["Wave 5 (0.618 of waves 1+3)"] = wave4.end.price - (waves1_3_length * 0.618)

    elif "expecting A-B-C correction" in current_position:
        # After a complete impulse
        if "5" in wave_dict or "12345" in wave_dict:
            # Use either wave 5 or the complete impulse
            impulse_wave = wave_dict.get("12345", wave_dict.get("5"))
            impulse_length = impulse_wave.length

            # Wave A projections (38.2%, 50%, 61.8% retracement of impulse)
            for name, ratio in [("0.382", 0.382), ("0.5", 0.5), ("0.618", 0.618)]:
                if impulse_wave.is_up:
                    level = impulse_wave.end.price - (impulse_length * float(name))
                    projections[f"Wave A ({name} retracement)"] = level
                else:
                    level = impulse_wave.end.price + (impulse_length * float(name))
                    projections[f"Wave A ({name} retracement)"] = level

    elif "expecting Wave B correction" in current_position:
        # After wave A
        if "A" in wave_dict:
            waveA = wave_dict["A"]
            waveA_length = waveA.length

            # Wave B typically retraces 50% to 78.6% of wave A
            for name, ratio in [("0.5", 0.5), ("0.618", 0.618), ("0.786", 0.786)]:
                if waveA.is_up:
                    level = waveA.end.price + (waveA_length * float(name))
                    projections[f"Wave B ({name} retracement)"] = level
                else:
                    level = waveA.end.price - (waveA_length * float(name))
                    projections[f"Wave B ({name} retracement)"] = level

    elif "expecting Wave C impulse" in current_position:
        # After wave B
        if "A" in wave_dict and "B" in wave_dict:
            waveA = wave_dict["A"]
            waveB = wave_dict["B"]
            waveA_length = waveA.length

            # Wave C often equals wave A or extends to 1.618 times wave A
            for name, ratio in [("1.0", 1.0), ("1.618", 1.618)]:
                if waveA.is_up:
                    level = waveB.end.price - (waveA_length * float(name))
                    projections[f"Wave C ({name} of wave A)"] = level
                else:
                    level = waveB.end.price + (waveA_length * float(name))
                    projections[f"Wave C ({name} of wave A)"] = level

    elif "expecting new impulse wave" in current_position:
        # After a complete correction
        if "ABC" in wave_dict:
            correction = wave_dict["ABC"]
            correction_length = correction.length

            # New impulse wave projections (typical 1.618 extension of correction)
            if correction.is_up:
                projections["Wave 1 (1.618 of correction)"] = correction.end.price + (correction_length * 1.618)
            else:
                projections["Wave 1 (1.618 of correction)"] = correction.end.price - (correction_length * 1.618)

    return projections

def __init__(self, username=None, password=None):
    """
    Initialize the wave prediction system.

    Parameters:
        username (str, optional): TradingView username
        password (str, optional): TradingView password
    """
    # Analysis components
    self.wave_detector = WaveDetector()
    self.elliott_counter = ElliottWaveCounter(self.wave_detector)

    # Initialize TVDatafeed
    self.tv = TvDatafeed(username, password)

    # Data containers
    self.historical_data = None
    self.impulse_waves = []
    self.corrective_waves = []
    self.wave_points = []
    self.current_position = "No analysis performed yet"
    self.fibonacci_projections = {}

    # Default parameters
    self.params = {
        'n_cycles': 3,              # Number of dominant cycles to detect
        'min_period': 5,            # Minimum cycle period (in bars)
        'max_period': None,         # Maximum cycle period (auto-calculated if None)
        'cycle_method': 'cycle_extrema',  # Method for cycle-based pivot detection
        'window_factor': 0.5,       # Window size as a factor of cycle period
        'min_strength': 0.3,        # Minimum strength for pivot detection
        'timeframe_adjust': True,   # Automatically adjust parameters for timeframe
    }

def load_historical_data(self, symbol, exchange='NASDAQ', interval=Interval.in_daily,
                       start_date=None, end_date=None, n_bars=None):
    """
    Load market data from TradingView with flexible date options.

    Parameters:
        symbol (str): Stock or crypto symbol
        exchange (str): Exchange name (default: 'NASDAQ')
        interval (Interval): Time interval from tvDatafeed.Interval
        start_date (str or datetime, optional): Start date in 'YYYY-MM-DD' format or datetime
        end_date (str or datetime, optional): End date in 'YYYY-MM-DD' format or datetime
        n_bars (int, optional): Number of bars to load if dates not specified

    Returns:
        pandas.DataFrame: Historical price data
    """
    print(f"Loading data for {symbol} from {exchange}...")

    # Process date parameters
    if end_date is None:
        end_date = datetime.now()
    elif isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')

    if start_date is None and n_bars is None:
        # Default to 2 years of data
        start_date = end_date - timedelta(days=365 * 2)
    elif start_date is None:
        # Use n_bars if start_date not provided
        start_date = None  # TVDatafeed will handle this with n_bars
    elif isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')

    # Get data from TradingView
    if start_date is not None:
        # Get data based on date range
        self.historical_data = self.tv.get_hist(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            n_bars=5000,  # Maximum, will be filtered by date range
            extended_session=False
        )

        # Filter by date range
        self.historical_data = self.historical_data.loc[start_date:end_date]
    else:
        # Get data based on n_bars
        self.historical_data = self.tv.get_hist(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            n_bars=n_bars or 500,
            extended_session=False
        )

    print(f"Loaded {len(self.historical_data)} data points from {self.historical_data.index[0].date()} to {self.historical_data.index[-1].date()}")

    # Adjust parameters based on timeframe if enabled
    if self.params['timeframe_adjust']:
        self._adjust_params_for_timeframe(interval)

    return self.historical_data


def _adjust_params_for_timeframe(self, interval):
    """
    Adjust analysis parameters based on the selected timeframe.

    Parameters:
        interval (Interval): Time interval from tvDatafeed.Interval
    """
    # Map time intervals to scaling factors
    scaling_factors = {
        Interval.in_1_minute: 0.2,
        Interval.in_3_minute: 0.3,
        Interval.in_5_minute: 0.4,
        Interval.in_15_minute: 0.5,
        Interval.in_30_minute: 0.6,
        Interval.in_45_minute: 0.7,
        Interval.in_1_hour: 0.8,
        Interval.in_2_hour: 0.9,
        Interval.in_4_hour: 1.0,
        Interval.in_daily: 1.2,
        Interval.in_weekly: 1.5,
        Interval.in_monthly: 2.0,
    }

    # Get scaling factor for the current interval
    scaling = scaling_factors.get(interval, 1.0)

    # Adjust parameters
    self.params['min_period'] = max(3, int(5 * scaling))

    if interval in [Interval.in_1_minute, Interval.in_3_minute, Interval.in_5_minute]:
        self.params['n_cycles'] = 5  # More cycles for shorter timeframes
        self.params['window_factor'] = 0.3
        self.params['min_strength'] = 0.2
    elif interval in [Interval.in_15_minute, Interval.in_30_minute, Interval.in_45_minute]:
        self.params['n_cycles'] = 4
        self.params['window_factor'] = 0.4
        self.params['min_strength'] = 0.25
    elif interval in [Interval.in_1_hour, Interval.in_2_hour, Interval.in_4_hour]:
        self.params['n_cycles'] = 3
        self.params['window_factor'] = 0.5
        self.params['min_strength'] = 0.3
    else:  # Daily and above
        self.params['n_cycles'] = 3
        self.params['window_factor'] = 0.5
        self.params['min_strength'] = 0.4

def set_analysis_params(self, **kwargs):
    """
    Set custom parameters for the analysis.

    Parameters:
        **kwargs: Parameter key-value pairs to override defaults

    Returns:
        dict: Updated parameters
    """
    # Update parameters with provided values
    for key, value in kwargs.items():
        if key in self.params:
            self.params[key] = value
        else:
            print(f"Warning: Unknown parameter '{key}'")

    return self.params

def analyze_waves(self, timeframe='daily'):
    """
    Perform Elliott Wave analysis on the loaded data.

    Parameters:
        timeframe (str): Chart timeframe label for context

    Returns:
        dict: Analysis results
    """
    if self.historical_data is None:
        raise ValueError("No data loaded. Please call load_historical_data first.")

    print("Analyzing wave patterns using FFT cycle detection...")

    # Create parameter dictionary for wave detection
    detection_params = {
        'n_cycles': self.params['n_cycles'],
        'min_period': self.params['min_period'],
        'max_period': self.params['max_period'],
        'method': self.params['cycle_method'],
        'window_factor': self.params['window_factor'],
        'min_strength': self.params['min_strength']
    }

    # Identify Elliott Waves
    self.impulse_waves, self.corrective_waves, self.wave_points = self.elliott_counter.identify_elliott_waves(
        self.historical_data,
        timeframe=timeframe,
        **detection_params
    )

    # Determine current position in wave sequence
    self.current_position = self.elliott_counter.get_current_wave_position(
        self.impulse_waves,
        self.corrective_waves
    )

    # Calculate Fibonacci projections
    current_price = self.historical_data['close'].iloc[-1]
    self.fibonacci_projections = self.elliott_counter.calculate_fibonacci_projections(
        self.impulse_waves,
        self.corrective_waves,
        current_price
    )

    # Prepare analysis results
    results = {
        "current_price": current_price,
        "current_position": self.current_position,
        "impulse_waves": len(self.impulse_waves),
        "corrective_waves": len(self.corrective_waves),
        "wave_points": len(self.wave_points),
        "projections": self.fibonacci_projections,
        "dominant_cycles": self.wave_detector.cycle_analyzer.dominant_cycles
    }

    print(f"Analysis complete. Current position: {self.current_position}")
    print(f"Dominant cycles detected: {self.wave_detector.cycle_analyzer.dominant_cycles}")

    return results

def plot_cycle_analysis(self):
    """
    Plot the cycle analysis results.

    Returns:
        matplotlib.figure.Figure: The cycle analysis plot
    """
    if not hasattr(self.wave_detector.cycle_analyzer, 'cycles') or not self.wave_detector.cycle_analyzer.cycles:
        raise ValueError("No cycle analysis performed yet. Call analyze_waves first.")

    return self.wave_detector.cycle_analyzer.plot_cycles(
        self.historical_data['close'],
        title="FFT Cycle Analysis"
    )

def plot_wave_analysis(self, show_cycles=True, show_indicators=False):
    """
    Create visualization of the Elliott Wave analysis.

    Parameters:
        show_cycles (bool): Whether to show dominant cycle patterns
        show_indicators (bool): Whether to show technical indicators

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    if self.historical_data is None or not (self.impulse_waves or self.corrective_waves):
        raise ValueError("Analysis must be performed first")

    # Create figure and subplot
    fig = plt.figure(figsize=(15, 10))

    # Determine number of rows based on what we're showing
    n_rows = 1 + (1 if show_cycles else 0) + (1 if show_indicators else 0)

    # Create subplots
    gs = fig.add_gridspec(n_rows, 1, height_ratios=[3] + [1] * (n_rows - 1))

    # Main price chart with wave labels
    ax1 = fig.add_subplot(gs[0])

    # Plot price data
    ax1.plot(self.historical_data.index, self.historical_data['close'],
             label='Price', color='blue', linewidth=1.5)

    # Define colors for different wave types
    impulse_colors = {'1': 'green', '3': 'green', '5': 'green', '12345': 'darkgreen'}
    corrective_colors = {'2': 'red', '4': 'red', 'A': 'red', 'B': 'purple', 'C': 'red', 'ABC': 'darkred'}

    # Plot impulse waves
    for wave in self.impulse_waves:
        color = impulse_colors.get(wave.wave_num, 'green')

        # Plot the wave
        dates = [wave.start.date, wave.end.date]
        prices = [wave.start.price, wave.end.price]
        ax1.plot(dates, prices, color=color, linewidth=2)

        # Add wave label
        mid_point = dates[0] + (dates[1] - dates[0]) / 2
        mid_price = (prices[0] + prices[1]) / 2
        ax1.text(mid_point, mid_price,
                wave.wave_num, fontsize=10, ha='center', va='bottom',
                bbox=dict(facecolor='white', alpha=0.7))

        # Plot subwaves if available
        if hasattr(wave, 'subwaves') and wave.subwaves:
            for subwave in wave.subwaves:
                subcolor = impulse_colors.get(subwave.wave_num, 'green') if subwave.wave_type == WaveType.IMPULSE else corrective_colors.get(subwave.wave_num, 'red')

                # Plot the subwave
                subdates = [subwave.start.date, subwave.end.date]
                subprices = [subwave.start.price, subwave.end.price]
                ax1.plot(subdates, subprices, color=subcolor, linewidth=1.5, alpha=0.7)

    # Plot corrective waves
    for wave in self.corrective_waves:
        color = corrective_colors.get(wave.wave_num, 'red')

        # Plot the wave
        dates = [wave.start.date, wave.end.date]
        prices = [wave.start.price, wave.end.price]
        ax1.plot(dates, prices, color=color, linewidth=2)

        # Add wave label
        mid_point = dates[0] + (dates[1] - dates[0]) / 2
        mid_price = (prices[0] + prices[1]) / 2
        ax1.text(mid_point, mid_price,
                wave.wave_num, fontsize=10, ha='center', va='bottom',
                bbox=dict(facecolor='white', alpha=0.7))

        # Plot subwaves if available
        if hasattr(wave, 'subwaves') and wave.subwaves:
            for subwave in wave.subwaves:
                subcolor = impulse_colors.get(subwave.wave_num, 'green') if subwave.wave_type == WaveType.IMPULSE else corrective_colors.get(subwave.wave_num, 'red')

                # Plot the subwave
                subdates = [subwave.start.date, subwave.end.date]
                subprices = [subwave.start.price, subwave.end.price]
                ax1.plot(subdates, subprices, color=subcolor, linewidth=1.5, alpha=0.7)

    # Plot price projections from Fibonacci analysis
    current_price = self.historical_data['close'].iloc[-1]

    if self.fibonacci_projections:
        # Sort projections by price level
        up_projections = {k: v for k, v in self.fibonacci_projections.items() if v > current_price}
        down_projections = {k: v for k, v in self.fibonacci_projections.items() if v <= current_price}

        # Plot up projections (green)
        for i, (label, price) in enumerate(sorted(up_projections.items(), key=lambda x: x[1])):
            alpha = 0.7 - (i * 0.05 if i < 10 else 0.5)  # Decrease alpha for less important levels
            ax1.axhline(y=price, linestyle='--', alpha=max(0.2, alpha),
                       color='green', linewidth=1)

            # Add label at the right edge of the chart
            ax1.text(self.historical_data.index[-1], price,
                     f"{label}: {price:.2f}", va='center', alpha=alpha, color='green',
                     bbox=dict(facecolor='white', alpha=0.5))

        # Plot down projections (red)
        for i, (label, price) in enumerate(sorted(down_projections.items(), key=lambda x: x[1], reverse=True)):
            alpha = 0.7 - (i * 0.05 if i < 10 else 0.5)
            ax1.axhline(y=price, linestyle='--', alpha=max(0.2, alpha),
                       color='red', linewidth=1)

            # Add label at the right edge of the chart
            ax1.text(self.historical_data.index[-1], price,
                     f"{label}: {price:.2f}", va='center', alpha=alpha, color='red',
                     bbox=dict(facecolor='white', alpha=0.5))

    # Highlight current price
    ax1.axhline(y=current_price, color='black', linestyle='-', alpha=0.5)
    ax1.text(self.historical_data.index[-1], current_price,
             f"Current: {current_price:.2f}", va='center', ha='right',
             bbox=dict(facecolor='white', alpha=0.8))

    # Add dominant cycle information to the plot
    if hasattr(self.wave_detector.cycle_analyzer, 'dominant_cycles'):
        cycles_text = "Dominant Cycles: " + ", ".join([str(c) for c in self.wave_detector.cycle_analyzer.dominant_cycles])
        ax1.text(0.02, 0.02, cycles_text, transform=ax1.transAxes,
                bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='bottom')

    # Configure main chart
    ax1.set_title(f'Elliott Wave Analysis with FFT Cycles - {self.current_position}',
                 fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Plot dominant cycle overlays if requested
    if show_cycles and hasattr(self.wave_detector.cycle_analyzer, 'dominant_cycles'):
        subplot_index = 1

        # Create cycle subplot
        ax_cycle = fig.add_subplot(gs[subplot_index], sharex=ax1)

        # Plot cycle reconstructions
        colors = ['red', 'green', 'blue', 'purple', 'orange']

        for i, cycle_period in enumerate(self.wave_detector.cycle_analyzer.dominant_cycles[:3]):  # Show top 3 cycles
            cycle_data = self.wave_detector.cycle_analyzer.reconstruct_cycle(
                self.historical_data['close'],
                cycle_period
            )

            color = colors[i % len(colors)]
            ax_cycle.plot(self.historical_data.index, cycle_data,
                         label=f'Cycle {cycle_period}', color=color, alpha=0.7)

        ax_cycle.set_title('Dominant Market Cycles', fontsize=12)
        ax_cycle.set_ylabel('Cycle Value', fontsize=10)
        ax_cycle.grid(True, alpha=0.3)
        ax_cycle.legend(loc='upper left')

        subplot_index += 1

    # Plot technical indicators if requested
    if show_indicators:
        # Add technical indicators
        df_indicators = self.add_technical_indicators(['RSI'])

        # Add RSI subplot
        ax_rsi = fig.add_subplot(gs[-1], sharex=ax1)
        ax_rsi.plot(df_indicators.index, df_indicators['RSI'], label='RSI', color='purple')
        ax_rsi.axhline(y=70, color='red', linestyle='--', alpha=0.5)
        ax_rsi.axhline(y=30, color='green', linestyle='--', alpha=0.5)
        ax_rsi.set_ylim(0, 100)
        ax_rsi.set_title('RSI', fontsize=12)
        ax_rsi.grid(True, alpha=0.3)

    # Format date ticks
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    return fig

def add_technical_indicators(self, indicators=None):
    """
    Add technical indicators for trading decisions (not for wave counting).
    This method is separated from wave counting to maintain pure wave analysis.

    Parameters:
        indicators (list, optional): List of indicators to add

    Returns:
        pd.DataFrame: DataFrame with indicators
    """
    if self.historical_data is None:
        raise ValueError("No data loaded")

    if indicators is None:
        indicators = ['RSI', 'MACD', 'BBands']

    df = self.historical_data.copy()

    # Calculate RSI (Relative Strength Index)
    if 'RSI' in indicators:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).fillna(0)
        loss = -delta.where(delta < 0, 0).fillna(0)

        # Calculate average gain and average loss
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()

        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

    # Calculate MACD (Moving Average Convergence Divergence)
    if 'MACD' in indicators:
        df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']

    # Calculate Bollinger Bands
    if 'BBands' in indicators:
        df['BBMiddle'] = df['close'].rolling(window=20).mean()
        df = df.dropna()
        std = df['close'].rolling(window=20).std()
        df['BBUpper'] = df['BBMiddle'] + (2 * std)
        df['BBLower'] = df['BBMiddle'] - (2 * std)

    return df

def save_analysis_report(self, filename=None):
    """
    Save the complete analysis to an HTML report.

    Parameters:
        filename (str, optional): Output filename

    Returns:
        str: Path to saved file
    """
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"elliott_wave_analysis_{timestamp}.html"

    if self.historical_data is None or not hasattr(self, 'current_position'):
        raise ValueError("Analysis must be performed first")

    # Current price
    current_price = self.historical_data['close'].iloc[-1]

    # Generate HTML report
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>FFT-Based Elliott Wave Analysis</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; color: #333; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .section {{ margin-bottom: 30px; border: 1px solid #ddd; border-radius: 5px; padding: 20px; background-color: #f9f9f9; }}
            .current-price {{ font-size: 24px; font-weight: bold; color: #333; }}
            .wave-position {{ font-size: 18px; color: #2980b9; margin-bottom: 20px; }}
            .projection {{ margin: 5px 0; }}
            .up {{ color: green; }}
            .down {{ color: red; }}
            .neutral {{ color: blue; }}
            table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
            th, td {{ text-align: left; padding: 12px; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #2c3e50; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            tr:hover {{ background-color: #e9e9e9; }}
            .cycle-info {{ background-color: #e8f4f8; padding: 15px; border-left: 5px solid #3498db; margin-bottom: 10px; }}
            .info-box {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .disclaimer {{ font-size: 12px; color: #7f8c8d; margin-top: 30px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>FFT-Based Elliott Wave Analysis Report</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

            <div class="section">
                <h2>Current Market Position</h2>
                <p class="current-price">Current Price: {current_price:.2f}</p>
                <p class="wave-position">{self.current_position}</p>

                <div class="cycle-info">
                    <h3>Dominant Market Cycles</h3>
    """

    # Add cycle information
    if hasattr(self.wave_detector.cycle_analyzer, 'dominant_cycles'):
        html += "<ul>"
        for cycle in self.wave_detector.cycle_analyzer.dominant_cycles:
            html += f"<li>Cycle length: {cycle} bars</li>"
        html += "</ul>"
    else:
        html += "<p>No cycle analysis performed.</p>"

    html += """
                </div>

                <div class="info-box">
                    <h3>Wave Count Summary</h3>
    """

    # Add wave count summary
    html += f"""
                    <p>Identified {len(self.impulse_waves)} impulse wave structures and {len(self.corrective_waves)} corrective wave structures.</p>
                </div>
            </div>

            <div class="section">
                <h2>Price Projections</h2>
                <table>
                    <tr>
                        <th>Target Description</th>
                        <th>Price Level</th>
                        <th>Change from Current</th>
                    </tr>
    """

    # Add price projections
    if self.fibonacci_projections:
        sorted_projections = sorted(
            self.fibonacci_projections.items(),
            key=lambda x: abs(x[1] - current_price)
        )

        for label, price in sorted_projections:
            pct_change = ((price / current_price) - 1) * 100
            direction_class = "up" if price > current_price else "down"

            html += f"""
                <tr>
                    <td>{label}</td>
                    <td class="{direction_class}">{price:.2f}</td>
                    <td class="{direction_class}">{pct_change:+.2f}%</td>
                </tr>
            """
    else:
        html += """
                <tr>
                    <td colspan="3">No price projections available</td>
                </tr>
        """

    html += """
                </table>
            </div>

            <div class="info-box">
                <h3>Analysis Parameters</h3>
                <table>
                    <tr>
                        <th>Parameter</th>
                        <th>Value</th>
                    </tr>
    """

    # Add analysis parameters
    for param, value in self.params.items():
        html += f"""
                <tr>
                    <td>{param}</td>
                    <td>{value}</td>
                </tr>
        """

    html += """
                </table>
            </div>

            <div class="section">
                <h2>Wave Details</h2>
                <h3>Impulse Waves</h3>
                <table>
                    <tr>
                        <th>Wave</th>
                        <th>Direction</th>
                        <th>Start Price</th>
                        <th>End Price</th>
                        <th>Length</th>
                        <th>Duration</th>
                    </tr>
    """

    # Add impulse wave details
    if self.impulse_waves:
        for wave in sorted(self.impulse_waves, key=lambda w: w.start.index):
            direction = "Up" if wave.is_up else "Down"
            html += f"""
                <tr>
                    <td>{wave.wave_num}</td>
                    <td>{direction}</td>
                    <td>{wave.start.price:.2f}</td>
                    <td>{wave.end.price:.2f}</td>
                    <td>{wave.length:.2f}</td>
                    <td>{wave.duration} bars</td>
                </tr>
            """
    else:
        html += """
                <tr>
                    <td colspan="6">No impulse waves identified</td>
                </tr>
        """

    html += """
                </table>

                <h3>Corrective Waves</h3>
                <table>
                    <tr>
                        <th>Wave</th>
                        <th>Pattern</th>
                        <th>Direction</th>
                        <th>Start Price</th>
                        <th>End Price</th>
                        <th>Length</th>
                    </tr>
    """

    # Add corrective wave details
    if self.corrective_waves:
        for wave in sorted(self.corrective_waves, key=lambda w: w.start.index):
            direction = "Up" if wave.is_up else "Down"
            pattern = wave.pattern if hasattr(wave, 'pattern') else "Unknown"
            html += f"""
                <tr>
                    <td>{wave.wave_num}</td>
                    <td>{pattern}</td>
                    <td>{direction}</td>
                    <td>{wave.start.price:.2f}</td>
                    <td>{wave.end.price:.2f}</td>
                    <td>{wave.length:.2f}</td>
                </tr>
            """
    else:
        html += """
                <tr>
                    <td colspan="6">No corrective waves identified</td>
                </tr>
        """

    html += """
                </table>
            </div>

            <div class="disclaimer">
                <p><strong>Disclaimer:</strong> This analysis is based on Elliott Wave Theory and FFT cycle detection and represents a probabilistic view of potential market movements. All trading decisions should be made with appropriate risk management and in consideration of your personal financial situation. Past performance is not indicative of future results.</p>
            </div>
        </div>
    </body>
    </html>
    """

    # Save to file
    with open(filename, 'w') as f:
        f.write(html)

    print(f"Analysis report saved to {filename}")
    return filename



def main():
    """
    Main function to demonstrate the FFT-based Elliott Wave Prediction System with user inputs.
    """
    import matplotlib.pyplot as plt
    from tvDatafeed import TvDatafeed, Interval

    # Initialize the prediction system
    predictor = WavePredictionSystem()

    # User inputs
    print("=== FFT-Based Elliott Wave Analysis ===")

    # Symbol and exchange
    symbol = input("Enter symbol (default: BTCUSDT): ") or "BTCUSDT"
    exchange = input("Enter exchange (default: BINANCE): ") or "BINANCE"

    # Timeframe selection
    print("\nTimeframe options:")
    print("1. 5 minute")
    print("2. 15 minute")
    print("3. 1 hour")
    print("4. 4 hour")
    print("5. Daily")
    print("6. Weekly")

    timeframe_choice = input("Select timeframe (1-6, default: 5): ") or "5"

    interval_map = {
        "1": Interval.in_5_minute,
        "2": Interval.in_15_minute,
        "3": Interval.in_1_hour,
        "4": Interval.in_4_hour,
        "5": Interval.in_daily,
        "6": Interval.in_weekly
    }

    timeframe_labels = {
        "1": "5min",
        "2": "15min",
        "3": "1h",
        "4": "4h",
        "5": "daily",
        "6": "weekly"
    }

    interval = interval_map.get(timeframe_choice, Interval.in_daily)
    timeframe_label = timeframe_labels.get(timeframe_choice, "daily")

    # Date range or bar count
    date_choice = input("\nUse specific date range? (y/n, default: y): ").lower() or "y"

    if date_choice == "y":
        start_date = input("Enter start date (YYYY-MM-DD, default: 2 years ago): ") or None
        end_date = input("Enter end date (YYYY-MM-DD, leave blank for current date): ") or None

        # Load historical data with date range
        data = predictor.load_historical_data(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            start_date=start_date,
            end_date=end_date
        )
    else:
        n_bars = int(input("Enter number of bars to analyze (default: 500): ") or 500)

        # Load historical data with bar count
        data = predictor.load_historical_data(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            n_bars=n_bars
        )

    # Advanced parameter settings
    advanced = input("\nConfigure advanced parameters? (y/n, default: n): ").lower() or "n"

    if advanced == "y":
        print("\nEnter parameters (press Enter to use defaults):")
        n_cycles = int(input("Number of cycles to detect (default: 3): ") or 3)
        min_period = int(input("Minimum cycle period (default: 5): ") or 5)
        max_period_input = input("Maximum cycle period (default: auto): ")
        max_period = int(max_period_input) if max_period_input else None

        method_options = {
            "1": "cycle_extrema",
            "2": "adaptive_threshold",
            "3": "cycle_projection"
        }

        print("\nPivot detection method:")
        print("1. Cycle extrema (default)")
        print("2. Adaptive threshold")
        print("3. Cycle projection")

        method_choice = input("Select method (1-3): ") or "1"
        cycle_method = method_options.get(method_choice, "cycle_extrema")

        window_factor = float(input("Window size factor (0.3-0.7, default: 0.5): ") or 0.5)
        min_strength = float(input("Minimum pivot strength (0.2-0.5, default: 0.3): ") or 0.3)

        # Set custom parameters
        predictor.set_analysis_params(
            n_cycles=n_cycles,
            min_period=min_period,
            max_period=max_period,
            cycle_method=cycle_method,
            window_factor=window_factor,
            min_strength=min_strength,
            timeframe_adjust=True
        )

    # Run the analysis
    print("\nAnalyzing wave patterns...")
    results = predictor.analyze_waves(timeframe=timeframe_label)

    # Display analysis results
    print(f"\nCurrent position: {predictor.current_position}")

    if hasattr(predictor.wave_detector, 'cycle_analyzer') and hasattr(predictor.wave_detector.cycle_analyzer, 'dominant_cycles'):
        print(f"Dominant cycles detected: {predictor.wave_detector.cycle_analyzer.dominant_cycles}")

    # Output options
    print("\nOutput options:")
    print("1. Show wave analysis chart")
    print("2. Show cycle analysis chart")
    print("3. Save analysis report")
    print("4. All of the above")

    output_choice = input("Select output (1-4, default: 4): ") or "4"

    if output_choice in ["1", "4"]:
        # Plot wave analysis
        fig = predictor.plot_wave_analysis(show_cycles=True)
        plt.show()

    if output_choice in ["2", "4"]:
        # Plot cycle analysis
        fig_cycle = predictor.plot_cycle_analysis()
        plt.show()

    if output_choice in ["3", "4"]:
        # Save report
        filename = f"{symbol}_{timeframe_label}_wave_analysis.html"
        report_path = predictor.save_analysis_report(filename)
        print(f"\nAnalysis report saved to: {report_path}")

if __name__ == "__main__":
    main()
