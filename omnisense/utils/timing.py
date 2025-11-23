"""
Performance monitoring and timing utilities.

Provides decorators and context managers for profiling code execution time,
FPS calculation, and performance metrics collection.
"""

import time
import functools
from typing import Optional, Dict, Callable
from collections import deque
import threading


class FPSCounter:
    """
    FPS counter that tracks frame processing rate over a sliding window.
    Thread-safe for concurrent updates.
    """

    def __init__(self, window_size: int = 30):
        """
        Initialize FPS counter.

        Args:
            window_size: Number of frames to average over
        """
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.lock = threading.Lock()
        self.last_time = time.perf_counter()

    def update(self) -> float:
        """
        Update FPS counter with a new frame.

        Returns:
            Current FPS
        """
        current_time = time.perf_counter()
        with self.lock:
            if self.last_time is not None:
                frame_time = current_time - self.last_time
                self.frame_times.append(frame_time)
            self.last_time = current_time

            if len(self.frame_times) > 0:
                avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
            return 0.0

    def get_fps(self) -> float:
        """Get current FPS without updating."""
        with self.lock:
            if len(self.frame_times) > 0:
                avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
            return 0.0

    def reset(self):
        """Reset the FPS counter."""
        with self.lock:
            self.frame_times.clear()
            self.last_time = time.perf_counter()


class TimingStats:
    """
    Collects timing statistics for named operations.
    Thread-safe for concurrent updates.
    """

    def __init__(self):
        self.stats: Dict[str, deque] = {}
        self.lock = threading.Lock()

    def add_timing(self, name: str, duration: float):
        """Add a timing measurement."""
        with self.lock:
            if name not in self.stats:
                self.stats[name] = deque(maxlen=100)
            self.stats[name].append(duration)

    def get_stats(self, name: str) -> Optional[Dict[str, float]]:
        """
        Get statistics for a named operation.

        Returns:
            Dictionary with 'mean', 'min', 'max', 'count' or None if no data
        """
        with self.lock:
            if name not in self.stats or len(self.stats[name]) == 0:
                return None

            timings = list(self.stats[name])
            return {
                'mean': sum(timings) / len(timings),
                'min': min(timings),
                'max': max(timings),
                'count': len(timings),
                'fps': 1.0 / (sum(timings) / len(timings)) if sum(timings) > 0 else 0.0
            }

    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all operations."""
        with self.lock:
            return {name: self.get_stats(name) for name in self.stats.keys()}

    def reset(self, name: Optional[str] = None):
        """Reset statistics for a specific operation or all operations."""
        with self.lock:
            if name is None:
                self.stats.clear()
            elif name in self.stats:
                self.stats[name].clear()


# Global timing stats instance
_global_timing_stats = TimingStats()


def get_timing_stats() -> TimingStats:
    """Get the global timing stats instance."""
    return _global_timing_stats


class Timer:
    """
    Context manager for timing code blocks.

    Usage:
        with Timer() as t:
            # code to time
        print(f"Elapsed: {t.elapsed}s")
    """

    def __init__(self, name: Optional[str] = None, record: bool = False):
        """
        Initialize timer.

        Args:
            name: Optional name for the timing operation
            record: If True, record timing to global stats
        """
        self.name = name
        self.record = record
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self.start_time
        if self.record and self.name:
            _global_timing_stats.add_timing(self.name, self.elapsed)
        return False


def timeit(func: Optional[Callable] = None, *, name: Optional[str] = None, record: bool = True):
    """
    Decorator for timing function execution.

    Args:
        func: Function to decorate
        name: Custom name for timing stats (defaults to function name)
        record: If True, record timing to global stats

    Usage:
        @timeit
        def my_function():
            pass

        @timeit(name="custom_name")
        def another_function():
            pass
    """
    def decorator(f):
        operation_name = name or f.__name__

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = f(*args, **kwargs)
                return result
            finally:
                elapsed = time.perf_counter() - start
                if record:
                    _global_timing_stats.add_timing(operation_name, elapsed)

        return wrapper

    # Handle both @timeit and @timeit() syntax
    if func is None:
        return decorator
    else:
        return decorator(func)


class RateLimiter:
    """
    Rate limiter to control operation frequency.

    Ensures operations don't exceed a specified rate.
    """

    def __init__(self, max_rate: float):
        """
        Initialize rate limiter.

        Args:
            max_rate: Maximum operations per second
        """
        self.min_interval = 1.0 / max_rate if max_rate > 0 else 0
        self.last_time = 0
        self.lock = threading.Lock()

    def wait(self):
        """Wait if necessary to maintain the rate limit."""
        with self.lock:
            current_time = time.perf_counter()
            elapsed = current_time - self.last_time

            if elapsed < self.min_interval:
                sleep_time = self.min_interval - elapsed
                time.sleep(sleep_time)
                self.last_time = time.perf_counter()
            else:
                self.last_time = current_time

    def check(self) -> bool:
        """
        Check if operation should proceed without waiting.

        Returns:
            True if sufficient time has passed, False otherwise
        """
        with self.lock:
            current_time = time.perf_counter()
            elapsed = current_time - self.last_time

            if elapsed >= self.min_interval:
                self.last_time = current_time
                return True
            return False


class ExponentialMovingAverage:
    """Exponential moving average for smoothing time-series values."""

    def __init__(self, alpha: float = 0.1):
        """
        Initialize EMA.

        Args:
            alpha: Smoothing factor (0 < alpha <= 1)
                  Lower values = more smoothing
        """
        self.alpha = alpha
        self.value = None

    def update(self, new_value: float) -> float:
        """
        Update EMA with new value.

        Args:
            new_value: New measurement

        Returns:
            Updated EMA value
        """
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value

    def get(self) -> Optional[float]:
        """Get current EMA value."""
        return self.value

    def reset(self):
        """Reset the EMA."""
        self.value = None
