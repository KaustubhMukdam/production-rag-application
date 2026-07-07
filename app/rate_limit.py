"""
Rate limiting utilities — Phase 4.

TokenBucket    — leaky-bucket implementation (existing, extended).
PerIpRateLimiter — per-client-IP wrapper used by the FastAPI middleware.

Changes from Phase 3:
  - TokenBucket gains try_acquire() — a non-blocking variant that returns
    True/False instead of sleeping.  The original acquire() is unchanged.
  - PerIpRateLimiter is new — maintains one bucket per client IP and is
    safe to call from concurrent request handlers.
"""
import threading
import time


class TokenBucket:
    """Token bucket rate limiter.

    Args:
        rate:     Tokens added per second.
        capacity: Maximum burst capacity.
    """

    def __init__(self, rate: float, capacity: int):
        self.rate = rate
        self.capacity = capacity
        self.tokens = float(capacity)
        self.last = 0.0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal refill (must be called with self._lock held)
    # ------------------------------------------------------------------

    def _refill(self) -> None:
        now = time.time()
        if self.last:
            elapsed = now - self.last
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last = now

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def acquire(self, tokens: int = 1) -> None:
        """Blocking acquire — sleeps until a token is available."""
        with self._lock:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return
            wait = (tokens - self.tokens) / self.rate
        # Sleep outside the lock so other threads can refill while we wait
        time.sleep(wait)
        with self._lock:
            self._refill()
            self.tokens -= tokens

    def try_acquire(self, tokens: int = 1) -> bool:
        """Non-blocking acquire — returns True if token is available, False otherwise.

        Unlike acquire(), this method never sleeps and always returns immediately.
        Suitable for HTTP middleware where blocking would stall the event loop.
        """
        with self._lock:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False


class PerIpRateLimiter:
    """Maintain one TokenBucket per client IP address.

    Thread-safe: bucket creation is protected by a lock so concurrent
    first-requests from the same IP don't create duplicate buckets.

    Args:
        rate:     Tokens added per second (shared across all IP buckets).
        capacity: Burst capacity per IP.
    """

    def __init__(self, rate: float, capacity: int):
        self._rate = rate
        self._capacity = capacity
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def is_allowed(self, ip: str) -> bool:
        """Return True if the request from *ip* is within the rate limit.

        Creates a new bucket for first-time IPs.  Always non-blocking.
        """
        with self._lock:
            if ip not in self._buckets:
                self._buckets[ip] = TokenBucket(self._rate, self._capacity)
            bucket = self._buckets[ip]

        # try_acquire is thread-safe internally; no outer lock needed
        return bucket.try_acquire()


# ---------------------------------------------------------------------------
# Shared instances
# ---------------------------------------------------------------------------

# Groq API limiter: 30 RPM, burst of 6 (covers citation retries + judge)
groq_limiter = TokenBucket(rate=0.5, capacity=6)
