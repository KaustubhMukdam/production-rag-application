"""
TDD tests for app.rate_limit
-------------------------------
Covers:
  - TokenBucket.try_acquire() — non-blocking variant
  - PerIpRateLimiter — per-client-IP independence
"""
import time
import threading

import pytest

from app.rate_limit import TokenBucket, PerIpRateLimiter


# ---------------------------------------------------------------------------
# TokenBucket.try_acquire (non-blocking)
# ---------------------------------------------------------------------------

class TestTokenBucketTryAcquire:
    def test_try_acquire_returns_true_when_tokens_available(self):
        bucket = TokenBucket(rate=1.0, capacity=5)
        assert bucket.try_acquire() is True

    def test_try_acquire_returns_false_when_exhausted(self):
        bucket = TokenBucket(rate=0.001, capacity=2)
        bucket.try_acquire()  # 1
        bucket.try_acquire()  # 2 — now empty
        assert bucket.try_acquire() is False

    def test_try_acquire_does_not_block(self):
        """try_acquire must return immediately even when bucket is empty."""
        bucket = TokenBucket(rate=0.001, capacity=1)
        bucket.try_acquire()  # drain
        start = time.monotonic()
        bucket.try_acquire()
        elapsed = time.monotonic() - start
        assert elapsed < 0.1  # must complete in well under 100 ms

    def test_try_acquire_returns_bool(self):
        bucket = TokenBucket(rate=1.0, capacity=3)
        result = bucket.try_acquire()
        assert isinstance(result, bool)

    def test_try_acquire_respects_capacity(self):
        capacity = 3
        bucket = TokenBucket(rate=0.001, capacity=capacity)
        successes = sum(1 for _ in range(10) if bucket.try_acquire())
        assert successes == capacity

    def test_acquire_blocking_still_works(self):
        """Original blocking acquire() must be unchanged."""
        bucket = TokenBucket(rate=100.0, capacity=5)
        # Should not raise and should complete quickly with high rate
        bucket.acquire()

    def test_refills_over_time(self):
        bucket = TokenBucket(rate=100.0, capacity=2)
        bucket.try_acquire()
        bucket.try_acquire()  # drain
        time.sleep(0.05)     # wait 50 ms → refill ~5 tokens at rate=100
        assert bucket.try_acquire() is True


# ---------------------------------------------------------------------------
# PerIpRateLimiter
# ---------------------------------------------------------------------------

class TestPerIpRateLimiter:
    def test_allows_first_request(self):
        limiter = PerIpRateLimiter(rate=1.0, capacity=5)
        assert limiter.is_allowed("192.168.1.1") is True

    def test_blocks_after_burst_exhausted(self):
        capacity = 2
        limiter = PerIpRateLimiter(rate=0.001, capacity=capacity)
        results = [limiter.is_allowed("10.0.0.1") for _ in range(capacity + 2)]
        # First `capacity` requests should succeed, the rest should fail
        assert results[:capacity] == [True] * capacity
        assert results[capacity] is False

    def test_different_ips_are_independent(self):
        limiter = PerIpRateLimiter(rate=0.001, capacity=1)
        limiter.is_allowed("1.1.1.1")  # drain IP 1
        # IP 2 should still be allowed (fresh bucket)
        assert limiter.is_allowed("2.2.2.2") is True

    def test_same_ip_shares_bucket(self):
        limiter = PerIpRateLimiter(rate=0.001, capacity=1)
        limiter.is_allowed("1.1.1.1")  # drain
        assert limiter.is_allowed("1.1.1.1") is False

    def test_is_allowed_returns_bool(self):
        limiter = PerIpRateLimiter(rate=1.0, capacity=5)
        result = limiter.is_allowed("127.0.0.1")
        assert isinstance(result, bool)

    def test_thread_safety(self):
        """Multiple threads hitting the same IP must not exceed capacity."""
        capacity = 5
        limiter = PerIpRateLimiter(rate=0.001, capacity=capacity)
        successes = []
        lock = threading.Lock()

        def _hit():
            result = limiter.is_allowed("shared-ip")
            with lock:
                successes.append(result)

        threads = [threading.Thread(target=_hit) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert sum(successes) <= capacity

    def test_refills_over_time(self):
        limiter = PerIpRateLimiter(rate=100.0, capacity=1)
        limiter.is_allowed("x")  # drain
        time.sleep(0.05)         # refill
        assert limiter.is_allowed("x") is True
