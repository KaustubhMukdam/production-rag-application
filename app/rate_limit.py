import time

class TokenBucket:
    def __init__(self, rate: float, capacity: int):
        self.rate = rate
        self.capacity = capacity
        self.tokens = float(capacity)
        self.last = 0.0

    def acquire(self, tokens: int = 1) -> None:
        now = time.time()
        if self.last:
            elapsed = now - self.last
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        if self.tokens >= tokens:
            self.tokens -= tokens
            self.last = now
            return
        wait = (tokens - self.tokens) / self.rate
        time.sleep(wait)
        self.last = now + wait
        self.tokens = self.tokens + wait * self.rate - tokens


# Shared instance: 30 RPM, burst of 6 (covers citation retries + judge)
groq_limiter = TokenBucket(rate=0.5, capacity=6)
