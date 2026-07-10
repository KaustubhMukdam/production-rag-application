import { useState, useEffect, useCallback } from 'react';
import { fetchHealth } from '../api';
import type { HealthData } from '../types';

/**
 * Polls GET /health and exposes a manual refresh trigger.
 *
 * Polling interval is 30 s (not 5 s) to avoid exhausting Windows socket
 * buffers (WinError 10055) under `uvicorn --reload`.  Each fetch is
 * cancelled via AbortController if the component unmounts or a new poll
 * cycle starts before the previous one completes.
 */
export function useHealth() {
  const [health, setHealth] = useState<HealthData | null>(null);
  const [tick, setTick] = useState(0);

  /** Call after triggering re-index for an immediate re-check. */
  const triggerRefresh = useCallback(() => setTick(t => t + 1), []);

  useEffect(() => {
    const controller = new AbortController();
    let timerId: ReturnType<typeof setTimeout>;

    const check = async () => {
      try {
        const data = await fetchHealth();
        if (!controller.signal.aborted) setHealth(data);
      } catch {
        if (!controller.signal.aborted) setHealth(null);
      }
    };

    const schedule = () => {
      check();
      // Poll every 30 s — generous enough to avoid socket exhaustion on Windows
      timerId = setTimeout(schedule, 30_000);
    };

    schedule();

    return () => {
      controller.abort();
      clearTimeout(timerId);
    };
  }, [tick]);

  return { health, triggerRefresh };
}
