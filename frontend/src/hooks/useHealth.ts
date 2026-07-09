import { useState, useEffect, useCallback } from 'react';
import { fetchHealth } from '../api';
import type { HealthData } from '../types';

/**
 * Polls GET /health every 5 seconds and exposes a manual refresh trigger.
 * Useful for tracking the indexing state after POST /index.
 */
export function useHealth() {
  const [health, setHealth] = useState<HealthData | null>(null);
  const [tick, setTick] = useState(0);

  /** Call after triggering a re-index to immediately re-check health. */
  const triggerRefresh = useCallback(() => setTick(t => t + 1), []);

  useEffect(() => {
    let cancelled = false;

    const check = async () => {
      try {
        const data = await fetchHealth();
        if (!cancelled) setHealth(data);
      } catch {
        if (!cancelled) setHealth(null);
      }
    };

    check();
    const id = setInterval(check, 5_000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [tick]); // re-run the whole effect (and reset interval) on manual refresh

  return { health, triggerRefresh };
}
