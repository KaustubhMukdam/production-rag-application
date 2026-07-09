import { useState, useEffect } from 'react';

/**
 * Animates text word-by-word at the given speed (ms per word).
 *
 * @param text    - Full text to display.
 * @param enabled - When false the full text is shown immediately (used for
 *                  older queries that have already been animated).
 * @param speed   - Milliseconds between each word. Default 14ms ≈ fast streaming feel.
 */
export function useTypewriter(text: string, enabled: boolean, speed = 14) {
  const [displayed, setDisplayed] = useState(enabled ? '' : text);
  const [complete, setComplete] = useState(!enabled);

  useEffect(() => {
    if (!enabled) {
      setDisplayed(text);
      setComplete(true);
      return;
    }

    setDisplayed('');
    setComplete(false);

    const words = text.split(' ');
    let idx = 0;

    const timer = setInterval(() => {
      if (idx < words.length) {
        // Build string incrementally to avoid O(n²) concat
        setDisplayed(prev => (idx === 0 ? words[0] : prev + ' ' + words[idx]));
        idx++;
      } else {
        setComplete(true);
        clearInterval(timer);
      }
    }, speed);

    return () => clearInterval(timer);
  }, [text, enabled, speed]);

  return { displayed, complete };
}
