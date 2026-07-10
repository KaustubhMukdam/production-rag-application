import { useState, useEffect } from 'react';

/**
 * Animates text word-by-word using recursive setTimeout (NOT setInterval).
 *
 * Why setTimeout over setInterval:
 *   setInterval captures `idx` by reference — by the time React batches and
 *   executes the state-updater callback, `idx` may have advanced past
 *   words.length, giving `words[idx] === undefined` which appends the string
 *   "undefined". Recursive setTimeout passes idx as a parameter, so each
 *   call captures the correct value at invocation time.
 *
 * @param text    - Full text to animate.
 * @param enabled - When false the full text is shown instantly (older cards).
 * @param speed   - Milliseconds between words. Default 14ms.
 */
export function useTypewriter(text: string, enabled: boolean, speed = 14) {
  const [displayed, setDisplayed] = useState(enabled ? '' : text);
  const [complete, setComplete]   = useState(!enabled);

  useEffect(() => {
    if (!enabled) {
      setDisplayed(text);
      setComplete(true);
      return;
    }

    setDisplayed('');
    setComplete(false);

    const words = text.split(' ');
    let cancelled = false;

    const typeNext = (idx: number) => {
      if (cancelled) return;

      if (idx < words.length) {
        // Capture word value here — not via closure over a mutable index variable
        const word    = words[idx];
        const isFirst = idx === 0;
        setDisplayed(prev => (isFirst ? word : prev + ' ' + word));
        setTimeout(() => typeNext(idx + 1), speed);
      } else {
        if (!cancelled) setComplete(true);
      }
    };

    // Small initial delay so the card's fade-in finishes before typing starts
    const kickoff = setTimeout(() => typeNext(0), 80);

    return () => {
      cancelled = true;
      clearTimeout(kickoff);
    };
  }, [text, enabled, speed]);

  return { displayed, complete };
}
