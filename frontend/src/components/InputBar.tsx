import { useState, useRef, useEffect } from 'react';

interface InputBarProps {
  onAsk: (question: string) => void;
  busy: boolean;
}

export default function InputBar({ onAsk, busy }: InputBarProps) {
  const [value, setValue] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = () => {
    const q = value.trim();
    if (q && !busy) {
      onAsk(q);
      setValue('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      handleSubmit();
    }
  };

  // Auto-focus input on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  return (
    <footer className="fixed bottom-0 left-0 w-full h-[80px] z-50 flex items-center px-margin bg-surface border-t border-outline-variant md:ml-[64px] lg:ml-[240px] xl:pr-[320px] xl:w-[calc(100%-320px)] md:w-[calc(100%-64px)] lg:w-[calc(100%-240px)] transition-all duration-300">
      <div className="w-full max-w-4xl mx-auto flex items-center gap-md bg-surface-container rounded-lg border border-outline-variant p-sm focus-within:border-primary focus-within:ring-1 focus-within:ring-primary transition-all relative">
        <span className="material-symbols-outlined text-on-surface-variant ml-xs shrink-0">
          terminal
        </span>
        
        <input
          ref={inputRef}
          type="text"
          value={value}
          onChange={e => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={busy}
          placeholder={busy ? "Thinking..." : "Ask anything about the SLP3 textbook..."}
          className="flex-1 bg-transparent border-none text-on-surface font-sans text-body-md focus:ring-0 placeholder:text-on-surface-variant/50 outline-none h-full py-sm disabled:opacity-50"
        />
        
        <div className="flex items-center gap-sm shrink-0">
          <span className="hidden sm:inline-flex font-mono text-mono-label text-on-surface-variant border border-outline-variant rounded px-xs py-[2px] bg-surface opacity-70">
            ⌘↵
          </span>
          <button
            onClick={handleSubmit}
            disabled={!value.trim() || busy}
            className="bg-primary-container text-on-primary-container font-mono text-mono-ui uppercase px-md py-sm rounded hover:bg-primary transition-colors flex items-center gap-xs disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Ask
            <span className="material-symbols-outlined text-[16px]">
              arrow_forward
            </span>
          </button>
        </div>
      </div>
    </footer>
  );
}
