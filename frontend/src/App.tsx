import { useState, useRef, useEffect } from 'react';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import SourcesPanel from './components/SourcesPanel';
import QueryCard from './components/QueryCard';
import InputBar from './components/InputBar';
import { useHealth } from './hooks/useHealth';
import { queryAPI, triggerReindex } from './api';
import type { QueryResult } from './types';

export default function App() {
  const { health, triggerRefresh } = useHealth();
  const [queries, setQueries] = useState<QueryResult[]>([]);
  const [isAsking, setIsAsking] = useState(false);
  
  // Ref for auto-scrolling to bottom
  const scrollRef = useRef<HTMLDivElement>(null);

  const handleAsk = async (question: string) => {
    setIsAsking(true);
    
    // Add a temporary "loading" query to the UI? No, we just show "Thinking..." in the input bar
    // and wait for the response, then append.
    
    try {
      const response = await queryAPI(question);
      
      const newQuery: QueryResult = {
        id: crypto.randomUUID(),
        question: response.question,
        answer: response.answer,
        provider: response.provider,
        supported: response.supported,
        gated: response.gated,
        source_chunk_ids: response.source_chunk_ids,
        sources: response.sources,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
      
      setQueries(prev => [...prev, newQuery]);
    } catch (err) {
      // In a real app we'd show a toast or error card. Here we'll append an error card.
      const errorMessage = err instanceof Error ? err.message : 'An unknown error occurred.';
      const errorQuery: QueryResult = {
        id: crypto.randomUUID(),
        question,
        answer: `❌ Error: ${errorMessage}`,
        provider: 'system',
        supported: false,
        gated: false,
        source_chunk_ids: [],
        sources: [],
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
      setQueries(prev => [...prev, errorQuery]);
    } finally {
      setIsAsking(false);
    }
  };

  const handleReindex = async () => {
    try {
      await triggerReindex();
      triggerRefresh();
    } catch (err) {
      console.error(err);
      alert('Failed to trigger re-indexing');
    }
  };

  // Auto-scroll on new queries
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTo({
        top: scrollRef.current.scrollHeight,
        behavior: 'smooth'
      });
    }
  }, [queries.length]);

  // Determine sources for the right panel (take sources from the latest query)
  const latestQuery = queries.length > 0 ? queries[queries.length - 1] : null;
  const currentSources = latestQuery?.sources || [];
  const currentSourceCount = latestQuery?.source_chunk_ids?.length || 0;

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-background">
      <Header health={health} onReindex={handleReindex} />

      <div className="flex flex-1 overflow-hidden relative">
        <Sidebar />

        <main
          ref={scrollRef}
          className="flex-1 flex flex-col md:ml-[64px] lg:ml-[240px] xl:mr-[320px] bg-background relative overflow-y-auto h-[calc(100vh-128px)] transition-all duration-300"
        >
          <div className="max-w-4xl w-full mx-auto p-gutter pb-[100px] flex flex-col gap-lg">
            {queries.length === 0 ? (
              <div className="flex-1 flex flex-col items-center justify-center h-full min-h-[40vh] text-center opacity-50 mt-xl">
                <span className="material-symbols-outlined text-[48px] mb-sm text-primary">
                  library_books
                </span>
                <h2 className="font-playfair text-heading text-on-surface mb-xs">
                  Welcome to the Scholar's Terminal
                </h2>
                <p className="font-sans text-body-md text-on-surface-variant max-w-md">
                  Ask questions about the SLP3 corpus. Your queries will be grounded in the text and cited accurately.
                </p>
              </div>
            ) : (
              queries.map((q, idx) => (
                <QueryCard
                  key={q.id}
                  query={q}
                  animate={idx === queries.length - 1} // Only animate the newest one
                />
              ))
            )}
            
            {isAsking && (
              <div className="flex items-center gap-sm p-lg text-on-surface-variant animate-pulse">
                <span className="material-symbols-outlined spin">autorenew</span>
                <span className="font-mono text-mono-label">Retrieving and synthesizing...</span>
              </div>
            )}
          </div>
        </main>

        <SourcesPanel sources={currentSources} count={currentSourceCount} />
      </div>

      <InputBar onAsk={handleAsk} busy={isAsking} />
    </div>
  );
}
