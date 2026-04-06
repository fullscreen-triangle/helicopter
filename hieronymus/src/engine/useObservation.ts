'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import type { ObservationResult, MatchResult, WorkerOutput } from './types';

export function useObservation() {
  const workerRef = useRef<Worker | null>(null);
  const [ready, setReady] = useState(false);
  const [result, setResult] = useState<ObservationResult | null>(null);
  const [matchResult, setMatchResult] = useState<MatchResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const worker = new Worker(
      new URL('./observation.worker.ts', import.meta.url)
    );

    worker.onmessage = (e: MessageEvent<WorkerOutput>) => {
      switch (e.data.type) {
        case 'ready':
          setReady(true);
          setLoading(false);
          break;
        case 'result':
          setResult(e.data.payload);
          setLoading(false);
          break;
        case 'matchResult':
          setMatchResult(e.data.payload);
          setLoading(false);
          break;
        case 'error':
          setError(e.data.message);
          setLoading(false);
          break;
      }
    };

    worker.onerror = (e) => {
      setError(e.message || 'Worker error');
      setLoading(false);
    };

    workerRef.current = worker;
    worker.postMessage({ type: 'init' });
    setLoading(true);

    return () => {
      worker.terminate();
    };
  }, []);

  const observe = useCallback(
    (imageData: ImageData, encoder = 'microscopy') => {
      if (!workerRef.current || !ready) return;
      setLoading(true);
      setError(null);
      const buffer = imageData.data.buffer.slice(0);
      workerRef.current.postMessage(
        {
          type: 'observe',
          imageData: buffer,
          width: imageData.width,
          height: imageData.height,
          encoder,
        },
        [buffer]
      );
    },
    [ready]
  );

  const setUniforms = useCallback(
    (uniforms: Record<string, number>) => {
      if (!workerRef.current) return;
      workerRef.current.postMessage({ type: 'setUniforms', uniforms });
    },
    []
  );

  return { ready, loading, result, matchResult, error, observe, setUniforms };
}
