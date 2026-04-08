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

  const match = useCallback(
    (imageDataA: ImageData, imageDataB: ImageData) => {
      if (!workerRef.current || !ready) return;
      setLoading(true);
      setError(null);
      setMatchResult(null);

      // Resize both images to the same dimensions (use the smaller of each axis)
      const w = Math.min(imageDataA.width, imageDataB.width);
      const h = Math.min(imageDataA.height, imageDataB.height);

      const resizeToBuffer = (img: ImageData, tw: number, th: number): ArrayBuffer => {
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d')!;
        ctx.putImageData(img, 0, 0);

        const outCanvas = document.createElement('canvas');
        outCanvas.width = tw;
        outCanvas.height = th;
        const outCtx = outCanvas.getContext('2d')!;
        outCtx.drawImage(canvas, 0, 0, tw, th);
        return outCtx.getImageData(0, 0, tw, th).data.buffer.slice(0);
      };

      const bufA = resizeToBuffer(imageDataA, w, h);
      const bufB = resizeToBuffer(imageDataB, w, h);

      workerRef.current.postMessage(
        {
          type: 'match',
          imageDataA: bufA,
          imageDataB: bufB,
          width: w,
          height: h,
        },
        [bufA, bufB]
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

  return { ready, loading, result, matchResult, error, observe, match, setUniforms };
}
