/// <reference lib="webworker" />

import type { WorkerInput, WorkerOutput } from './types';

// Dynamic import to avoid bundling issues with WebGL in worker context
let engine: import('./ObservationEngine').ObservationEngine | null = null;

self.onmessage = async (e: MessageEvent<WorkerInput>) => {
  const msg = e.data;

  try {
    switch (msg.type) {
      case 'init': {
        if (typeof OffscreenCanvas === 'undefined') {
          throw new Error(
            'OffscreenCanvas is not supported in this browser. Please use Chrome, Edge, or Firefox.'
          );
        }
        const { ObservationEngine } = await import('./ObservationEngine');
        const canvas = new OffscreenCanvas(512, 512);
        engine = new ObservationEngine(canvas, 512, 512);
        self.postMessage({ type: 'ready' } as WorkerOutput);
        break;
      }
      case 'observe': {
        if (!engine) throw new Error('Engine not initialized');
        const result = engine.observe(
          new Uint8Array(msg.imageData),
          msg.width,
          msg.height,
          msg.encoder
        );
        // Don't transfer partitionTexture to keep it simple
        const { partitionTexture, ...metrics } = result;
        self.postMessage({
          type: 'result',
          payload: { ...metrics, partitionTexture: undefined },
        } as WorkerOutput);
        break;
      }
      case 'match': {
        if (!engine) throw new Error('Engine not initialized');
        const result = engine.match(
          new Uint8Array(msg.imageDataA),
          new Uint8Array(msg.imageDataB),
          msg.width,
          msg.height
        );
        self.postMessage({
          type: 'matchResult',
          payload: result,
        } as WorkerOutput);
        break;
      }
      case 'setUniforms': {
        engine?.setUniforms(msg.uniforms);
        break;
      }
    }
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    self.postMessage({ type: 'error', message } as WorkerOutput);
  }
};
