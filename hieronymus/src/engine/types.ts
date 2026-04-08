export interface SEntropyCoords {
  S_k: number;
  S_t: number;
  S_e: number;
}

export interface ObservationResult {
  S_k: number;
  S_t: number;
  S_e: number;
  conservation: number;
  partitionDepth: number;
  sharpness: number;
  noise: number;
  coherence: number;
  visibility: number;
  elapsed_ms: number;
  partitionTexture?: Float32Array;
}

export interface MatchResult {
  score: number;
  visibility: number;
  circuits: number;
  S_distance: number;
  elapsed_ms: number;
  imageA: { S_k: number; S_t: number; S_e: number };
  imageB: { S_k: number; S_t: number; S_e: number };
}

export type WorkerInput =
  | { type: 'init' }
  | { type: 'observe'; imageData: ArrayBuffer; width: number; height: number; encoder: string }
  | { type: 'match'; imageDataA: ArrayBuffer; imageDataB: ArrayBuffer; width: number; height: number }
  | { type: 'setUniforms'; uniforms: Record<string, number> };

export type WorkerOutput =
  | { type: 'ready' }
  | { type: 'result'; payload: ObservationResult }
  | { type: 'matchResult'; payload: MatchResult }
  | { type: 'error'; message: string }
  | { type: 'progress'; stage: string; percent: number };
