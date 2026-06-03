/**
 * SCOPE Backend Client
 *
 * TypeScript client for communicating with the SCOPE Python backend.
 * Handles program listing, execution, and result serialization.
 */

const SCOPE_API_BASE = process.env.NEXT_PUBLIC_SCOPE_API || 'http://localhost:5000';

export interface TimingEvent {
  delta_p: number;  // Time deviation in seconds
  channel_id: number;
  intensity?: number;
}

export interface FrameData {
  data: string;  // Base64-encoded frame bytes
  shape: [number, number];  // [height, width]
  dtype: string;  // e.g., "float32"
}

export interface PartitionState {
  n: number;
  'ℓ': number;
  m: number;
  s: number;
}

export interface SEntropy {
  S_k: number;
  S_t: number;
  S_e: number;
}

export interface SCOPEResult {
  structure: string | null;
  position: {
    x: number;
    y: number;
    z: number;
  };
  distance: number | null;
  uncertainty: number;
  s_entropy: SEntropy;
  partition_state: PartitionState;
}

export interface ExecutionResponse {
  success: boolean;
  result?: SCOPEResult;
  timing_ms?: number;
  error?: string;
}

export interface ProgramInfo {
  id: string;
  name: string;
  depth: number;
  field_size: {
    x: number;
    y: number;
  };
  resolution: number;
  morphisms: string[];
  timing_cells: Array<{
    cell_id: string;
    bounds_delta_p: [number, number];
  }>;
}

export interface ProgramsListResponse {
  programs: ProgramInfo[];
  count: number;
}

/**
 * SCOPE Client for backend communication
 */
export class SCOPEClient {
  private baseUrl: string;

  constructor(baseUrl?: string) {
    this.baseUrl = baseUrl || SCOPE_API_BASE;
  }

  /**
   * Check if backend is available
   */
  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/health`);
      return response.ok;
    } catch (error) {
      console.error('Health check failed:', error);
      return false;
    }
  }

  /**
   * List all available SCOPE programs
   */
  async listPrograms(): Promise<ProgramInfo[]> {
    try {
      const response = await fetch(`${this.baseUrl}/programs`);
      const data = (await response.json()) as ProgramsListResponse;
      return data.programs;
    } catch (error) {
      console.error('Failed to list programs:', error);
      throw error;
    }
  }

  /**
   * Get details of a specific program
   */
  async getProgram(programId: string): Promise<Omit<ProgramInfo, 'morphisms' | 'timing_cells'>> {
    try {
      const response = await fetch(`${this.baseUrl}/programs/${programId}`);
      if (!response.ok) {
        throw new Error(`Program not found: ${programId}`);
      }
      return await response.json();
    } catch (error) {
      console.error(`Failed to get program ${programId}:`, error);
      throw error;
    }
  }

  /**
   * Execute a SCOPE program
   */
  async execute(
    programId: string,
    timingEvents: TimingEvent[],
    frame: FrameData
  ): Promise<ExecutionResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/execute`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          program_id: programId,
          timing_events: timingEvents,
          frame: frame,
        }),
      });

      const data = (await response.json()) as ExecutionResponse;
      return data;
    } catch (error) {
      console.error('Execution failed:', error);
      throw error;
    }
  }

  /**
   * Execute a SCOPE program on multiple event streams (batch)
   */
  async executeBatch(
    programId: string,
    timingEventsList: TimingEvent[][],
    frame: FrameData
  ): Promise<{
    success: boolean;
    results?: SCOPEResult[];
    count?: number;
    timing_ms?: number;
    error?: string;
  }> {
    try {
      const response = await fetch(`${this.baseUrl}/execute-batch`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          program_id: programId,
          timing_events_list: timingEventsList,
          frame: frame,
        }),
      });

      return await response.json();
    } catch (error) {
      console.error('Batch execution failed:', error);
      throw error;
    }
  }

  /**
   * Encode a frame for API transmission
   */
  static encodeFrame(frameData: Float32Array, shape: [number, number]): FrameData {
    // Convert Float32Array to base64
    const buffer = new ArrayBuffer(frameData.length * 4);
    const view = new Float32Array(buffer);
    view.set(frameData);

    const bytes = new Uint8Array(buffer);
    let binaryString = '';
    for (let i = 0; i < bytes.length; i++) {
      binaryString += String.fromCharCode(bytes[i]);
    }

    const base64 = btoa(binaryString);

    return {
      data: base64,
      shape: shape,
      dtype: 'float32',
    };
  }

  /**
   * Decode a result from the backend
   */
  static decodeResult(raw: any): SCOPEResult {
    return {
      structure: raw.structure,
      position: {
        x: raw.position.x,
        y: raw.position.y,
        z: raw.position.z,
      },
      distance: raw.distance,
      uncertainty: raw.uncertainty,
      s_entropy: raw.s_entropy,
      partition_state: raw.partition_state,
    };
  }
}

/**
 * Generate synthetic timing events for testing
 */
export function generateTimingEvents(
  phase: 'PROPHASE' | 'METAPHASE' | 'ANAPHASE',
  count: number = 1000
): TimingEvent[] {
  const phaseMeans: Record<string, number> = {
    PROPHASE: -1.4e-6,
    METAPHASE: 0.0e-6,
    ANAPHASE: 1.4e-6,
  };

  const mean = phaseMeans[phase] || 0;
  const sigma = 0.3e-6;

  const events: TimingEvent[] = [];
  for (let i = 0; i < count; i++) {
    // Box-Muller transform for normal distribution
    const u1 = Math.random();
    const u2 = Math.random();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    const deltaP = mean + sigma * z;

    events.push({
      delta_p: deltaP,
      channel_id: i % 2,
      intensity: Math.random() * 200,
    });
  }

  return events;
}

/**
 * Generate a synthetic frame for testing
 */
export function generateSyntheticFrame(
  width: number = 1024,
  height: number = 1024,
  numNuclei: number = 2
): Float32Array {
  const frame = new Float32Array(width * height);

  // Add Gaussian blobs for nuclei
  for (let n = 0; n < numNuclei; n++) {
    const cy = Math.random() * (height - 200) + 100;
    const cx = Math.random() * (width - 200) + 100;
    const sigma = Math.random() * 30 + 20;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const dx = x - cx;
        const dy = y - cy;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const blob = 1000 * Math.exp(-(dist * dist) / (2 * sigma * sigma));
        frame[y * width + x] += blob;
      }
    }
  }

  // Add background noise
  for (let i = 0; i < frame.length; i++) {
    frame[i] += Math.random() * 20;
    frame[i] = Math.max(0, Math.min(2000, frame[i]));
  }

  return frame;
}
