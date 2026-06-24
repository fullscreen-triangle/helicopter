'use client';

import React, { useState, useRef, useCallback } from 'react';
import { motion } from 'framer-motion';
import SceneViewer from '@/components/mic-demo/SceneViewer';
import CodeEditor from '@/components/mic-demo/CodeEditor';
import ControlPanel from '@/components/mic-demo/ControlPanel';
import {
  parseMICScript,
  generateSyntheticImage,
  estimateScaleField,
  measureWorldDistance,
  computeEntropyMetrics,
  segmentImage,
  tikhonov_deconvolve,
  type MICAnalysisResult,
  type ScaleFieldResult,
} from '@/lib/mic-engine';

const SAMPLE_CODE = `analyze {
  load channel: "synthetic"
  estimate scale_field
  visualize as: heatmap
  measure_distance from: [64, 64] to: [192, 192]
}`;

type AnalysisResult = {
  status: 'idle' | 'running' | 'complete' | 'error';
  micResult?: MICAnalysisResult;
  error?: string;
  elapsedMs?: number;
  visualMode: 'scale-field' | 'segmentation' | 'distance';
};

export default function MICDemoPage() {
  const [code, setCode] = useState(SAMPLE_CODE);
  const [result, setResult] = useState<AnalysisResult>({
    status: 'idle',
    visualMode: 'scale-field',
  });
  const sceneRef = useRef<any>(null);

  const handleRun = useCallback(async () => {
    setResult(r => ({ ...r, status: 'running' }));
    const startTime = performance.now();

    try {
      const cmd = parseMICScript(code);
      if (!cmd) throw new Error('Failed to parse analysis script. Check syntax.');

      // Generate image for the requested channel
      const img = generateSyntheticImage(256, 256, cmd.channel);

      // Algorithm 1: Spectral Scale Field Estimation (Theorem 10)
      const scaleField = estimateScaleField(img);

      // Information theory metrics (Theorems 22, 23)
      const entropy = computeEntropyMetrics(img);

      // Segmentation via level-set (Definition 14, Theorem 17)
      const segmentation = segmentImage(img);

      // Process operations from script
      let distance;
      let deconvolution;
      let visualMode: 'scale-field' | 'segmentation' | 'distance' = 'scale-field';

      for (const op of cmd.operations) {
        if (op.op === 'visualize') {
          visualMode = op.mode === 'heatmap' ? 'scale-field'
            : op.mode === 'segmentation' ? 'segmentation'
            : 'distance';
        } else if (op.op === 'measure_distance') {
          // Fast marching geodesic distance (Definition 9, Theorem 11)
          distance = measureWorldDistance(
            scaleField,
            op.from[0], op.from[1],
            op.to[0], op.to[1]
          );
          if (!cmd.operations.some(o => o.op === 'visualize')) {
            visualMode = 'distance';
          }
        } else if (op.op === 'deconvolve') {
          // Multigrid deconvolution (Algorithm 2, Theorem 20)
          deconvolution = tikhonov_deconvolve(img, 2.5, op.lambda ?? 1e-4);
        }
      }

      const elapsedMs = performance.now() - startTime;
      const micResult: MICAnalysisResult = {
        scaleField,
        distance,
        deconvolution,
        entropy,
        segmentation,
        elapsedMs,
      };

      setResult({
        status: 'complete',
        micResult,
        elapsedMs,
        visualMode,
      });

      // Push data to 3D viewer
      if (sceneRef.current) {
        const fieldData = visualMode === 'segmentation'
          ? Float32Array.from(segmentation.mask)
          : visualMode === 'distance' && distance
          ? normalizeField(distance.distanceMap)
          : normalizeField(scaleField.alpha);
        sceneRef.current.updateVisualization(fieldData, visualMode);
      }
    } catch (err) {
      setResult({
        status: 'error',
        error: err instanceof Error ? err.message : String(err),
        visualMode: 'scale-field',
      });
    }
  }, [code]);

  return (
    <div className="min-h-[calc(100vh-140px)] flex flex-col bg-[#050810]">
      {/* Header */}
      <div className="px-6 py-3 border-b border-gray-800/50 flex items-center justify-between">
        <div>
          <h1 className="text-sm font-semibold tracking-wider text-cyan-400">
            MIC ANALYZER
          </h1>
          <p className="text-[10px] text-gray-600 mt-1">
            Microscopy Image Calculus · W^&#123;1,2&#125;(Ω) · Spectral Scale Field · Geodesic Distance
          </p>
        </div>
        <div className="flex items-center gap-3">
          {result.status === 'complete' && result.micResult && (
            <div className="flex gap-3 text-[9px] text-gray-500">
              <span>
                H={result.micResult.entropy.shannonEntropy.toFixed(2)} bits
              </span>
              <span>
                SNR={result.micResult.entropy.snr.toFixed(1)}
              </span>
              <span>
                CRLB={result.micResult.entropy.crlbPixels.toFixed(3)}px
              </span>
            </div>
          )}
          <motion.button
            onClick={handleRun}
            disabled={result.status === 'running'}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className={`px-4 py-2 rounded text-sm font-semibold tracking-wider transition-colors ${
              result.status === 'running'
                ? 'bg-gray-700 text-gray-500 cursor-not-allowed'
                : 'bg-cyan-400 text-black hover:bg-cyan-300'
            }`}
          >
            {result.status === 'running' ? 'Computing...' : 'Run Analysis'}
          </motion.button>
        </div>
      </div>

      {/* Main workspace */}
      <div className="flex-1 grid grid-cols-2 overflow-hidden">
        {/* Left side: Code Editor */}
        <div className="border-r border-gray-800/50 flex flex-col overflow-hidden">
          <div className="px-4 py-3 border-b border-gray-800/50">
            <div className="text-[9px] text-gray-600 uppercase tracking-widest mb-1">
              Analysis Script
            </div>
            <div className="text-[10px] text-gray-500">
              MIC DSL · Scale Field · Geodesic Measurement
            </div>
          </div>
          <div className="flex-1 overflow-hidden">
            <CodeEditor value={code} onChange={setCode} />
          </div>
        </div>

        {/* Right side: 3D Visualization + Control Panel */}
        <div className="flex flex-col overflow-hidden">
          {/* 3D Viewport */}
          <div className="flex-1 relative overflow-hidden">
            <SceneViewer
              ref={sceneRef}
              mode={result.visualMode}
              scaleField={
                result.micResult
                  ? (result.visualMode === 'segmentation'
                      ? Float32Array.from(result.micResult.segmentation.mask)
                      : result.visualMode === 'distance' && result.micResult.distance
                      ? normalizeField(result.micResult.distance.distanceMap)
                      : normalizeField(result.micResult.scaleField.alpha))
                  : undefined
              }
            />

            {result.status === 'running' && (
              <motion.div
                className="absolute inset-0 flex items-center justify-center bg-black/50 z-10 backdrop-blur-sm"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
              >
                <div className="text-center space-y-2">
                  <div className="w-8 h-8 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin mx-auto" />
                  <div className="text-cyan-400 text-xs font-semibold">
                    Running MIC pipeline…
                  </div>
                  <div className="text-gray-500 text-[9px]">
                    Spectral scale field · Fast marching · Entropy
                  </div>
                </div>
              </motion.div>
            )}

            {result.status === 'complete' && (
              <div className="absolute bottom-3 left-3 flex gap-2">
                <span className="text-[9px] text-green-400 bg-black/70 border border-green-400/30 px-2 py-1 rounded-full">
                  ✓ Complete
                </span>
                <span className="text-[9px] text-gray-400 bg-black/70 border border-gray-800 px-2 py-1 rounded-full">
                  {result.elapsedMs?.toFixed(1)}ms
                </span>
                {result.micResult?.scaleField && (
                  <span className="text-[9px] text-cyan-400/60 bg-black/70 border border-cyan-400/20 px-2 py-1 rounded-full">
                    α={powerLawLabel(result.micResult.scaleField.powerLawExponent)}
                  </span>
                )}
              </div>
            )}

            {result.status === 'error' && (
              <div className="absolute bottom-3 left-3 text-[9px] text-red-400 bg-red-900/20 border border-red-400/30 px-2 py-1 rounded max-w-xs">
                Error: {result.error}
              </div>
            )}
          </div>

          {/* Control Panel */}
          <ControlPanel
            mode={result.visualMode}
            onModeChange={(mode) => {
              setResult(r => ({ ...r, visualMode: mode }));
              if (sceneRef.current && result.micResult) {
                const fieldData = mode === 'segmentation'
                  ? Float32Array.from(result.micResult.segmentation.mask)
                  : mode === 'distance' && result.micResult.distance
                  ? normalizeField(result.micResult.distance.distanceMap)
                  : normalizeField(result.micResult.scaleField.alpha);
                sceneRef.current.updateVisualization(fieldData, mode);
              }
            }}
            result={result.micResult as any}
          />
        </div>
      </div>
    </div>
  );
}

function normalizeField(field: Float32Array): Float32Array {
  let min = Infinity, max = -Infinity;
  for (const v of field) {
    if (isFinite(v) && v < min) min = v;
    if (isFinite(v) && v > max) max = v;
  }
  const range = max - min || 1;
  const out = new Float32Array(field.length);
  for (let i = 0; i < field.length; i++) {
    out[i] = isFinite(field[i]) ? (field[i] - min) / range : 0;
  }
  return out;
}

function powerLawLabel(exp: number): string {
  // Theorem 2: expected range [-3, 0] for smooth images
  return isFinite(exp) ? exp.toFixed(3) : 'N/A';
}
