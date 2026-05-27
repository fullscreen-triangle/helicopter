'use client';

import React, { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import SceneViewer from '@/components/mic-demo/SceneViewer';
import CodeEditor from '@/components/mic-demo/CodeEditor';
import ControlPanel from '@/components/mic-demo/ControlPanel';

const SAMPLE_CODE = `analyze {
  load channel: "synthetic"
  estimate scale_field
  visualize as: heatmap
  measure_distance from: [64, 64] to: [192, 192]
}`;

type AnalysisResult = {
  status: 'idle' | 'running' | 'complete' | 'error';
  distance?: number;
  scaleField?: Float32Array;
  error?: string;
  elapsedMs?: number;
};

export default function MICDemoPage() {
  const [code, setCode] = useState(SAMPLE_CODE);
  const [result, setResult] = useState<AnalysisResult>({ status: 'idle' });
  const [visualizationMode, setVisualizationMode] = useState<'scale-field' | 'segmentation' | 'distance'>('scale-field');
  const sceneRef = useRef<any>(null);

  const handleRun = async () => {
    setResult({ status: 'running' });
    const startTime = performance.now();

    try {
      // Simulate compilation and execution
      // In real implementation, this would call TypeScript compiler
      // and generate GLSL shaders

      await new Promise(resolve => setTimeout(resolve, 500));

      const elapsedMs = performance.now() - startTime;

      // Generate dummy scale field for visualization
      const scaleField = new Float32Array(256 * 256);
      for (let i = 0; i < scaleField.length; i++) {
        const x = (i % 256) / 256;
        const y = Math.floor(i / 256) / 256;
        // Gaussian-like distribution
        scaleField[i] = 0.5 + 0.3 * Math.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) * 10);
      }

      setResult({
        status: 'complete',
        distance: 181.9,
        scaleField,
        elapsedMs,
      });

      if (sceneRef.current) {
        sceneRef.current.updateVisualization(scaleField, visualizationMode);
      }
    } catch (error) {
      setResult({
        status: 'error',
        error: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  };

  return (
    <div className="min-h-[calc(100vh-140px)] flex flex-col bg-[#050810]">
      {/* Header */}
      <div className="px-6 py-3 border-b border-gray-800/50 flex items-center justify-between">
        <div>
          <h1 className="text-sm font-semibold tracking-wider text-cyan-400">
            MIC ANALYZER
          </h1>
          <p className="text-[10px] text-gray-600 mt-1">
            Microscopy Image Calculus • Interactive Sandbox
          </p>
        </div>
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
          {result.status === 'running' ? 'Compiling...' : 'Run Analysis'}
        </motion.button>
      </div>

      {/* Main workspace */}
      <div className="flex-1 grid grid-cols-2 overflow-hidden">
        {/* Left side: Code Editor */}
        <div className="border-r border-gray-800/50 flex flex-col overflow-hidden">
          <div className="px-4 py-3 border-b border-gray-800/50">
            <div className="text-[9px] text-gray-600 uppercase tracking-widest mb-2">
              Analysis Script
            </div>
            <div className="text-[10px] text-gray-500">
              TypeScript DSL • Real-time compilation
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
              mode={visualizationMode}
              scaleField={result.scaleField}
            />

            {/* Loading overlay */}
            {result.status === 'running' && (
              <motion.div
                className="absolute inset-0 flex items-center justify-center bg-black/50 z-10 backdrop-blur-sm"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
              >
                <div className="text-center">
                  <div className="w-8 h-8 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
                  <div className="text-cyan-400 text-sm font-semibold">Compiling shaders...</div>
                </div>
              </motion.div>
            )}

            {/* Status indicator */}
            {result.status === 'complete' && (
              <div className="absolute bottom-3 left-3 flex gap-2">
                <span className="text-[9px] text-green-400 bg-black/70 border border-green-400/30 px-2 py-1 rounded-full">
                  ✓ Complete
                </span>
                <span className="text-[9px] text-gray-400 bg-black/70 border border-gray-800 px-2 py-1 rounded-full">
                  {result.elapsedMs?.toFixed(1)}ms
                </span>
              </div>
            )}

            {result.status === 'error' && (
              <div className="absolute bottom-3 left-3 text-[9px] text-red-400 bg-red-900/20 border border-red-400/30 px-2 py-1 rounded">
                Error: {result.error}
              </div>
            )}
          </div>

          {/* Control Panel */}
          <ControlPanel
            mode={visualizationMode}
            onModeChange={setVisualizationMode}
            result={result}
          />
        </div>
      </div>
    </div>
  );
}
