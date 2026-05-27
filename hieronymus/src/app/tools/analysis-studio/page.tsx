'use client';

import React, { useState, useRef } from 'react';
import { motion } from 'framer-motion';
import { ChartManagerProvider, useChartManager, createChartBuilder } from '@/components/charts/ChartManager';
import ChartGrid from '@/components/charts/ChartGrid';
import AnalysisEditor from '@/components/analysis/AnalysisEditor';

const SAMPLE_SCRIPT = `// Microscopy Image Analysis Demo
// Generate analysis visualizations

const data = {
  fourier: [
    { frequency: 1, energy: 1500 },
    { frequency: 5, energy: 800 },
    { frequency: 10, energy: 450 },
    { frequency: 20, energy: 200 },
    { frequency: 40, energy: 80 },
    { frequency: 80, energy: 30 },
  ],

  scales: [
    { location: 'edge', scale: 0.45 },
    { location: 'center', scale: 0.92 },
    { location: 'corner', scale: 0.38 },
    { location: 'mid-left', scale: 0.68 },
    { location: 'mid-right', scale: 0.75 },
  ],

  distance: [
    { pair: 'nuclei-A', measured: 210, true: 212, error: 0.9 },
    { pair: 'nuclei-B', measured: 305, true: 307, error: 0.7 },
    { pair: 'mitochondria-1', measured: 45, true: 45.1, error: 0.2 },
    { pair: 'mitochondria-2', measured: 88, true: 89, error: 1.1 },
  ]
};

// Chart 1: Spectral Energy Distribution (log-log)
log('Creating spectral analysis...');
c.line('spectral')
  .title('Fourier Power Law Decay')
  .data(data.fourier)
  .x('frequency')
  .key('energy')
  .build();

// Chart 2: Scale Field Distribution
log('Estimating scale fields...');
c.bar('scale-distribution')
  .title('Metric Scale Field Estimation')
  .data(data.scales)
  .x('location')
  .key('scale')
  .build();

// Chart 3: Distance Measurement Accuracy
log('Computing distance measurements...');
c.scatter('distance-accuracy')
  .title('Distance Measurement Validation')
  .data(data.distance.map(d => ({ measured: d.measured, true: d.true, pair: d.pair })))
  .x('true')
  .y('measured')
  .build();

// Chart 4: Error Distribution
log('Analyzing error statistics...');
c.bar('error-distribution')
  .title('Relative Error by Measurement Type')
  .data(data.distance)
  .x('pair')
  .key('error')
  .build();

log('Analysis complete! Generated 4 charts.');
`;

interface AnalysisResult {
  status: 'idle' | 'running' | 'complete' | 'error';
  output: string[];
  error?: string;
  elapsedMs?: number;
}

function AnalysisStudioContent() {
  const [code, setCode] = useState(SAMPLE_SCRIPT);
  const [result, setResult] = useState<AnalysisResult>({
    status: 'idle',
    output: [],
  });
  const chartManagerCtx = useChartManager();
  const consoleRef = useRef<HTMLDivElement>(null);

  const executeScript = async () => {
    setResult({ status: 'running', output: [] });
    const startTime = performance.now();

    try {
      // Create a custom environment for the script
      const output: string[] = [];
      const log = (msg: any) => {
        const str = typeof msg === 'string' ? msg : JSON.stringify(msg);
        output.push(str);
        setResult((prev) => ({
          ...prev,
          output: [...prev.output, str],
        }));
      };

      // Create chart builder context
      const c = createChartBuilder(chartManagerCtx);

      // Create a safe eval environment
      // eslint-disable-next-line no-eval
      const fn = new Function('c', 'log', code);
      await fn(c, log);

      const elapsedMs = performance.now() - startTime;
      setResult({
        status: 'complete',
        output,
        elapsedMs,
      });
    } catch (error) {
      const elapsedMs = performance.now() - startTime;
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      setResult({
        status: 'error',
        output: [],
        error: errorMsg,
        elapsedMs,
      });
    }

    // Scroll to bottom of console
    setTimeout(() => {
      if (consoleRef.current) {
        consoleRef.current.scrollTop = consoleRef.current.scrollHeight;
      }
    }, 0);
  };

  const handleRun = async () => {
    chartManagerCtx.clearCharts();
    await executeScript();
  };

  return (
    <div className="min-h-[calc(100vh-140px)] flex flex-col bg-[#050810]">
      {/* Header */}
      <div className="px-6 py-3 border-b border-gray-800/50 flex items-center justify-between">
        <div>
          <h1 className="text-sm font-semibold tracking-wider text-cyan-400">
            ANALYSIS STUDIO
          </h1>
          <p className="text-[10px] text-gray-600 mt-1">
            Scientific Computing IDE • Progressive Chart Generation
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
          {result.status === 'running' ? 'Executing...' : 'Run Script'}
        </motion.button>
      </div>

      {/* Main workspace */}
      <div className="flex-1 grid grid-cols-2 overflow-hidden gap-0">
        {/* Left: Code Editor */}
        <div className="border-r border-gray-800/50 flex flex-col overflow-hidden">
          <div className="px-4 py-3 border-b border-gray-800/50">
            <div className="text-[9px] text-gray-600 uppercase tracking-widest mb-2">
              Analysis Script
            </div>
            <div className="text-[10px] text-gray-500">
              JavaScript • Real-time execution
            </div>
          </div>
          <AnalysisEditor value={code} onChange={setCode} />

          {/* Console Output */}
          <div className="border-t border-gray-800/50 bg-[#0f1420] overflow-hidden flex flex-col">
            <div className="px-4 py-2 border-b border-gray-800/50">
              <div className="text-[9px] text-gray-600 uppercase tracking-widest">
                Console
              </div>
            </div>
            <div
              ref={consoleRef}
              className="flex-1 overflow-y-auto p-3 font-mono text-[9px] space-y-1"
            >
              {result.output.length === 0 ? (
                <div className="text-gray-700">
                  {result.status === 'idle' && 'Ready to run...'}
                  {result.status === 'running' && 'Executing script...'}
                  {result.status === 'complete' && 'Execution completed.'}
                </div>
              ) : (
                result.output.map((line, i) => (
                  <div key={i} className="text-gray-400">
                    {'> '} {line}
                  </div>
                ))
              )}
              {result.error && (
                <div className="text-red-400 mt-2">
                  <div className="font-semibold">ERROR</div>
                  <div>{result.error}</div>
                </div>
              )}
              {result.status === 'complete' && result.elapsedMs && (
                <div className="text-green-400 mt-2 pt-2 border-t border-gray-800">
                  ✓ Complete in {result.elapsedMs.toFixed(1)}ms
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right: Charts */}
        <div className="overflow-hidden">
          <ChartGrid />
        </div>
      </div>
    </div>
  );
}

// Wrapper with ChartManager
export default function AnalysisStudioPage() {
  return (
    <ChartManagerProvider>
      <AnalysisStudioContent />
    </ChartManagerProvider>
  );
}
