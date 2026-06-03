'use client';

import React, { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { compileScope, CompiledProgram } from '@/lib/scope-compiler';
import { executeSCOPE, ExecutionResult, MicroscopyDatabaseClient, HuggingFaceClient, ReactomeClient } from '@/lib/scope-runtime';
import { generateTimingEvents, generateSyntheticFrame } from '@/lib/scope-client';
import { getScopeExample, getExampleDescription } from '@/lib/scope-examples';
import AnalysisEditor from '@/components/analysis/AnalysisEditor';

// Get initial example based on default selections
const getInitialCode = () => getScopeExample('PROPHASE', 'synthetic');

interface Result {
  status: 'idle' | 'compiling' | 'executing' | 'success' | 'error';
  compiled?: CompiledProgram;
  executed?: ExecutionResult;
  output: string[];
  error?: string;
  elapsedMs?: number;
}

function AnalysisStudioContent() {
  const [selectedPhase, setSelectedPhase] = useState<'PROPHASE' | 'METAPHASE' | 'ANAPHASE'>('PROPHASE');
  const [dataSource, setDataSource] = useState<'synthetic' | 'microscopy' | 'huggingface' | 'reactome'>('synthetic');
  const [code, setCode] = useState(() => getScopeExample(selectedPhase, dataSource));
  const [result, setResult] = useState<Result>({
    status: 'idle',
    output: [],
  });
  const consoleRef = useRef<HTMLDivElement>(null);

  // Update code when phase or data source changes
  const handlePhaseChange = (phase: 'PROPHASE' | 'METAPHASE' | 'ANAPHASE') => {
    setSelectedPhase(phase);
    setCode(getScopeExample(phase, dataSource));
  };

  const handleDataSourceChange = (source: 'synthetic' | 'microscopy' | 'huggingface' | 'reactome') => {
    setDataSource(source);
    setCode(getScopeExample(selectedPhase, source));
  };

  const handleCompileAndExecute = async () => {
    setResult({ status: 'compiling', output: [] });
    const startTime = performance.now();

    try {
      const output: string[] = [];

      // Step 1: Compile
      output.push('Compiling SCOPE program...');
      const compiled = compileScope(code);

      if (compiled.errors.length > 0) {
        output.push('❌ Compilation failed:');
        compiled.errors.forEach((err) => {
          output.push(`  Line ${err.line}: ${err.message}`);
        });
        setResult({
          status: 'error',
          compiled,
          output,
          error: `${compiled.errors.length} compilation error(s)`,
          elapsedMs: performance.now() - startTime,
        });
        return;
      }

      output.push(`✓ Compilation successful`);
      output.push(`Program: ${compiled.name}`);

      if (compiled.warnings.length > 0) {
        output.push('⚠ Warnings:');
        compiled.warnings.forEach((warn) => {
          output.push(`  ${warn.message}`);
        });
      }

      output.push('');
      output.push('Executing SCOPE program...');
      setResult({ status: 'executing', compiled, output });

      // Step 2: Generate timing events for selected phase
      const timingEvents = generateTimingEvents(selectedPhase, 1000);
      output.push(`Generated ${timingEvents.length} timing events for ${selectedPhase}`);

      // Step 3: Execute with TypeScript runtime
      const executionResult = await executeSCOPE(compiled.ir, timingEvents);

      if (executionResult.success) {
        output.push(...executionResult.logs);
        setResult({
          status: 'success',
          compiled,
          executed: executionResult,
          output,
          elapsedMs: performance.now() - startTime,
        });
      } else {
        output.push(`❌ Execution failed: ${executionResult.error}`);
        setResult({
          status: 'error',
          compiled,
          output,
          error: executionResult.error,
          elapsedMs: performance.now() - startTime,
        });
      }
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : String(error);
      const output = [`Error: ${errorMsg}`];
      setResult({
        status: 'error',
        output,
        error: errorMsg,
        elapsedMs: performance.now() - startTime,
      });
    }

    setTimeout(() => {
      if (consoleRef.current) {
        consoleRef.current.scrollTop = consoleRef.current.scrollHeight;
      }
    }, 0);
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
            TypeScript SCOPE Runtime • Microscopy Analysis Execution Engine
          </p>
        </div>

        {/* Execute button */}
        <motion.button
          onClick={handleCompileAndExecute}
          disabled={result.status === 'compiling' || result.status === 'executing'}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          className={`px-4 py-2 rounded text-sm font-semibold tracking-wider transition-colors ${
            result.status === 'compiling' || result.status === 'executing'
              ? 'bg-gray-700 text-gray-500 cursor-not-allowed'
              : 'bg-cyan-400 text-black hover:bg-cyan-300'
          }`}
        >
          {result.status === 'compiling' ? 'Compiling...' : result.status === 'executing' ? 'Executing...' : 'Compile & Execute'}
        </motion.button>
      </div>

      {/* Main workspace */}
      <div className="flex-1 grid grid-cols-2 overflow-hidden gap-0">
        {/* Left: Editor + Controls */}
        <div className="border-r border-gray-800/50 flex flex-col overflow-hidden">
          <div className="px-4 py-3 border-b border-gray-800/50 space-y-1">
            <div className="text-[9px] text-gray-600 uppercase tracking-widest mb-2">
              SCOPE Program
            </div>
            <div className="text-[10px] text-gray-500">
              {getExampleDescription(selectedPhase, dataSource)}
            </div>
            <div className="text-[9px] text-gray-700">
              Edit the program, or change Phase/Data Source to see different examples
            </div>
          </div>

          <div className="flex-1 overflow-hidden">
            <AnalysisEditor value={code} onChange={setCode} />
          </div>

          {/* Controls */}
          <div className="border-t border-gray-800/50 bg-[#0f1420] p-4 space-y-3 overflow-y-auto max-h-48">
            {/* Phase selector */}
            <div>
              <label className="text-[9px] text-gray-600 uppercase tracking-widest block mb-2">
                Cell Cycle Phase
              </label>
              <div className="space-y-1">
                {(['PROPHASE', 'METAPHASE', 'ANAPHASE'] as const).map((phase) => (
                  <button
                    key={phase}
                    onClick={() => handlePhaseChange(phase)}
                    className={`w-full px-2 py-1 rounded text-xs transition-colors text-left ${
                      selectedPhase === phase
                        ? 'bg-cyan-400/20 text-cyan-400 border border-cyan-400'
                        : 'bg-gray-800/30 text-gray-400 border border-gray-800 hover:border-gray-700'
                    }`}
                  >
                    {phase}
                  </button>
                ))}
              </div>
            </div>

            {/* Data source selector */}
            <div>
              <label className="text-[9px] text-gray-600 uppercase tracking-widest block mb-2">
                Data Source
              </label>
              <div className="space-y-1">
                {(
                  [
                    { id: 'synthetic', label: 'Synthetic Data' },
                    { id: 'microscopy', label: 'Microscopy (BBBC)' },
                    { id: 'huggingface', label: 'HuggingFace Models' },
                    { id: 'reactome', label: 'Reactome Pathways' },
                  ] as const
                ).map((source) => (
                  <button
                    key={source.id}
                    onClick={() => handleDataSourceChange(source.id)}
                    className={`w-full px-2 py-1 rounded text-xs transition-colors text-left ${
                      dataSource === source.id
                        ? 'bg-cyan-400/20 text-cyan-400 border border-cyan-400'
                        : 'bg-gray-800/30 text-gray-400 border border-gray-800 hover:border-gray-700'
                    }`}
                  >
                    {source.label}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Right: Results Console */}
        <div className="border-l border-gray-800/50 flex flex-col overflow-hidden">
          {/* Result Summary */}
          {result.status === 'success' && result.executed?.output?.result && (
            <div className="p-4 border-b border-gray-800/50 overflow-auto">
              <div className="text-[9px] text-gray-600 uppercase tracking-widest mb-3">
                Execution Result
              </div>
              <div className="space-y-2 text-[8px] text-gray-400 font-mono">
                <div>
                  <span className="text-gray-600">Structure:</span>{' '}
                  <span className="text-cyan-400">{result.executed.output.result.structure}</span>
                </div>
                {result.executed.output.result.distance && (
                  <>
                    <div>
                      <span className="text-gray-600">Distance:</span>{' '}
                      <span className="text-cyan-400">
                        {result.executed.output.result.distance.toExponential(3)}m
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-600">Uncertainty:</span>{' '}
                      <span className="text-cyan-400">
                        ±{result.executed.output.result.uncertainty.toExponential(3)}m
                      </span>
                    </div>
                  </>
                )}
                <div>
                  <span className="text-gray-600">Position:</span>{' '}
                  <span className="text-cyan-400">
                    ({result.executed.output.result.position.x.toFixed(3)},{' '}
                    {result.executed.output.result.position.y.toFixed(3)},{' '}
                    {result.executed.output.result.position.z.toFixed(3)})
                  </span>
                </div>
                <div className="pt-2 border-t border-gray-800">
                  <span className="text-gray-600">S-Entropy:</span>{' '}
                  <span className="text-cyan-400">
                    S_k={result.executed.output.result.s_entropy.S_k.toFixed(3)} S_t=
                    {result.executed.output.result.s_entropy.S_t.toExponential(1)} S_e=
                    {result.executed.output.result.s_entropy.S_e.toFixed(3)}
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* Console Output */}
          <div className="flex-1 overflow-hidden flex flex-col">
            <div className="px-4 py-2 border-b border-gray-800/50">
              <div className="text-[9px] text-gray-600 uppercase tracking-widest">
                Execution Log
              </div>
            </div>
            <div
              ref={consoleRef}
              className="flex-1 overflow-y-auto p-3 font-mono text-[9px] space-y-1"
            >
              {result.output.length === 0 ? (
                <div className="text-gray-700">Ready to execute...</div>
              ) : (
                result.output.map((line, i) => (
                  <div
                    key={i}
                    className={
                      line.startsWith('✓')
                        ? 'text-green-400'
                        : line.startsWith('❌')
                          ? 'text-red-400'
                          : line.startsWith('⚠')
                            ? 'text-yellow-400'
                            : 'text-gray-400'
                    }
                  >
                    {line}
                  </div>
                ))
              )}
              {result.status === 'success' && result.elapsedMs && (
                <div className="text-green-400 mt-2 pt-2 border-t border-gray-800">
                  ✓ Complete in {result.elapsedMs.toFixed(1)}ms
                </div>
              )}
              {result.error && (
                <div className="text-red-400 mt-2 pt-2 border-t border-gray-800">
                  Error: {result.error}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function AnalysisStudioPage() {
  return <AnalysisStudioContent />;
}
