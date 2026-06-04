'use client';

import React, { useState, useRef } from 'react';
import { motion } from 'framer-motion';
import { compileScope } from '@/lib/scope-compiler';
import { executeSCOPE } from '@/lib/scope-runtime';
import { generateTimingEvents } from '@/lib/scope-client';
import ScopeIDELayout from './ScopeIDELayout';

function AnalysisStudioContent() {
  const [code, setCode] = useState('');
  const [output, setOutput] = useState<string[]>([]);
  const [isExecuting, setIsExecuting] = useState(false);

  const handleCodeChange = (newCode: string) => {
    setCode(newCode);
  };

  const handleFileSelect = (path: string, fileCode: string) => {
    setCode(fileCode);
    setOutput([`Loaded: ${path}`]);
  };

  const handleCompileAndExecute = async () => {
    setIsExecuting(true);
    setOutput([]);
    const logs: string[] = ['Compiling SCOPE program...'];

    try {
      // Compile
      const compiled = compileScope(code);

      if (compiled.errors.length > 0) {
        logs.push('❌ Compilation failed:');
        compiled.errors.forEach((err) => {
          logs.push(`  Line ${err.line}: ${err.message}`);
        });
        setOutput(logs);
        setIsExecuting(false);
        return;
      }

      logs.push(`✓ Compilation successful`);
      logs.push(`Program: ${compiled.name}`);

      if (compiled.warnings.length > 0) {
        logs.push('⚠ Warnings:');
        compiled.warnings.forEach((warn) => {
          logs.push(`  ${warn.message}`);
        });
      }

      logs.push('');
      logs.push('Executing SCOPE program...');
      setOutput([...logs]);

      // Execute
      const timingEvents = generateTimingEvents('METAPHASE', 1000);
      logs.push(`Generated ${timingEvents.length} timing events`);

      const result = await executeSCOPE(compiled.ir, timingEvents, 'synthetic');

      if (result.success) {
        logs.push(...result.logs);
        if (result.output?.result) {
          logs.push('');
          logs.push('═══ RESULT ═══');
          logs.push(`Structure: ${result.output.result.structure}`);
          if (result.output.result.distance) {
            logs.push(`Distance: ${result.output.result.distance.toExponential(3)} m`);
            logs.push(`Uncertainty: ±${result.output.result.uncertainty.toExponential(3)} m`);
          }
          logs.push(
            `Position: (${result.output.result.position.x.toFixed(3)}, ${result.output.result.position.y.toFixed(3)}, ${result.output.result.position.z.toFixed(3)})`
          );
          logs.push(
            `S-Entropy: S_k=${result.output.result.s_entropy.S_k.toFixed(3)} S_t=${result.output.result.s_entropy.S_t.toExponential(1)} S_e=${result.output.result.s_entropy.S_e.toFixed(3)}`
          );
        }
        logs.push(`✓ Complete in ${result.timing_ms.toFixed(1)}ms`);
      } else {
        logs.push(`❌ Execution failed: ${result.error}`);
      }

      setOutput([...logs]);
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : String(error);
      logs.push(`❌ Error: ${errorMsg}`);
      setOutput([...logs]);
    }

    setIsExecuting(false);
  };

  return (
    <div className="min-h-[calc(100vh-140px)] flex flex-col bg-[#050810]">
      {/* Toolbar */}
      <div className="px-6 py-2 border-b border-gray-800/50 flex items-center gap-3 bg-[#0a0e18]">
        <motion.button
          onClick={handleCompileAndExecute}
          disabled={isExecuting || !code.trim()}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          className={`px-4 py-2 rounded text-sm font-semibold tracking-wider transition-colors ${
            isExecuting || !code.trim()
              ? 'bg-gray-700 text-gray-500 cursor-not-allowed'
              : 'bg-cyan-400 text-black hover:bg-cyan-300'
          }`}
        >
          {isExecuting ? 'Executing...' : 'Compile & Execute'}
        </motion.button>
        <div className="text-[9px] text-gray-500">
          {isExecuting ? 'Processing...' : 'Ready'}
        </div>
      </div>

      {/* IDE Layout */}
      <div className="flex-1 overflow-hidden">
        <ScopeIDELayout
          onCodeChange={handleCodeChange}
          onFileSelect={handleFileSelect}
          output={output}
          isExecuting={isExecuting}
        />
      </div>
    </div>
  );
}

export default function AnalysisStudioPage() {
  return <AnalysisStudioContent />;
}
