'use client';

import React, { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import TextDropZone from '@/components/TextDropZone';
import DomainSelector from '@/components/DomainSelector';
import ObservationPanel from '@/components/ObservationPanel';
import SimplexTriangle from '@/components/SimplexTriangle';
import { useObservation } from '@/engine/useObservation';
import { encodeGenomic, GENOMIC_EXAMPLE } from '@/encoders/genomic';

function UniformSlider({
  label,
  id,
  value,
  min,
  max,
  step,
  decimals,
  onChange,
}: {
  label: string;
  id: string;
  value: number;
  min: number;
  max: number;
  step: number;
  decimals: number;
  onChange: (id: string, val: number) => void;
}) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs">
        <span className="text-gray-500">{label}</span>
        <span className="text-gray-300">{value.toFixed(decimals)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(id, parseFloat(e.target.value))}
        className="w-full h-0.5 bg-gray-700 rounded appearance-none cursor-pointer
                   [&::-webkit-slider-thumb]:appearance-none
                   [&::-webkit-slider-thumb]:w-2.5
                   [&::-webkit-slider-thumb]:h-2.5
                   [&::-webkit-slider-thumb]:rounded-full
                   [&::-webkit-slider-thumb]:bg-green-400
                   [&::-webkit-slider-thumb]:cursor-pointer"
      />
    </div>
  );
}

export default function GenomicPage() {
  const { ready, loading, result, error, observe, setUniforms } = useObservation();

  const [uniforms, setUniformState] = useState({
    epsilon: 0.15,
    nmax: 8,
    beta: 2.0,
    J: 1.0,
    Aeg: 2.58,
    alpha: 0.5,
  });

  const [textInput, setTextInput] = useState('');
  const [encodingInfo, setEncodingInfo] = useState<string | null>(null);
  const [encodeError, setEncodeError] = useState<string | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);

  const handleUniformChange = useCallback(
    (id: string, val: number) => {
      setUniformState((prev) => {
        const next = { ...prev, [id]: val };
        setUniforms(next);
        return next;
      });
    },
    [setUniforms]
  );

  const processData = useCallback(
    (text: string) => {
      setTextInput(text);
      setEncodeError(null);
      try {
        const encoded = encodeGenomic(text);
        setEncodingInfo(
          `${encoded.metadata.originalSize} nt | CPU encode: ${encoded.metadata.encodingTime.toFixed(1)}ms`
        );

        const canvas = document.createElement('canvas');
        canvas.width = encoded.imageData.width;
        canvas.height = encoded.imageData.height;
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.putImageData(encoded.imageData, 0, 0);
          setImagePreview(canvas.toDataURL());
        }

        observe(encoded.imageData, 'genomic');
      } catch (err) {
        setEncodeError((err as Error).message);
      }
    },
    [observe]
  );

  const handleLoadExample = useCallback(() => {
    processData(GENOMIC_EXAMPLE);
  }, [processData]);

  return (
    <div className="min-h-[calc(100vh-140px)] flex flex-col">
      {/* Header */}
      <div className="px-6 py-3 border-b border-gray-800/50 flex flex-col gap-3">
        <div className="flex items-center gap-4">
          <h1 className="text-sm font-semibold tracking-wider text-green-400">
            GENOMIC ENCODER
          </h1>
          <span className="text-[10px] text-gray-600 tracking-widest">
            K-MER FREQUENCY ANALYSIS
          </span>
          <div className="ml-auto flex items-center gap-3">
            {!ready && !error && (
              <span className="text-[10px] text-amber-400 animate-pulse">
                Initializing GPU engine...
              </span>
            )}
            {ready && (
              <span className="text-[10px] text-green-400">Engine ready</span>
            )}
            {error && (
              <span className="text-[10px] text-red-400">{error}</span>
            )}
            <div className="flex gap-2">
              <span className="text-[9px] text-gray-700 border border-gray-800 px-2 py-0.5 rounded">
                WebGL2
              </span>
              <span className="text-[9px] text-gray-700 border border-gray-800 px-2 py-0.5 rounded">
                4-PASS
              </span>
            </div>
          </div>
        </div>
        <DomainSelector />
      </div>

      {/* Main workspace */}
      <div className="flex-1 grid grid-cols-[260px_1fr_280px] lg:grid-cols-1 lg:grid-rows-[auto_1fr_auto]">
        {/* Left panel */}
        <div className="border-r border-gray-800/50 overflow-y-auto lg:border-r-0 lg:border-b">
          <div className="p-4 border-b border-gray-800/50">
            <div className="text-[9px] text-gray-600 uppercase tracking-widest mb-3">
              Input
            </div>
            <TextDropZone
              onData={processData}
              accept=".fasta,.fa,.fna,.txt"
              fileTypes="FASTA / TXT"
              placeholder="Drop sequence file"
            />
            <div className="mt-3">
              <textarea
                value={textInput}
                onChange={(e) => setTextInput(e.target.value)}
                onBlur={() => { if (textInput.trim()) processData(textInput); }}
                placeholder="Paste FASTA or raw ACGT sequence..."
                className="w-full h-24 bg-gray-900 border border-gray-800 rounded p-2 text-xs font-mono text-gray-400 resize-none focus:border-green-400/50 focus:outline-none"
              />
            </div>
            <button
              onClick={handleLoadExample}
              className="mt-2 w-full text-xs text-green-400 border border-green-400/30 rounded px-3 py-1.5 hover:bg-green-400/10 transition-colors"
            >
              Load Example (GFP Sequence)
            </button>
            {encodeError && (
              <div className="mt-2 text-xs text-red-400">{encodeError}</div>
            )}
            {encodingInfo && (
              <div className="mt-2 text-[10px] text-gray-600">{encodingInfo}</div>
            )}
          </div>

          <div className="p-4">
            <div className="text-[9px] text-gray-600 uppercase tracking-widest mb-4">
              Uniforms
            </div>
            <div className="space-y-4">
              <UniformSlider label="epsilon" id="epsilon" value={uniforms.epsilon} min={0.01} max={0.5} step={0.01} decimals={2} onChange={handleUniformChange} />
              <UniformSlider label="n_max" id="nmax" value={uniforms.nmax} min={2} max={16} step={1} decimals={0} onChange={handleUniformChange} />
              <UniformSlider label="beta" id="beta" value={uniforms.beta} min={0.5} max={8.0} step={0.1} decimals={2} onChange={handleUniformChange} />
              <UniformSlider label="J" id="J" value={uniforms.J} min={0.1} max={3.0} step={0.05} decimals={2} onChange={handleUniformChange} />
              <UniformSlider label="A_eg (x10^-4)" id="Aeg" value={uniforms.Aeg} min={0.1} max={10.0} step={0.1} decimals={2} onChange={handleUniformChange} />
              <UniformSlider label="alpha" id="alpha" value={uniforms.alpha} min={0.0} max={1.0} step={0.01} decimals={2} onChange={handleUniformChange} />
            </div>
          </div>
        </div>

        {/* Center panel */}
        <div className="flex items-center justify-center bg-[#050810] relative overflow-hidden">
          {loading && (
            <motion.div
              className="absolute inset-0 flex items-center justify-center bg-dark/80 z-10"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
            >
              <div className="text-green-400 text-sm animate-pulse tracking-wider">
                Processing on GPU...
              </div>
            </motion.div>
          )}

          {imagePreview ? (
            <img
              src={imagePreview}
              alt="Encoded k-mer frequency image"
              className="max-w-full max-h-full object-contain"
              style={{ imageRendering: 'pixelated' }}
            />
          ) : (
            <div className="text-center text-gray-700">
              <div className="text-lg font-semibold mb-2 text-gray-800">
                No Sequence Loaded
              </div>
              <div className="text-sm">
                Drop a FASTA file or paste a nucleotide sequence
              </div>
              <div className="text-sm">to begin the observation pipeline</div>
            </div>
          )}

          {result && (
            <div className="absolute bottom-3 left-1/2 -translate-x-1/2 flex gap-2">
              <span className="text-[9px] text-green-400 bg-dark/85 border border-gray-800 px-3 py-1 rounded-full backdrop-blur">
                Genomic Encoder
              </span>
              <span className="text-[9px] text-gray-500 bg-dark/85 border border-gray-800 px-3 py-1 rounded-full backdrop-blur">
                {result.elapsed_ms.toFixed(1)} ms
              </span>
            </div>
          )}
        </div>

        {/* Right panel */}
        <div className="border-l border-gray-800/50 overflow-y-auto lg:border-l-0 lg:border-t">
          <div className="p-4 space-y-6">
            <ObservationPanel result={result} />
            {result ? (
              <SimplexTriangle S_k={result.S_k} S_t={result.S_t} S_e={result.S_e} />
            ) : (
              <SimplexTriangle S_k={0.33} S_t={0.33} S_e={0.34} />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
