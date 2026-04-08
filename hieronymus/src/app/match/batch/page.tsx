'use client';

import React, { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import Link from 'next/link';
import DropZone from '@/components/DropZone';
import ObservationPanel from '@/components/ObservationPanel';
import SimplexTriangle from '@/components/SimplexTriangle';
import BatchResults, { type BatchResultItem } from '@/components/BatchResults';
import { useObservation } from '@/engine/useObservation';

export default function BatchMatchPage() {
  const { ready, loading, result, error, observe } = useObservation();

  const [storeName, setStoreName] = useState('');
  const [storeDomain, setStoreDomain] = useState('microscopy');
  const [storeStatus, setStoreStatus] = useState<string | null>(null);
  const [batchResults, setBatchResults] = useState<BatchResultItem[]>([]);
  const [batchLoading, setBatchLoading] = useState(false);
  const [selectedResult, setSelectedResult] = useState<BatchResultItem | null>(null);
  const [storeCount, setStoreCount] = useState(0);

  const handleDrop = useCallback(
    (imageData: ImageData) => {
      observe(imageData, 'microscopy');
    },
    [observe]
  );

  const handleStore = useCallback(async () => {
    if (!result) return;
    setStoreStatus('Storing...');
    try {
      const res = await fetch('/api/store', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          S_k: result.S_k,
          S_t: result.S_t,
          S_e: result.S_e,
          name: storeName || `Observation ${Date.now()}`,
          domain: storeDomain,
        }),
      });
      const data = await res.json();
      if (data.stored) {
        setStoreStatus(`Stored as ${data.id.slice(0, 8)}...`);
        setStoreCount((c) => c + 1);
        setTimeout(() => setStoreStatus(null), 3000);
      } else {
        setStoreStatus('Failed to store');
      }
    } catch (err) {
      setStoreStatus('Error: ' + (err instanceof Error ? err.message : 'unknown'));
    }
  }, [result, storeName, storeDomain]);

  const handleBatchMatch = useCallback(async () => {
    if (!result) return;
    setBatchLoading(true);
    setBatchResults([]);
    setSelectedResult(null);
    try {
      const res = await fetch('/api/match', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          S_k: result.S_k,
          S_t: result.S_t,
          S_e: result.S_e,
          topK: 20,
        }),
      });
      const data = await res.json();
      if (Array.isArray(data)) {
        setBatchResults(data);
      }
    } catch (err) {
      console.error('Batch match failed:', err);
    } finally {
      setBatchLoading(false);
    }
  }, [result]);

  const handleSelectResult = useCallback((item: BatchResultItem) => {
    setSelectedResult(item);
  }, []);

  return (
    <div className="min-h-[calc(100vh-140px)] flex flex-col">
      {/* Header */}
      <div className="px-6 py-3 border-b border-gray-800/50 flex items-center gap-4">
        <h1 className="text-sm font-semibold tracking-wider text-amber-400">
          BATCH MATCH
        </h1>
        <span className="text-[10px] text-gray-600 tracking-widest">
          DATABASE COMPARISON
        </span>
        <div className="ml-auto flex items-center gap-3">
          <Link
            href="/match"
            className="text-[10px] text-gray-400 hover:text-amber-400 transition-colors tracking-wider border border-gray-800 px-3 py-1 rounded"
          >
            PAIR MATCH
          </Link>
          {storeCount > 0 && (
            <span className="text-[10px] text-gray-500 border border-gray-800 px-2 py-0.5 rounded">
              {storeCount} stored
            </span>
          )}
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
        </div>
      </div>

      {/* Main workspace */}
      <div className="flex-1 grid grid-cols-[300px_1fr] lg:grid-cols-1">
        {/* Left panel: Input + Store */}
        <div className="border-r border-gray-800/50 overflow-y-auto lg:border-r-0 lg:border-b">
          <div className="p-4 border-b border-gray-800/50">
            <div className="text-[9px] text-gray-600 uppercase tracking-widest mb-3">
              Query Image
            </div>
            <DropZone onDrop={handleDrop} />
          </div>

          {/* Store controls */}
          <div className="p-4 border-b border-gray-800/50 space-y-3">
            <div className="text-[9px] text-gray-600 uppercase tracking-widest">
              Store Observation
            </div>
            <input
              type="text"
              placeholder="Name"
              value={storeName}
              onChange={(e) => setStoreName(e.target.value)}
              className="w-full px-3 py-2 text-sm bg-gray-900 border border-gray-800 rounded text-gray-300 placeholder-gray-700 focus:border-amber-400/50 focus:outline-none"
            />
            <select
              value={storeDomain}
              onChange={(e) => setStoreDomain(e.target.value)}
              className="w-full px-3 py-2 text-sm bg-gray-900 border border-gray-800 rounded text-gray-300 focus:border-amber-400/50 focus:outline-none"
            >
              <option value="microscopy">Microscopy</option>
              <option value="spectroscopy">Spectroscopy</option>
              <option value="medical">Medical Imaging</option>
              <option value="satellite">Satellite</option>
              <option value="other">Other</option>
            </select>
            <button
              onClick={handleStore}
              disabled={!result}
              className={`w-full py-2 rounded text-xs font-semibold tracking-widest uppercase transition-colors
                ${
                  result
                    ? 'border border-green-400/50 text-green-400 hover:bg-green-400/10 cursor-pointer'
                    : 'border border-gray-800 text-gray-700 cursor-not-allowed'
                }`}
            >
              Store
            </button>
            {storeStatus && (
              <div className="text-[10px] text-gray-400 text-center">
                {storeStatus}
              </div>
            )}
          </div>

          {/* Match against DB */}
          <div className="p-4 border-b border-gray-800/50 space-y-3">
            <div className="text-[9px] text-gray-600 uppercase tracking-widest">
              Match Against Database
            </div>
            <button
              onClick={handleBatchMatch}
              disabled={!result || batchLoading}
              className={`w-full py-2 rounded text-xs font-semibold tracking-widest uppercase transition-colors
                ${
                  result && !batchLoading
                    ? 'border border-amber-400 text-amber-400 hover:bg-amber-400/10 cursor-pointer'
                    : 'border border-gray-800 text-gray-700 cursor-not-allowed'
                }`}
            >
              {batchLoading ? 'Matching...' : 'Match'}
            </button>
          </div>

          {/* Observation metrics */}
          <div className="p-4 space-y-4">
            <ObservationPanel result={result} />
            {result ? (
              <SimplexTriangle
                S_k={result.S_k}
                S_t={result.S_t}
                S_e={result.S_e}
              />
            ) : (
              <SimplexTriangle S_k={0.33} S_t={0.33} S_e={0.34} />
            )}
          </div>
        </div>

        {/* Right: Results */}
        <div className="overflow-y-auto">
          {loading && (
            <motion.div
              className="flex items-center justify-center py-12"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
            >
              <div className="text-cyan-400 text-sm animate-pulse tracking-wider">
                Processing on GPU...
              </div>
            </motion.div>
          )}

          {batchLoading && (
            <motion.div
              className="flex items-center justify-center py-12"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
            >
              <div className="text-amber-400 text-sm animate-pulse tracking-wider">
                Querying database...
              </div>
            </motion.div>
          )}

          {!loading && !batchLoading && batchResults.length > 0 && (
            <div className="p-4 space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-xs text-gray-600 uppercase tracking-widest">
                  Results ({batchResults.length})
                </h3>
                {selectedResult && (
                  <button
                    onClick={() => setSelectedResult(null)}
                    className="text-[10px] text-gray-500 hover:text-gray-300"
                  >
                    Clear selection
                  </button>
                )}
              </div>

              {/* Selected result detail */}
              {selectedResult && result && (
                <motion.div
                  className="p-4 bg-gray-900/50 rounded-lg border border-amber-400/20 space-y-3"
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-semibold text-amber-400">
                      {selectedResult.name}
                    </span>
                    <span className="text-xs text-gray-500">
                      {selectedResult.domain}
                    </span>
                  </div>
                  <div className="grid grid-cols-4 gap-2 text-center">
                    <div className="p-2 bg-gray-950 rounded">
                      <div className="text-[9px] text-gray-600 uppercase">
                        S-Dist
                      </div>
                      <div className="text-sm font-mono text-amber-400">
                        {selectedResult.distance.toFixed(6)}
                      </div>
                    </div>
                    <div className="p-2 bg-gray-950 rounded">
                      <div className="text-[9px] text-gray-600 uppercase">
                        Sk
                      </div>
                      <div className="text-sm font-mono text-cyan-400">
                        {selectedResult.S_k.toFixed(4)}
                      </div>
                    </div>
                    <div className="p-2 bg-gray-950 rounded">
                      <div className="text-[9px] text-gray-600 uppercase">
                        St
                      </div>
                      <div className="text-sm font-mono text-amber-400">
                        {selectedResult.S_t.toFixed(4)}
                      </div>
                    </div>
                    <div className="p-2 bg-gray-950 rounded">
                      <div className="text-[9px] text-gray-600 uppercase">
                        Se
                      </div>
                      <div className="text-sm font-mono text-violet-400">
                        {selectedResult.S_e.toFixed(4)}
                      </div>
                    </div>
                  </div>
                  {/* Comparison: query vs selected */}
                  <div className="text-[10px] text-gray-500">
                    Query: Sk={result.S_k.toFixed(4)} St={result.S_t.toFixed(4)}{' '}
                    Se={result.S_e.toFixed(4)}
                  </div>
                </motion.div>
              )}

              <BatchResults
                results={batchResults}
                onSelect={handleSelectResult}
              />
            </div>
          )}

          {!loading && !batchLoading && batchResults.length === 0 && (
            <div className="flex items-center justify-center py-20">
              <div className="text-center max-w-md">
                <div className="text-gray-700 text-lg font-semibold mb-2">
                  Batch Matching
                </div>
                <p className="text-gray-600 text-sm leading-relaxed">
                  1. Drop an image to observe it.
                  <br />
                  2. Store observations to build a database.
                  <br />
                  3. Match a query image against all stored observations by
                  S-entropy distance.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
