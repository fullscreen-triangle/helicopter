'use client';

import React, { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import Link from 'next/link';
import DropZone from '@/components/DropZone';
import MatchComparison from '@/components/MatchComparison';
import { useObservation } from '@/engine/useObservation';

export default function MatchPage() {
  const { ready, loading, matchResult, error, match } = useObservation();

  const [imageDataA, setImageDataA] = useState<ImageData | null>(null);
  const [imageDataB, setImageDataB] = useState<ImageData | null>(null);
  const [previewA, setPreviewA] = useState<string | null>(null);
  const [previewB, setPreviewB] = useState<string | null>(null);

  const makePreview = (imageData: ImageData): string => {
    const canvas = document.createElement('canvas');
    canvas.width = imageData.width;
    canvas.height = imageData.height;
    const ctx = canvas.getContext('2d');
    if (ctx) ctx.putImageData(imageData, 0, 0);
    return canvas.toDataURL();
  };

  const handleDropA = useCallback((imageData: ImageData) => {
    setImageDataA(imageData);
    setPreviewA(makePreview(imageData));
  }, []);

  const handleDropB = useCallback((imageData: ImageData) => {
    setImageDataB(imageData);
    setPreviewB(makePreview(imageData));
  }, []);

  const handleCompare = useCallback(() => {
    if (!imageDataA || !imageDataB) return;
    match(imageDataA, imageDataB);
  }, [imageDataA, imageDataB, match]);

  const canCompare = ready && imageDataA && imageDataB && !loading;

  return (
    <div className="min-h-[calc(100vh-140px)] flex flex-col">
      {/* Header */}
      <div className="px-6 py-3 border-b border-gray-800/50 flex items-center gap-4">
        <h1 className="text-sm font-semibold tracking-wider text-amber-400">
          MATCH
        </h1>
        <span className="text-[10px] text-gray-600 tracking-widest">
          INTERFERENCE COMPARISON
        </span>
        <div className="ml-auto flex items-center gap-3">
          <Link
            href="/match/batch"
            className="text-[10px] text-gray-400 hover:text-amber-400 transition-colors tracking-wider border border-gray-800 px-3 py-1 rounded"
          >
            BATCH MATCH
          </Link>
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

      {/* Main content */}
      <div className="flex-1 grid grid-cols-[1fr_360px] lg:grid-cols-1 lg:grid-rows-[auto_1fr]">
        {/* Left: Drop zones and compare */}
        <div className="p-6 space-y-6">
          <div className="grid grid-cols-2 gap-6 sm:grid-cols-1">
            <div className="space-y-2">
              <div className="text-[10px] text-cyan-400 uppercase tracking-widest">
                Image A
              </div>
              <DropZone onDrop={handleDropA} />
            </div>
            <div className="space-y-2">
              <div className="text-[10px] text-amber-400 uppercase tracking-widest">
                Image B
              </div>
              <DropZone onDrop={handleDropB} />
            </div>
          </div>

          <button
            onClick={handleCompare}
            disabled={!canCompare}
            className={`w-full py-3 rounded-lg text-sm font-semibold tracking-widest uppercase transition-colors duration-200
              ${
                canCompare
                  ? 'border border-amber-400 text-amber-400 hover:bg-amber-400 hover:text-dark cursor-pointer'
                  : 'border border-gray-800 text-gray-700 cursor-not-allowed'
              }`}
          >
            {loading ? 'Comparing...' : 'Compare'}
          </button>

          {/* Loading overlay */}
          {loading && (
            <motion.div
              className="flex items-center justify-center py-8"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
            >
              <div className="text-amber-400 text-sm animate-pulse tracking-wider">
                Running GPU interference pipeline...
              </div>
            </motion.div>
          )}

          {/* Preview area for large screens */}
          {matchResult && !loading && (
            <div className="hidden lg:block">
              <MatchComparison
                result={matchResult}
                previewA={previewA}
                previewB={previewB}
              />
            </div>
          )}

          {/* Empty state */}
          {!matchResult && !loading && (
            <div className="text-center py-16">
              <div className="text-gray-700 text-lg font-semibold mb-2">
                Drop Two Images to Compare
              </div>
              <p className="text-gray-600 text-sm max-w-md mx-auto">
                The GPU engine will encode each image into its partition
                coordinates, then compute the interference between them to
                produce a match score and S-distance on the entropy simplex.
              </p>
            </div>
          )}
        </div>

        {/* Right: Results panel */}
        <div className="border-l border-gray-800/50 overflow-y-auto lg:border-l-0 lg:border-t lg:hidden">
          <div className="p-4">
            {matchResult ? (
              <MatchComparison
                result={matchResult}
                previewA={previewA}
                previewB={previewB}
              />
            ) : (
              <div className="space-y-4">
                <h3 className="text-xs text-gray-600 uppercase tracking-widest">
                  Match Results
                </h3>
                <div className="text-gray-700 text-sm text-center py-8">
                  Select two images and click Compare
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
