'use client';

import React from 'react';

export default function FooterApp() {
  return (
    <footer className="w-full border-t border-gray-800/50 text-gray-600 text-sm">
      <div className="px-12 py-6 flex items-center justify-between sm:flex-col sm:gap-2 sm:px-6">
        <span>{new Date().getFullYear()} Hieronymus</span>
        <span className="text-xs tracking-wider">
          Client-side GPU Observation Engine
        </span>
        <span className="text-xs">
          Sk + St + Se = 1
        </span>
      </div>
    </footer>
  );
}
