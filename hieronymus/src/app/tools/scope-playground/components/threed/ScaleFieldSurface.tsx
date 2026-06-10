'use client';

import React, { useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import type { ScopeResult } from '@/lib/scope-runtime/runtime';

interface Props { result: ScopeResult; }

function Surface({ result }: Props) {
  const { visualData } = result;
  const sf = visualData.scaleField;
  const W = visualData.width, H = visualData.height;

  const geometry = useMemo(() => {
    if (!sf) return null;
    // Downsample to 64×64 for performance
    const step = Math.max(1, Math.floor(Math.min(W, H) / 64));
    const cols = Math.floor(W / step), rows = Math.floor(H / step);

    let min = Infinity, max = -Infinity;
    for (let i = 0; i < sf.length; i++) { if (sf[i] < min) min = sf[i]; if (sf[i] > max) max = sf[i]; }
    const range = max - min || 1;

    const geo = new THREE.PlaneGeometry(3, 3, cols - 1, rows - 1);
    const pos = geo.attributes.position;

    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        const idx = row * cols + col;
        const px = Math.min(col * step, W - 1);
        const py = Math.min(row * step, H - 1);
        const alpha = sf[py * W + px];
        pos.setZ(idx, ((alpha - min) / range) * 0.8);
      }
    }
    geo.computeVertexNormals();

    // Colour by z height (viridis-like)
    const colors = new Float32Array(pos.count * 3);
    for (let i = 0; i < pos.count; i++) {
      const t = (pos.getZ(i) / 0.8);
      colors[i * 3]     = 0.27 + t * 0.72;
      colors[i * 3 + 1] = 0.004 + t * 0.87;
      colors[i * 3 + 2] = 0.33 + t * (0.15 - 0.33);
    }
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    return geo;
  }, [sf, W, H]);

  if (!geometry) return null;

  return (
    <mesh geometry={geometry} rotation={[-Math.PI / 2.4, 0, 0]}>
      <meshPhongMaterial vertexColors side={THREE.DoubleSide} />
    </mesh>
  );
}

export default function ScaleFieldSurface({ result }: Props) {
  return (
    <div className="bg-[#1e1e1e] border border-[#3a3a3a] rounded overflow-hidden" style={{ height: 280 }}>
      <div className="text-[#858585] text-xs px-2 pt-1 pb-0">α(x,y) scale field surface</div>
      <Canvas camera={{ position: [0, 2.5, 3], fov: 50 }} gl={{ antialias: true }}>
        <ambientLight intensity={0.6} />
        <pointLight position={[3, 5, 3]} intensity={1.2} />
        <Surface result={result} />
        <OrbitControls />
      </Canvas>
    </div>
  );
}
