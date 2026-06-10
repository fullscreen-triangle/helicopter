'use client';

import React, { useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import type { ScopeResult } from '@/lib/scope-runtime/runtime';

interface Props { result: ScopeResult; }

function Tube({ result }: Props) {
  const { visualData } = result;
  const path = visualData.geodesicPath;
  const W = visualData.width, H = visualData.height;

  const tubeGeometry = useMemo(() => {
    if (!path || path.length < 2) return null;
    const pts = path.map(([px, py]) =>
      new THREE.Vector3((px / W - 0.5) * 3, 0, (py / H - 0.5) * 3)
    );
    const curve = new THREE.CatmullRomCurve3(pts);
    return new THREE.TubeGeometry(curve, Math.min(path.length * 2, 200), 0.025, 8, false);
  }, [path, W, H]);

  if (!tubeGeometry) {
    return (
      <mesh>
        <sphereGeometry args={[0.1, 8, 8]} />
        <meshBasicMaterial color={0x555555} />
      </mesh>
    );
  }

  return (
    <group>
      <mesh geometry={tubeGeometry}>
        <meshPhongMaterial color={0xffd700} emissive={0x886600} />
      </mesh>
      {/* endpoint spheres */}
      {[path[0], path[path.length - 1]].map(([px, py], i) => (
        <mesh key={i} position={[(px / W - 0.5) * 3, 0, (py / H - 0.5) * 3]}>
          <sphereGeometry args={[0.07, 12, 12]} />
          <meshPhongMaterial color={i === 0 ? 0x4ec9b0 : 0xc586c0} />
        </mesh>
      ))}
    </group>
  );
}

export default function DistanceTube({ result }: Props) {
  return (
    <div className="bg-[#1e1e1e] border border-[#3a3a3a] rounded overflow-hidden" style={{ height: 280 }}>
      <div className="text-[#858585] text-xs px-2 pt-1 pb-0">Geodesic path tube</div>
      {result.distance !== null && (
        <div className="text-[#ffd700] text-xs px-2">
          d = {result.distance.toFixed(3)} ± {result.uncertainty?.toFixed(3) ?? '?'} µm
        </div>
      )}
      <Canvas camera={{ position: [0, 2.5, 2.5], fov: 50 }} gl={{ antialias: true }}>
        <ambientLight intensity={0.5} />
        <pointLight position={[3, 4, 3]} intensity={1.2} />
        <Tube result={result} />
        <OrbitControls />
      </Canvas>
    </div>
  );
}
