'use client';

import React, { useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

interface Props { sk: number; st: number; se: number; }

function Sectors({ sk, st, se }: Props) {
  const groupRef = useRef<THREE.Group>(null);
  useFrame(() => { if (groupRef.current) groupRef.current.rotation.y += 0.005; });

  const R = 1.4;
  const total = sk + st + se;
  const angles = [sk / total * Math.PI * 2, st / total * Math.PI * 2, se / total * Math.PI * 2];
  const colors = [0x4ec9b0, 0x569cd6, 0xc586c0];
  const labels = [`S_k=${sk.toFixed(3)}`, `S_t=${st.toFixed(3)}`, `S_e=${se.toFixed(3)}`];

  let theta = 0;
  const meshes = angles.map((arc, i) => {
    const start = theta;
    theta += arc;
    const geo = new THREE.SphereGeometry(R, 24, 24, start, arc, 0, Math.PI);
    const mat = new THREE.MeshPhongMaterial({ color: colors[i], transparent: true, opacity: 0.72, side: THREE.DoubleSide });
    return <mesh key={i} geometry={geo} material={mat} />;
  });

  return <group ref={groupRef}>{meshes}</group>;
}

export default function EntropySphere({ sk, st, se }: Props) {
  return (
    <div className="bg-[#1e1e1e] border border-[#3a3a3a] rounded overflow-hidden" style={{ height: 280 }}>
      <div className="text-[#858585] text-xs px-2 pt-1 pb-0">S-entropy sphere (S_k / S_t / S_e sectors)</div>
      <Canvas camera={{ position: [0, 0, 4], fov: 45 }} gl={{ antialias: true }}>
        <ambientLight intensity={0.5} />
        <pointLight position={[4, 4, 4]} intensity={1.2} />
        <Sectors sk={sk} st={st} se={se} />
        <OrbitControls enablePan={false} />
      </Canvas>
    </div>
  );
}
