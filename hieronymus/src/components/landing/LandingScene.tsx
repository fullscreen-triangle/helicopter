'use client';

import React, { Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { ContactShadows, Environment } from '@react-three/drei';
import {
  Bloom,
  EffectComposer,
  Vignette,
} from '@react-three/postprocessing';
import PiCamera from './PiCamera';
import HolographicCell from './HolographicCell';

function SceneContent() {
  return (
    <>
      {/* Lighting */}
      <ambientLight intensity={0.4} />
      <spotLight
        position={[10, 10, 10]}
        angle={0.3}
        penumbra={1}
        intensity={1.5}
        castShadow
        shadow-mapSize={1024}
      />
      <spotLight
        position={[-10, 5, -10]}
        angle={0.4}
        penumbra={1}
        intensity={0.8}
        color="#00d4ff"
      />
      <spotLight
        position={[0, -5, 5]}
        angle={0.5}
        penumbra={1}
        intensity={0.5}
        color="#c084fc"
      />

      {/* Pi Camera - left side, slightly smaller */}
      <PiCamera position={[-2.8, -0.3, 0]} scale={0.2} />

      {/* Holographic Cell - right/center, the hero element */}
      <HolographicCell position={[1.5, 0.5, 0]} scale={1.8} />

      {/* Ground shadows */}
      <ContactShadows
        position={[0, -2.5, 0]}
        opacity={0.4}
        scale={12}
        blur={2.5}
        far={4}
        color="#00d4ff"
      />

      {/* Environment for reflections */}
      <Environment preset="city" />

      {/* Post-processing */}
      <EffectComposer multisampling={4} enableNormalPass={false}>
        <Bloom
          luminanceThreshold={0.35}
          mipmapBlur
          radius={0.4}
          intensity={1}
        />
        <Bloom
          luminanceThreshold={0.1}
          mipmapBlur
          radius={0.5}
          intensity={0.6}
        />
        <Vignette darkness={0.55} />
      </EffectComposer>
    </>
  );
}

export default function LandingScene() {
  return (
    <Canvas
      shadows
      camera={{ position: [0, 1, 8], fov: 45 }}
      gl={{ antialias: true, alpha: true }}
      style={{ width: '100%', height: '100%' }}
    >
      <Suspense fallback={null}>
        <SceneContent />
      </Suspense>
    </Canvas>
  );
}
