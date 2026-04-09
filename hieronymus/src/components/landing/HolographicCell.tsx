'use client';

import React, { useRef, useEffect } from 'react';
import { useGLTF, Float } from '@react-three/drei';
import { useFrame } from '@react-three/fiber';
import type { Group, Mesh } from 'three';
import type { ThreeElements } from '@react-three/fiber';
import HolographicMaterial from './HolographicMaterial';

export default function HolographicCell(props: ThreeElements['group']) {
  const groupRef = useRef<Group>(null!);
  const { scene } = useGLTF('/human_epidermal_cell.glb') as any;

  // Replace all mesh materials with a placeholder; we render holographic material via children
  // Instead, we clone the scene and strip materials so we can attach HolographicMaterial
  const meshRefs = useRef<Mesh[]>([]);

  useEffect(() => {
    const meshes: Mesh[] = [];
    scene.traverse((child: any) => {
      if (child.isMesh) {
        meshes.push(child);
        // Make original material transparent so holographic material shows
        child.material.transparent = true;
        child.material.opacity = 0;
      }
    });
    meshRefs.current = meshes;
  }, [scene]);

  useFrame((_state, delta) => {
    if (groupRef.current) {
      groupRef.current.rotation.y += delta * 0.1;
    }
  });

  return (
    <Float speed={1.5} rotationIntensity={0.3} floatIntensity={0.8}>
      <group ref={groupRef} {...props} dispose={null}>
        <primitive object={scene} />
        {/* Overlay holographic meshes on top of the model geometry */}
        <HolographicOverlay scene={scene} />
      </group>
    </Float>
  );
}

/**
 * Traverses the loaded scene and renders each mesh geometry
 * with the HolographicMaterial applied.
 */
function HolographicOverlay({ scene }: { scene: any }) {
  const meshes: { geometry: any; matrix: any; key: string }[] = [];

  scene.traverse((child: any) => {
    if (child.isMesh && child.geometry) {
      meshes.push({
        geometry: child.geometry,
        matrix: child.matrixWorld,
        key: child.uuid,
      });
    }
  });

  return (
    <>
      {meshes.map((m) => (
        <mesh key={m.key} geometry={m.geometry} matrixAutoUpdate={false} matrix={m.matrix}>
          <HolographicMaterial
            hologramColor="#00d4ff"
            hologramBrightness={1.5}
            signalSpeed={0.3}
            scanlineSize={6}
            fresnelAmount={0.45}
            fresnelOpacity={1.0}
            enableBlinking={true}
            blinkFresnelOnly={true}
            enableAdditive={true}
            hologramOpacity={1.0}
            side="DoubleSide"
          />
        </mesh>
      ))}
    </>
  );
}

useGLTF.preload('/human_epidermal_cell.glb');
