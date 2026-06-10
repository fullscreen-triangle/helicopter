'use client';

// PointCloud — microscopy pixels as a 3D point cloud.
//
// Each pixel (x,y) in the image becomes a 3D point:
//   x = column × pixelSizeµm
//   y = row    × pixelSizeµm
//   z = intensity × α(x,y) × zScale   (scale field lifts bright nucleus pixels)
//
// Colour follows viridis on intensity.  Nucleus centroids (from segmentation
// mask if provided) are shown as glowing spheres.
//
// This is the spatial embodiment of the `observe` operation: the image is
// rendered in the partition coordinate space where height encodes n-depth.

import React, { useMemo, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

// ── Viridis 11-stop colour map ─────────────────────────────────────────────────
const VIRIDIS: [number, number, number][] = [
  [0.267, 0.005, 0.329],
  [0.283, 0.141, 0.458],
  [0.253, 0.265, 0.530],
  [0.207, 0.372, 0.553],
  [0.164, 0.471, 0.558],
  [0.128, 0.567, 0.551],
  [0.135, 0.659, 0.518],
  [0.267, 0.749, 0.441],
  [0.477, 0.821, 0.318],
  [0.741, 0.873, 0.150],
  [0.993, 0.906, 0.144],
];
function viridis(t: number): [number, number, number] {
  const s = Math.max(0, Math.min(1, t)) * (VIRIDIS.length - 1);
  const lo = Math.floor(s), hi = Math.min(lo + 1, VIRIDIS.length - 1);
  const f = s - lo;
  return [
    VIRIDIS[lo][0] * (1 - f) + VIRIDIS[hi][0] * f,
    VIRIDIS[lo][1] * (1 - f) + VIRIDIS[hi][1] * f,
    VIRIDIS[lo][2] * (1 - f) + VIRIDIS[hi][2] * f,
  ];
}

// ── Nucleus glow sphere ────────────────────────────────────────────────────────
function NucleusSphere({ pos, colour }: { pos: [number, number, number]; colour: string }) {
  const meshRef = useRef<THREE.Mesh>(null);
  useFrame(({ clock }) => {
    if (meshRef.current) {
      const s = 1 + 0.12 * Math.sin(clock.elapsedTime * 2.0);
      meshRef.current.scale.setScalar(s);
    }
  });
  return (
    <mesh ref={meshRef} position={pos}>
      <sphereGeometry args={[0.6, 16, 16]} />
      <meshStandardMaterial color={colour} emissive={colour} emissiveIntensity={0.8} transparent opacity={0.85} />
    </mesh>
  );
}

// ── Inner point cloud — built from image data ──────────────────────────────────
interface CloudProps {
  imageData: Float32Array;
  width: number;
  height: number;
  scaleField?: Float32Array | null;
  segMask?: Uint8Array | null;
  pixelSizeµm?: number;
  zScale?: number;
  maxPoints?: number;
}

function Cloud({ imageData, width, height, scaleField, segMask, pixelSizeµm = 0.1, zScale = 4, maxPoints = 8192 }: CloudProps) {
  const geometry = useMemo(() => {
    // Subsample if image is large
    const total = width * height;
    const stride = Math.max(1, Math.ceil(Math.sqrt(total / maxPoints)));

    const pts: number[] = [];
    const cols: number[] = [];

    // Find centroid of nucleus_a and nucleus_b from segmentation mask
    let nAx = 0, nAy = 0, nAn = 0;
    let nBx = 0, nBy = 0, nBn = 0;
    let foundSplit = false;

    if (segMask) {
      for (let py = 0; py < height; py++) {
        for (let px = 0; px < width; px++) {
          const v = segMask[py * width + px];
          if (v === 1) { nAx += px; nAy += py; nAn++; foundSplit = true; }
          else if (v === 2) { nBx += px; nBy += py; nBn++; }
        }
      }
    }

    const centreX = (width  / 2) * pixelSizeµm;
    const centreY = (height / 2) * pixelSizeµm;

    for (let py = 0; py < height; py += stride) {
      for (let px = 0; px < width; px += stride) {
        const idx = py * width + px;
        const intensity = imageData[idx] ?? 0;
        if (intensity < 0.02) continue; // skip near-black background

        const alpha = scaleField ? (scaleField[idx] ?? 1) : 1;
        const x = px * pixelSizeµm - centreX;
        const y = py * pixelSizeµm - centreY;
        const z = intensity * alpha * zScale;

        pts.push(x, z, y); // three.js: y=up, so swap y/z so field grows upward

        const [r, g, b] = viridis(intensity);
        cols.push(r, g, b);
      }
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
    geo.setAttribute('color',    new THREE.Float32BufferAttribute(cols, 3));
    return { geo, foundSplit,
      cA: nAn ? [nAx / nAn * pixelSizeµm - centreX, 0, nAy / nAn * pixelSizeµm - centreY] as [number,number,number] : null,
      cB: nBn ? [nBx / nBn * pixelSizeµm - centreX, 0, nBy / nBn * pixelSizeµm - centreY] as [number,number,number] : null,
    };
  }, [imageData, width, height, scaleField, segMask, pixelSizeµm, zScale, maxPoints]);

  const { geo, cA, cB } = geometry;

  // Compute average z for nucleus sphere placement
  const avgZ = useMemo(() => {
    if (!imageData || !width || !height) return zScale * 0.6;
    let sum = 0, n = 0;
    for (let i = 0; i < imageData.length; i += 4) { sum += imageData[i]; n++; }
    return (sum / Math.max(n, 1)) * zScale * 1.5;
  }, [imageData, width, height, zScale]);

  const cAz: [number, number, number] | null = cA ? [cA[0], avgZ, cA[2]] : null;
  const cBz: [number, number, number] | null = cB ? [cB[0], avgZ, cB[2]] : null;

  return (
    <group>
      <points geometry={geo}>
        <pointsMaterial vertexColors size={0.15} sizeAttenuation transparent opacity={0.85} />
      </points>
      {cAz && <NucleusSphere pos={cAz} colour="#4ec9b0" />}
      {cBz && <NucleusSphere pos={cBz} colour="#c586c0" />}
      {/* Grid floor */}
      <gridHelper args={[20, 20, '#1a2a2a', '#1a2a2a']} position={[0, -0.1, 0]} />
    </group>
  );
}

// ── Scene with rotating camera default ────────────────────────────────────────
function Scene(props: CloudProps) {
  return (
    <>
      <ambientLight intensity={0.4} />
      <directionalLight position={[5, 8, 5]} intensity={0.8} />
      <pointLight position={[-5, 6, -5]} intensity={0.5} color="#4ec9b0" />
      <Cloud {...props} />
      <OrbitControls enableDamping dampingFactor={0.05} />
    </>
  );
}

// ── Exported component ─────────────────────────────────────────────────────────
export interface PointCloudProps {
  imageData: Float32Array | number[];
  width: number;
  height: number;
  scaleField?: Float32Array | null;
  segMask?: Uint8Array | null;
  pixelSizeµm?: number;
  zScale?: number;
}

export default function PointCloud({ imageData, width, height, scaleField, segMask, pixelSizeµm = 0.1, zScale = 4 }: PointCloudProps) {
  const data = useMemo(
    () => imageData instanceof Float32Array ? imageData : new Float32Array(imageData),
    [imageData]
  );

  return (
    <div className="w-full h-full min-h-[320px]" style={{ background: '#0d1117' }}>
      <Canvas
        camera={{ position: [8, 10, 8], fov: 50 }}
        gl={{ antialias: true }}
        style={{ width: '100%', height: '100%' }}
      >
        <Scene
          imageData={data}
          width={width}
          height={height}
          scaleField={scaleField}
          segMask={segMask}
          pixelSizeµm={pixelSizeµm}
          zScale={zScale}
        />
      </Canvas>
    </div>
  );
}
