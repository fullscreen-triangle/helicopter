'use client';

import React, { useRef, useState, useEffect, Suspense } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Html, Grid } from '@react-three/drei';
import { MeshProcessingService } from '@/lib/cell-data/cellDataService';
import { Download, Upload } from 'lucide-react';
import * as THREE from 'three';

// ── Geometry helpers ──────────────────────────────────────────────────────────

function generateOrganicMesh(): { positions: Float32Array; normals: Float32Array } {
  const segments = 32, rings = 16;
  const positions: number[] = [], normals: number[] = [];
  for (let lat = 0; lat <= rings; lat++) {
    const theta = (lat * Math.PI) / rings;
    const sinTheta = Math.sin(theta), cosTheta = Math.cos(theta);
    for (let lon = 0; lon <= segments; lon++) {
      const phi = (lon * 2 * Math.PI) / segments;
      const sinPhi = Math.sin(phi), cosPhi = Math.cos(phi);
      const x = cosPhi * sinTheta, y = cosTheta, z = sinPhi * sinTheta;
      const distortion = 1 + 0.2 * Math.sin(4 * phi) * Math.sin(4 * theta);
      positions.push(x * distortion, y * distortion, z * distortion);
      normals.push(x, y, z);
    }
  }
  return { positions: new Float32Array(positions), normals: new Float32Array(normals) };
}

// ── Three.js scene components ─────────────────────────────────────────────────

function CellMesh({ meshData, color, opacity }: {
  meshData: { positions: Float32Array; normals?: Float32Array; indices?: Uint32Array } | null;
  color: string; opacity: number;
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);

  useFrame((state) => {
    if (meshRef.current) meshRef.current.rotation.y = state.clock.getElapsedTime() * 0.1;
  });

  if (!meshData) return null;
  return (
    <mesh ref={meshRef} scale={hovered ? 1.05 : 1}
      onPointerOver={() => setHovered(true)} onPointerOut={() => setHovered(false)}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position"
          count={meshData.positions.length / 3} array={meshData.positions} itemSize={3} args={[meshData.positions, 3]} />
        {meshData.normals && (
          <bufferAttribute attach="attributes-normal"
            count={meshData.normals.length / 3} array={meshData.normals} itemSize={3} args={[meshData.normals, 3]} />
        )}
      </bufferGeometry>
      <meshStandardMaterial color={color} opacity={opacity} transparent roughness={0.3} metalness={0.2} />
    </mesh>
  );
}

function Nucleus({ visible, color = '#ff6b6b', size = 0.4 }: { visible: boolean; color?: string; size?: number }) {
  if (!visible) return null;
  return (
    <mesh position={[0, 0, 0]}>
      <sphereGeometry args={[size, 32, 32]} />
      <meshStandardMaterial color={color} opacity={0.7} transparent roughness={0.4} />
    </mesh>
  );
}

const ORGANELLES = [
  { position: [0.3, 0.2, 0.1] as [number,number,number],  size: 0.12, color: '#ffd700', label: 'Mitochondria' },
  { position: [-0.25, 0.3, 0.2] as [number,number,number], size: 0.10, color: '#9370db', label: 'Golgi' },
  { position: [0.1, -0.28, 0.15] as [number,number,number], size: 0.08, color: '#20b2aa', label: 'ER' },
  { position: [-0.3, -0.2, -0.1] as [number,number,number], size: 0.09, color: '#ff69b4', label: 'Lysosome' },
];

function Organelle({ position, size, color, label }: {
  position: [number,number,number]; size: number; color: string; label: string;
}) {
  const [show, setShow] = useState(false);
  return (
    <mesh position={position} onPointerOver={() => setShow(true)} onPointerOut={() => setShow(false)}>
      <sphereGeometry args={[size, 16, 16]} />
      <meshStandardMaterial color={color} opacity={0.85} transparent />
      {show && (
        <Html distanceFactor={8}>
          <div style={{ background: '#000c', color: '#fff', padding: '2px 6px', borderRadius: 4, fontSize: 11, whiteSpace: 'nowrap' }}>
            {label}
          </div>
        </Html>
      )}
    </mesh>
  );
}

function SlicingPlane({ y, visible }: { y: number; visible: boolean }) {
  if (!visible) return null;
  return (
    <mesh position={[0, y, 0]} rotation={[Math.PI / 2, 0, 0]}>
      <planeGeometry args={[3, 3]} />
      <meshBasicMaterial color="#00ff88" opacity={0.18} transparent side={THREE.DoubleSide} />
    </mesh>
  );
}

// ── Main component ─────────────────────────────────────────────────────────────

export default function CellViewer3D() {
  const [meshData, setMeshData]         = useState<{ positions: Float32Array; normals?: Float32Array } | null>(null);
  const [loading, setLoading]           = useState(false);
  const [showNucleus, setShowNucleus]   = useState(true);
  const [showOrganelles, setShowOrganelles] = useState(true);
  const [opacity, setOpacity]           = useState(0.75);
  const [sliceY, setSliceY]             = useState(0);
  const [showSlice, setShowSlice]       = useState(false);
  const [color, setColor]               = useState('#4a90e2');
  const [autoRotate, setAutoRotate]     = useState(true);
  const [info, setInfo]                 = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => { setMeshData(generateOrganicMesh()); }, []);

  const handleFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setLoading(true);
    setInfo(null);
    try {
      const svc = new MeshProcessingService();
      if (file.name.endsWith('.stl')) {
        const buf = await file.arrayBuffer();
        setMeshData(svc.parseSTL(buf));
        setInfo(`STL: ${Math.round((await file.arrayBuffer()).byteLength / 1024)} KB`);
      } else if (file.name.endsWith('.obj')) {
        const text = await file.text();
        setMeshData(svc.parseOBJ(text));
        setInfo(`OBJ: ${file.name}`);
      } else {
        setInfo('Unsupported format — use .stl or .obj');
      }
    } finally { setLoading(false); }
  };

  const handleExport = () => {
    const a = document.createElement('a');
    a.href = document.querySelector('canvas')?.toDataURL('image/png') ?? '#';
    a.download = 'cell-3d.png';
    a.click();
  };

  return (
    <div className="w-full h-screen bg-[#111] relative flex flex-col">
      {/* Header */}
      <div className="flex items-center gap-4 px-4 py-2 bg-[#1e1e1e] border-b border-[#3a3a3a] z-10 shrink-0">
        <span className="text-white font-semibold text-sm tracking-wide">ADVANCED 3D CELL VIEWER</span>
        <div className="flex-1" />
        <span className="text-[#858585] text-xs">React Three Fiber · OrbitControls · STL/OBJ upload</span>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Controls panel */}
        <div className="w-56 shrink-0 bg-[#1e1e1e] border-r border-[#3a3a3a] p-3 space-y-4 overflow-y-auto z-10">

          {/* Upload */}
          <div>
            <div className="text-[#858585] text-[0.7rem] uppercase mb-1">Mesh File</div>
            <button onClick={() => fileRef.current?.click()}
              className="flex items-center gap-2 w-full px-3 py-1.5 rounded border border-[#3a3a3a] hover:bg-[#2a2a2a] text-xs text-[#d4d4d4]">
              <Upload size={12} /> Upload STL / OBJ
            </button>
            <input ref={fileRef} type="file" accept=".stl,.obj" className="hidden" onChange={handleFile} />
            {info && <div className="text-[#858585] text-[0.65rem] mt-1">{info}</div>}
            {loading && <div className="text-[#ffa500] text-xs mt-1">Loading…</div>}
          </div>

          {/* Visibility */}
          <div>
            <div className="text-[#858585] text-[0.7rem] uppercase mb-1">Visibility</div>
            {[
              ['Nucleus', showNucleus, setShowNucleus],
              ['Organelles', showOrganelles, setShowOrganelles],
              ['Auto Rotate', autoRotate, setAutoRotate],
              ['Slicing Plane', showSlice, setShowSlice],
            ].map(([label, val, set]: any) => (
              <label key={label} className="flex items-center gap-2 py-1 text-xs cursor-pointer">
                <input type="checkbox" checked={val} onChange={e => set(e.target.checked)} className="w-3 h-3" />
                <span className="text-[#d4d4d4]">{label}</span>
              </label>
            ))}
          </div>

          {/* Opacity */}
          <div>
            <div className="text-[#858585] text-[0.7rem] uppercase mb-1">Cell Opacity: {opacity.toFixed(2)}</div>
            <input type="range" min={0} max={1} step={0.01} value={opacity}
              onChange={e => setOpacity(parseFloat(e.target.value))} className="w-full accent-[#007acc]" />
          </div>

          {/* Slice position */}
          {showSlice && (
            <div>
              <div className="text-[#858585] text-[0.7rem] uppercase mb-1">Slice Y: {sliceY.toFixed(2)}</div>
              <input type="range" min={-1.5} max={1.5} step={0.05} value={sliceY}
                onChange={e => setSliceY(parseFloat(e.target.value))} className="w-full accent-[#4ec9b0]" />
            </div>
          )}

          {/* Cell color */}
          <div>
            <div className="text-[#858585] text-[0.7rem] uppercase mb-1">Cell Color</div>
            <input type="color" value={color} onChange={e => setColor(e.target.value)}
              className="w-full h-8 rounded cursor-pointer border border-[#3a3a3a]" />
          </div>

          {/* Export */}
          <button onClick={handleExport}
            className="flex items-center gap-2 w-full px-3 py-1.5 rounded bg-[#007acc] hover:bg-[#0098ff] text-white text-xs">
            <Download size={12} /> Export PNG
          </button>

          {/* Info */}
          <div className="border border-[#3a3a3a] rounded p-2 text-[0.65rem] space-y-1">
            <div className="text-[#858585]">Cell Info</div>
            <div className="text-[#d4d4d4]">Type: HeLa A9</div>
            <div className="text-[#858585]">Vertices: {meshData ? (meshData.positions.length / 3).toFixed(0) : 0}</div>
            <div className="text-[#858585]">Organelles: {ORGANELLES.length}</div>
          </div>
        </div>

        {/* 3D Canvas */}
        <div className="flex-1 relative">
          <Canvas>
            <PerspectiveCamera makeDefault position={[0, 0, 4]} fov={50} />
            <ambientLight intensity={0.5} />
            <pointLight position={[8, 8, 8]} intensity={1.2} />
            <pointLight position={[-8, -8, -8]} intensity={0.4} />
            <spotLight position={[0, 8, 0]} angle={0.35} intensity={0.6} />

            <Suspense fallback={null}>
              <CellMesh meshData={meshData} color={color} opacity={opacity} />
              <Nucleus visible={showNucleus} />
              {showOrganelles && ORGANELLES.map((org, i) => <Organelle key={i} {...org} />)}
              <SlicingPlane y={sliceY} visible={showSlice} />
            </Suspense>

            <OrbitControls autoRotate={autoRotate} autoRotateSpeed={0.6}
              enableZoom enablePan minDistance={1.5} maxDistance={12} />
            <Grid args={[10, 10]} cellColor="#222" sectionColor="#333" position={[0, -1.6, 0]} />
          </Canvas>
        </div>
      </div>
    </div>
  );
}
