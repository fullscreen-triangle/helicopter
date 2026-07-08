'use client';

import React, { useRef, useEffect, forwardRef, useImperativeHandle } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Grid } from '@react-three/drei';
import { Points, BufferGeometry, BufferAttribute } from 'three';

type VisualizationMode = 'scale-field' | 'segmentation' | 'distance';

interface SceneViewerProps {
  mode: VisualizationMode;
  scaleField?: Float32Array;
}

function SceneContent({
  mode,
  scaleField,
  onUpdateRef,
}: {
  mode: VisualizationMode;
  scaleField?: Float32Array;
  onUpdateRef: (ref: any) => void;
}) {
  const meshRef = useRef<Points>(null);

  useEffect(() => {
    if (meshRef.current && scaleField) {
      updateVisualization(scaleField, mode);
    }
  }, [scaleField, mode]);

  const updateVisualization = (scalefield: Float32Array, visualMode: VisualizationMode) => {
    if (!meshRef.current) return;

    const geometry = meshRef.current.geometry as BufferGeometry;
    const positions = geometry.attributes.position?.array as Float32Array;

    if (!positions) return;

    // Create colored voxel visualization
    const colors = new Float32Array((scalefield.length / (256 * 256)) * 256 * 256 * 3);

    for (let i = 0; i < scalefield.length; i++) {
      const value = scalefield[i];

      // Map value to color (blue -> cyan -> green -> yellow -> red)
      let r = 0, g = 0, b = 0;

      if (visualMode === 'scale-field') {
        // Heat map: blue (0) -> red (1)
        if (value < 0.5) {
          b = 1 - value * 2;
          r = value * 2;
        } else {
          r = 1;
          g = (value - 0.5) * 2;
        }
      } else if (visualMode === 'segmentation') {
        // Threshold-based
        if (value > 0.6) {
          r = 0.2;
          g = 0.8;
          b = 0.2;
        }
      } else if (visualMode === 'distance') {
        // Gradient visualization
        r = value;
        g = 1 - Math.abs(value - 0.5) * 2;
        b = 1 - value;
      }

      colors[i * 3] = r;
      colors[i * 3 + 1] = g;
      colors[i * 3 + 2] = b;
    }

    geometry.setAttribute('color', new BufferAttribute(colors, 3));
  };

  // Create point cloud visualization
  useEffect(() => {
    if (!meshRef.current || !scaleField) return;

    const geometry = meshRef.current.geometry as BufferGeometry;
    const positions = new Float32Array(scaleField.length * 3);

    let idx = 0;
    for (let y = 0; y < 256; y++) {
      for (let x = 0; x < 256; x++) {
        const i = y * 256 + x;
        positions[idx++] = (x / 256) * 10 - 5;
        positions[idx++] = (y / 256) * 10 - 5;
        positions[idx++] = scaleField[i] * 2;
      }
    }

    geometry.setAttribute('position', new BufferAttribute(positions, 3));
    geometry.computeBoundingSphere();
  }, [scaleField]);

  useImperativeHandle(onUpdateRef, () => ({
    updateVisualization,
  }));

  return (
    <>
      <points ref={meshRef}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={0}
            array={new Float32Array(0)}
            itemSize={3}
          />
        </bufferGeometry>
        <pointsMaterial size={0.02} sizeAttenuation vertexColors />
      </points>

      <Grid args={[20, 20]} cellSize={1} cellColor="#4a5568" sectionSize={5} sectionColor="#718096" fadeDistance={30} />

      <axesHelper args={[3]} />

      <OrbitControls
        autoRotate={false}
        autoRotateSpeed={4}
        enableZoom
        enablePan
        enableRotate
      />
    </>
  );
}

const SceneViewer = forwardRef<any, SceneViewerProps>(
  ({ mode, scaleField }, ref) => {
    const sceneRef = useRef<any>(null);

    useImperativeHandle(ref, () => ({
      updateVisualization: (data: Float32Array, visualMode: VisualizationMode) => {
        if (sceneRef.current) {
          sceneRef.current.updateVisualization(data, visualMode);
        }
      },
    }));

    return (
      <Canvas
        camera={{
          position: [5, 5, 5],
          fov: 50,
        }}
        style={{
          background: 'linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%)',
        }}
      >
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} intensity={0.8} />
        <SceneContent
          mode={mode}
          scaleField={scaleField}
          onUpdateRef={(r) => {
            sceneRef.current = r;
          }}
        />
      </Canvas>
    );
  }
);

SceneViewer.displayName = 'SceneViewer';

export default SceneViewer;
