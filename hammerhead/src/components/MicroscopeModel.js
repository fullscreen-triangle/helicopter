/* eslint-disable react/no-unknown-property */
import React, { useRef, useEffect, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { useGLTF, useAnimations, OrbitControls, ContactShadows } from '@react-three/drei';

function Model(props) {
    const group = useRef();
    const { nodes, materials, animations, scene } = useGLTF('/compound_microscope.glb');
    const { actions, names } = useAnimations(animations, group);
    const [hoveredPart, setHoveredPart] = useState(null);

    useEffect(() => {
        // Play all animations by default
        names.forEach((name) => {
            if (actions[name]) {
                actions[name].reset().play();
                actions[name].paused = true;
            }
        });
    }, [actions, names]);

    // Slow auto-rotation
    useFrame((state) => {
        if (group.current) {
            group.current.rotation.y += 0.002;
        }
    });

    const handleClick = (partName) => {
        // Find and play an animation related to this part
        names.forEach((name) => {
            if (actions[name]) {
                actions[name].paused = false;
                actions[name].reset().play();
                // Pause again after one cycle
                setTimeout(() => {
                    if (actions[name]) {
                        actions[name].paused = true;
                    }
                }, 2000);
            }
        });
    };

    const interactiveParts = ['eye_glass', 'KNOB', 'EyeRotate', 'FineAdjustmentKnob', 'Mirror_Base'];

    return (
        <group ref={group} {...props} dispose={null}>
            <primitive
                object={scene}
                scale={2.5}
                position={[0, -1.5, 0]}
                onClick={(e) => {
                    e.stopPropagation();
                    const name = e.object.name;
                    handleClick(name);
                }}
                onPointerOver={(e) => {
                    e.stopPropagation();
                    if (interactiveParts.includes(e.object.name)) {
                        setHoveredPart(e.object.name);
                        document.body.style.cursor = 'pointer';
                    }
                }}
                onPointerOut={(e) => {
                    setHoveredPart(null);
                    document.body.style.cursor = 'auto';
                }}
            />
        </group>
    );
}

export default function MicroscopeScene() {
    return (
        <div style={{ width: '100%', height: '100%', position: 'absolute', top: 0, left: 0 }}>
            <Canvas
                camera={{ position: [3, 2, 5], fov: 45 }}
                style={{ background: 'transparent' }}
                gl={{ alpha: true, antialias: true }}
            >
                <ambientLight intensity={0.4} />
                <directionalLight position={[5, 5, 5]} intensity={0.8} castShadow />
                <directionalLight position={[-3, 3, -3]} intensity={0.3} />
                <pointLight position={[0, 3, 0]} intensity={0.5} color="#4a9eda" />
                <Model />
                <ContactShadows
                    position={[0, -1.5, 0]}
                    opacity={0.4}
                    scale={8}
                    blur={2}
                />
                <OrbitControls
                    enableZoom={false}
                    enablePan={false}
                    minPolarAngle={Math.PI / 4}
                    maxPolarAngle={Math.PI / 2}
                    autoRotate={false}
                />
            </Canvas>
        </div>
    );
}

// Preload the model
useGLTF.preload('/compound_microscope.glb');
