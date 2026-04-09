'use client';

import React, { useRef, useEffect } from 'react';
import { useGLTF, useAnimations } from '@react-three/drei';
import { useFrame } from '@react-three/fiber';
import type { Group } from 'three';
import type { ThreeElements } from '@react-three/fiber';

export default function PiCamera(props: ThreeElements['group']) {
  const groupRef = useRef<Group>(null!);
  const { scene, animations } = useGLTF('/raspberry_pi_cam.glb') as any;
  const { actions } = useAnimations(animations, groupRef);

  useEffect(() => {
    if (actions) {
      Object.values(actions).forEach((action: any) => {
        if (action) {
          action.play();
        }
      });
    }
  }, [actions]);

  useFrame((_state, delta) => {
    if (groupRef.current) {
      groupRef.current.rotation.y += delta * 0.15;
    }
  });

  return (
    <group ref={groupRef} {...props} dispose={null}>
      <primitive object={scene} />
    </group>
  );
}

useGLTF.preload('/raspberry_pi_cam.glb');
