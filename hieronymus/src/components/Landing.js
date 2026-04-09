import React, { Suspense, useRef } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { ContactShadows, useGLTF, useAnimations } from '@react-three/drei'

export function Model(props) {
  const group = useRef()
  const { nodes, materials, animations } = useGLTF('/raspberry_pi_cam.glb')
  const { actions } = useAnimations(animations, group)
  return (
    <group ref={group} {...props} dispose={null}>
      <group name="Sketchfab_Scene">
        <group name="Sketchfab_model" rotation={[-Math.PI / 2, 0, 0]}>
          <group name="Root">
            <group
              name="cam4"
              position={[-2.5, -2, 0.08]}
              rotation={[Math.PI, 1.571, 0]}
              scale={0.28}>
              <mesh
                name="cam4_0"
                castShadow
                receiveShadow
                geometry={nodes.cam4_0.geometry}
                material={materials.GREY}
              />
            </group>
            <group name="cam5" position={[-2.39, -2.1, 0.1]} scale={[3.2, 0.3, 3.2]}>
              <mesh
                name="cam5_0"
                castShadow
                receiveShadow
                geometry={nodes.cam5_0.geometry}
                material={materials.FILTR_EINFRA_ROUGE}
              />
            </group>
            <group
              name="cam3"
              position={[-2.4, -3.34, 0.02]}
              rotation={[Math.PI, 1.571, 0]}
              scale={0.2}>
              <mesh
                name="cam3_0"
                castShadow
                receiveShadow
                geometry={nodes.cam3_0.geometry}
                material={materials.LENTILLE}
              />
            </group>
            <group
              name="Lamp"
              position={[-7.024, -44.705, 0.744]}
              rotation={[1.667, -0.281, -2.067]}>
              <group name="Lamp_1" />
            </group>
            <group name="cam1" rotation={[Math.PI / 2, 0, 0]}>
              <mesh
                name="cam1_0"
                castShadow
                receiveShadow
                geometry={nodes.cam1_0.geometry}
                material={materials.material_9}
              />
            </group>
            <group name="cam2" rotation={[Math.PI / 2, 0, 0]}>
              <mesh
                name="cam2_0"
                castShadow
                receiveShadow
                geometry={nodes.cam2_0.geometry}
                material={materials.GREY}
              />
            </group>
            <group
              name="Lamp001"
              position={[0.14, 33.071, -0.161]}
              rotation={[-1.602, 0.101, 1.408]}>
              <group name="Lamp001_1" />
            </group>
            <group
              name="Bottom_Overlay"
              position={[1.352, 0.019, 0.101]}
              rotation={[-Math.PI / 2, 0, -Math.PI]}
              scale={[24, 23.5, 0.01]}>
              <mesh
                name="Bottom_Overlay_0"
                castShadow
                receiveShadow
                wireframe={true}
                geometry={nodes.Bottom_Overlay_0.geometry}
                material={materials.Bottom_Overlay}
              />
            </group>
            <group
              name="cam6"
              position={[-2.5, -3, 0.08]}
              rotation={[Math.PI / 2, 0, Math.PI / 2]}
              scale={[2.8, 2.8, 0.15]}>
              <mesh
                name="cam6_0"
                castShadow
                receiveShadow
                geometry={nodes.cam6_0.geometry}
                material={materials.LENTILLE}
              />
            </group>
            <group
              name="BOARD"
              position={[-5.696, 0, -0.333]}
              rotation={[-Math.PI, 0, 0]}
              scale={[0.2, 0.52, 0.24]}>
              <mesh
                name="BOARD_0"
                castShadow
                receiveShadow
                geometry={nodes.BOARD_0.geometry}
                material={materials.solder}
              />
              <mesh
                name="BOARD_0_1"
                castShadow
                receiveShadow
                wireframe={true}
                geometry={nodes.BOARD_0_1.geometry}
                material={materials.solder}
              />
              <mesh
                name="BOARD_1"
                castShadow
                receiveShadow
                geometry={nodes.BOARD_1.geometry}
                material={materials.TEXTURE_flexcable}
              />
              <mesh
                name="BOARD_2"
                castShadow
                receiveShadow
                wireframe={true}
                geometry={nodes.BOARD_2.geometry}
                material={materials.TEXTURE_IC}
              />
              <mesh
                name="BOARD_3"
                castShadow
                receiveShadow
                geometry={nodes.BOARD_3.geometry}
                material={materials.capacitor}
              />
              <mesh
                name="BOARD_4"
                castShadow
                receiveShadow
                geometry={nodes.BOARD_4.geometry}
                material={materials.material}
              />
              <mesh
                name="BOARD_5"
                castShadow
                receiveShadow
                wireframe={true}
                geometry={nodes.BOARD_5.geometry}
                material={materials.teardrop}
              />
              <mesh
                name="BOARD_6"
                castShadow
                receiveShadow
                geometry={nodes.BOARD_6.geometry}
                material={materials.gold}
              />
              <mesh
                name="BOARD_6_1"
                castShadow
                receiveShadow
                geometry={nodes.BOARD_6_1.geometry}
                material={materials.gold}
              />
              <mesh
                name="BOARD_6_2"
                castShadow
                receiveShadow
                geometry={nodes.BOARD_6_2.geometry}
                material={materials.gold}
              />
              <mesh
                name="BOARD_7"
                castShadow
                receiveShadow
                geometry={nodes.BOARD_7.geometry}
                material={materials.white}
              />
              <mesh
                name="BOARD_8"
                castShadow
                receiveShadow
                geometry={nodes.BOARD_8.geometry}
                material={materials.Logo_RaspBerry}
              />
              <mesh
                name="BOARD_9"
                castShadow
                receiveShadow
                geometry={nodes.BOARD_9.geometry}
                material={materials.material_9}
              />
              <mesh
                name="BOARD_10"
                castShadow
                receiveShadow
                wireframe={true}
                geometry={nodes.BOARD_10.geometry}
                material={materials.Conn_black}
              />
              <mesh
                name="BOARD_11"
                castShadow
                receiveShadow
                geometry={nodes.BOARD_11.geometry}
                material={materials.flexcable}
              />
              <mesh
                name="BOARD_12"
                castShadow
                receiveShadow
                geometry={nodes.BOARD_12.geometry}
                material={materials.material_12}
              />
              <mesh
                name="BOARD_13"
                castShadow
                receiveShadow
                geometry={nodes.BOARD_13.geometry}
                material={materials.GREY}
              />
              <mesh
                name="BOARD_14"
                castShadow
                receiveShadow
                geometry={nodes.BOARD_14.geometry}
                material={materials.GREEN}
              />
              <mesh
                name="BOARD_15"
                castShadow
                receiveShadow
                wireframe={true}
                geometry={nodes.BOARD_15.geometry}
                material={materials.TEXTURE_SENSOR}
              />
            </group>
          </group>
        </group>
      </group>
    </group>
  )
}

useGLTF.preload('/raspberry_pi_cam.glb')

export function App() {
  return (
    <Canvas shadows dpr={[1, 2]} camera={{ position: [0, 0, 1.1], fov: 50 }}>
      <ambientLight intensity={2} />
      <spotLight position={[1, 6, 1.5]} angle={0.2} penumbra={1} intensity={2.5} castShadow shadow-mapSize={[2048, 2048]} />
      <spotLight position={[-5, 5, -1.5]} angle={0.03} penumbra={1} intensity={4} castShadow shadow-mapSize={[1024, 1024]} />
      <spotLight position={[5, 5, -5]} angle={0.3} penumbra={1} intensity={4} castShadow={true} shadow-mapSize={[256, 256]} color="#ffffc0" />
      <Suspense fallback={null}>
        <Model scale={0.225} position={[0, -0.09, 0]} />
        <ContactShadows frames={1} rotation-x={[Math.PI / 2]} position={[0, -0.33, 0]} far={0.4} width={2} height={2} blur={4} />
      </Suspense>
    </Canvas>
  )
}


import styled from 'styled-components'

const Grid = styled.div`
  position: absolute;
  width: 100%;
  height: 100%;
  margin: 0;
  padding: 0;
  display: grid;
  grid-template-columns: 1fr auto;
  align-items: start;
  font-family: 'ThunderBlack';
  font-size: 4em;
  line-height: 0.74em;
  color: #f7f4af;

  @media (max-width: 425px) {
    grid-template-columns: 1fr;
  }
`

const Left = styled.div`
  position: relative;
  display: grid;
  grid-template-rows: 1fr auto;
  width: 100%;
  height: 100%;
  padding: 100px;
  white-space: nowrap;
  font-family: 'ThunderThin';
  line-height: 1em;
  @media (max-width: 768px) {
    padding: 50px;
  }
`

const Right = styled.div`
  position: relative;
  display: flex;
  width: 100%;
  height: 100%;
  background: #f7f4af;
  color: #252525;
`

const Sub = styled.div`
  align-self: end;
  width: 200px;
  height: 2px;
  background: #f7f4af;
`

const Jumbo = styled.div`
  align-self: center;
  padding: 100px;
  font-size: 3.5em;
  color: #252525;
`

const Label = styled.div`
  position: absolute;
  bottom: 0;
  right: 0;
  font-family: 'ThunderThin';
  font-size: 0.2em;
  line-height: 1em;
  color: #252525;
  padding: 100px;

  @media (max-width: 768px) {
    padding: 50px;
  }

  @media (max-width: 425px) {
    padding: 20px;
  }
`

export default function Underlay() {
  return (
    <Grid>
      <Left>
        <div>
          THE SISTRUM
          <br />
          IS DESIGNED
          <br />
          TO REPRESENT
          <br />
          CONTINUAL
          <br />
          AGITATION.
        </div>
        <Sub />
      </Left>
      <Right>
        <Jumbo>M1</Jumbo>
      </Right>
    </Grid>
  )
}
