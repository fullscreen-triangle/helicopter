import React, { useRef, useState, useEffect, Suspense } from 'react';
import { Canvas, useFrame, useLoader } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Environment, Html } from '@react-three/drei';
import { Loader2, Upload, Download, Slice, Layers, ZoomIn, ZoomOut } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';

// Cell Mesh Component
const CellMesh = ({ meshData, color = "#4a90e2", opacity = 0.8 }) => {
  const meshRef = useRef();
  const [hovered, setHovered] = useState(false);
  
  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.y = state.clock.getElapsedTime() * 0.1;
    }
  });

  if (!meshData) return null;

  return (
    <mesh
      ref={meshRef}
      onPointerOver={() => setHovered(true)}
      onPointerOut={() => setHovered(false)}
      scale={hovered ? 1.05 : 1}
    >
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={meshData.positions.length / 3}
          array={meshData.positions}
          itemSize={3}
        />
        {meshData.normals && (
          <bufferAttribute
            attach="attributes-normal"
            count={meshData.normals.length / 3}
            array={meshData.normals}
            itemSize={3}
          />
        )}
        {meshData.indices && (
          <bufferAttribute
            attach="index"
            count={meshData.indices.length}
            array={meshData.indices}
            itemSize={1}
          />
        )}
      </bufferGeometry>
      <meshStandardMaterial
        color={color}
        opacity={opacity}
        transparent
        roughness={0.3}
        metalness={0.2}
      />
    </mesh>
  );
};

// Nucleus Component
const Nucleus = ({ visible, color = "#ff6b6b", size = 0.4 }) => {
  if (!visible) return null;
  
  return (
    <mesh position={[0, 0, 0]}>
      <sphereGeometry args={[size, 32, 32]} />
      <meshStandardMaterial
        color={color}
        opacity={0.7}
        transparent
        roughness={0.4}
      />
    </mesh>
  );
};

// Organelle Component
const Organelle = ({ position, size, color, label }) => {
  const [showLabel, setShowLabel] = useState(false);
  
  return (
    <mesh
      position={position}
      onPointerOver={() => setShowLabel(true)}
      onPointerOut={() => setShowLabel(false)}
    >
      <sphereGeometry args={[size, 16, 16]} />
      <meshStandardMaterial color={color} opacity={0.8} transparent />
      {showLabel && (
        <Html distanceFactor={10}>
          <div className="bg-black text-white px-2 py-1 rounded text-xs">
            {label}
          </div>
        </Html>
      )}
    </mesh>
  );
};

// Slicing Plane Component
const SlicingPlane = ({ position, visible }) => {
  if (!visible) return null;
  
  return (
    <mesh position={[0, position, 0]} rotation={[Math.PI / 2, 0, 0]}>
      <planeGeometry args={[3, 3]} />
      <meshBasicMaterial color="#00ff00" opacity={0.2} transparent side={2} />
    </mesh>
  );
};

// Main Viewer Component
const AdvancedCellViewer = () => {
  const [meshData, setMeshData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showNucleus, setShowNucleus] = useState(true);
  const [showOrganelles, setShowOrganelles] = useState(true);
  const [cellOpacity, setCellOpacity] = useState(0.8);
  const [slicePosition, setSlicePosition] = useState(0);
  const [showSlice, setShowSlice] = useState(false);
  const [dataSource, setDataSource] = useState('allen');
  const [cellColor, setCellColor] = useState('#4a90e2');
  const [autoRotate, setAutoRotate] = useState(true);

  // Sample organelles data
  const organelles = [
    { position: [0.3, 0.2, 0.1], size: 0.15, color: '#ffd700', label: 'Mitochondria' },
    { position: [-0.2, 0.3, 0.2], size: 0.12, color: '#9370db', label: 'Golgi' },
    { position: [0.1, -0.3, 0.15], size: 0.1, color: '#20b2aa', label: 'Ribosome' },
    { position: [-0.3, -0.2, -0.1], size: 0.13, color: '#ff69b4', label: 'Lysosome' },
  ];

  // Generate sample mesh data
  const generateSampleMesh = () => {
    const segments = 32;
    const rings = 16;
    const positions = [];
    const normals = [];
    
    for (let lat = 0; lat <= rings; lat++) {
      const theta = (lat * Math.PI) / rings;
      const sinTheta = Math.sin(theta);
      const cosTheta = Math.cos(theta);
      
      for (let lon = 0; lon <= segments; lon++) {
        const phi = (lon * 2 * Math.PI) / segments;
        const sinPhi = Math.sin(phi);
        const cosPhi = Math.cos(phi);
        
        const x = cosPhi * sinTheta;
        const y = cosTheta;
        const z = sinPhi * sinTheta;
        
        // Add some distortion for organic look
        const distortion = 1 + 0.2 * Math.sin(4 * phi) * Math.sin(4 * theta);
        
        positions.push(x * distortion, y * distortion, z * distortion);
        normals.push(x, y, z);
      }
    }
    
    return {
      positions: new Float32Array(positions),
      normals: new Float32Array(normals),
    };
  };

  useEffect(() => {
    // Load sample data on mount
    setMeshData(generateSampleMesh());
  }, []);

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    
    setLoading(true);
    
    try {
      const arrayBuffer = await file.arrayBuffer();
      const text = new TextDecoder().decode(arrayBuffer);
      
      // Simple OBJ parser (you'd use the service module in production)
      if (file.name.endsWith('.obj')) {
        // Parse OBJ file
        setMeshData(generateSampleMesh()); // Placeholder
      } else if (file.name.endsWith('.stl')) {
        // Parse STL file
        setMeshData(generateSampleMesh()); // Placeholder
      }
    } catch (error) {
      console.error('Error loading file:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleExport = () => {
    // Export current view as image
    alert('Export functionality - would capture canvas and download');
  };

  return (
    <div className="w-full h-screen bg-gray-900 relative">
      {/* Control Panel */}
      <div className="absolute top-4 left-4 z-10 bg-white rounded-lg shadow-lg p-4 w-64 space-y-4">
        <h2 className="text-xl font-bold">3D Cell Viewer</h2>
        
        {/* Data Source Selection */}
        <div>
          <label className="text-sm font-medium mb-1 block">Data Source</label>
          <Select value={dataSource} onValueChange={setDataSource}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="allen">Allen Cell</SelectItem>
              <SelectItem value="cil">Cell Image Library</SelectItem>
              <SelectItem value="idr">IDR Database</SelectItem>
              <SelectItem value="local">Local File</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* File Upload */}
        <div>
          <label className="text-sm font-medium mb-1 block">Upload Mesh</label>
          <Button variant="outline" className="w-full" asChild>
            <label>
              <Upload className="w-4 h-4 mr-2" />
              Choose File
              <input
                type="file"
                accept=".obj,.stl,.ply"
                onChange={handleFileUpload}
                className="hidden"
              />
            </label>
          </Button>
        </div>

        {/* Visibility Controls */}
        <div className="space-y-2">
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={showNucleus}
              onChange={(e) => setShowNucleus(e.target.checked)}
              className="w-4 h-4"
            />
            <span className="text-sm">Show Nucleus</span>
          </label>
          
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={showOrganelles}
              onChange={(e) => setShowOrganelles(e.target.checked)}
              className="w-4 h-4"
            />
            <span className="text-sm">Show Organelles</span>
          </label>
          
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={autoRotate}
              onChange={(e) => setAutoRotate(e.target.checked)}
              className="w-4 h-4"
            />
            <span className="text-sm">Auto Rotate</span>
          </label>
        </div>

        {/* Opacity Control */}
        <div>
          <label className="text-sm font-medium mb-2 block">
            Cell Opacity: {cellOpacity.toFixed(2)}
          </label>
          <Slider
            value={[cellOpacity]}
            onValueChange={(v) => setCellOpacity(v[0])}
            min={0}
            max={1}
            step={0.01}
            className="w-full"
          />
        </div>

        {/* Slicing Control */}
        <div>
          <label className="flex items-center gap-2 mb-2">
            <input
              type="checkbox"
              checked={showSlice}
              onChange={(e) => setShowSlice(e.target.checked)}
              className="w-4 h-4"
            />
            <span className="text-sm font-medium">Enable Slicing</span>
          </label>
          {showSlice && (
            <Slider
              value={[slicePosition]}
              onValueChange={(v) => setSlicePosition(v[0])}
              min={-1.5}
              max={1.5}
              step={0.1}
              className="w-full"
            />
          )}
        </div>

        {/* Color Picker */}
        <div>
          <label className="text-sm font-medium mb-1 block">Cell Color</label>
          <input
            type="color"
            value={cellColor}
            onChange={(e) => setCellColor(e.target.value)}
            className="w-full h-10 rounded cursor-pointer"
          />
        </div>

        {/* Export Button */}
        <Button onClick={handleExport} className="w-full">
          <Download className="w-4 h-4 mr-2" />
          Export View
        </Button>
      </div>

      {/* Info Panel */}
      <div className="absolute top-4 right-4 z-10 bg-white rounded-lg shadow-lg p-4 w-48">
        <h3 className="font-bold mb-2">Cell Info</h3>
        <div className="text-xs space-y-1">
          <p><strong>Type:</strong> Stem Cell</p>
          <p><strong>Source:</strong> {dataSource}</p>
          <p><strong>Vertices:</strong> {meshData ? (meshData.positions.length / 3).toFixed(0) : 0}</p>
        </div>
      </div>

      {/* Loading Indicator */}
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50 z-20">
          <div className="bg-white rounded-lg p-6 flex items-center gap-3">
            <Loader2 className="w-6 h-6 animate-spin" />
            <span>Loading cell data...</span>
          </div>
        </div>
      )}

      {/* 3D Canvas */}
      <Canvas>
        <PerspectiveCamera makeDefault position={[0, 0, 5]} fov={50} />
        
        {/* Lighting */}
        <ambientLight intensity={0.4} />
        <pointLight position={[10, 10, 10]} intensity={1} />
        <pointLight position={[-10, -10, -10]} intensity={0.5} />
        <spotLight position={[0, 10, 0]} angle={0.3} intensity={0.5} />
        
        {/* Environment */}
        <Environment preset="city" />
        
        {/* Cell Components */}
        <Suspense fallback={null}>
          <CellMesh meshData={meshData} color={cellColor} opacity={cellOpacity} />
          <Nucleus visible={showNucleus} />
          {showOrganelles && organelles.map((org, idx) => (
            <Organelle key={idx} {...org} />
          ))}
          <SlicingPlane position={slicePosition} visible={showSlice} />
        </Suspense>
        
        {/* Controls */}
        <OrbitControls
          autoRotate={autoRotate}
          autoRotateSpeed={0.5}
          enableZoom={true}
          enablePan={true}
          minDistance={2}
          maxDistance={10}
        />
        
        {/* Grid Helper */}
        <gridHelper args={[10, 10, '#444444', '#222222']} />
      </Canvas>
    </div>
  );
};

export default AdvancedCellViewer;