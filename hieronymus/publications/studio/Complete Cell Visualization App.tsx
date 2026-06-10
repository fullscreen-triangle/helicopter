import React, { useState } from 'react';
import { useAllenCell, useHuggingFaceModel, useMeshData, useIDRData } from './hooks/useCellData';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Search, Upload, Brain, Database } from 'lucide-react';

const CompleteCellApp = () => {
  const [apiKey, setApiKey] = useState('');
  const [cellId, setCellId] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  // Hooks
  const { cellData, loading: allenLoading } = useAllenCell(cellId, apiKey);
  const { segmentCell, classifyCell, result: aiResult, loading: aiLoading } = useHuggingFaceModel(apiKey);
  const { meshData, loadSTL, loadOBJ, loading: meshLoading } = useMeshData(apiKey);
  const { projects, loadProjects, loading: idrLoading } = useIDRData(apiKey);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    
    setSelectedFile(file);
    
    if (file.name.endsWith('.stl')) {
      await loadSTL(file);
    } else if (file.name.endsWith('.obj')) {
      await loadOBJ(file);
    }
  };

  const handleImageAnalysis = async (imageFile: File) => {
    const blob = new Blob([await imageFile.arrayBuffer()], { type: imageFile.type });
    await segmentCell(blob);
  };

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-7xl mx-auto">
        <header className="mb-8">
          <h1 className="text-4xl font-bold mb-2">Cell Visualization Platform</h1>
          <p className="text-gray-600">
            Advanced 3D cell rendering with AI-powered analysis
          </p>
        </header>

        {/* API Key Input */}
        <Card className="mb-6">
          <CardHeader>
            <CardTitle>Configuration</CardTitle>
            <CardDescription>Enter your HuggingFace API key to enable AI features</CardDescription>
          </CardHeader>
          <CardContent>
            <Input
              type="password"
              placeholder="HuggingFace API Key"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              className="max-w-md"
            />
          </CardContent>
        </Card>

        <Tabs defaultValue="allen" className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="allen">
              <Database className="w-4 h-4 mr-2" />
              Allen Cell
            </TabsTrigger>
            <TabsTrigger value="upload">
              <Upload className="w-4 h-4 mr-2" />
              Upload
            </TabsTrigger>
            <TabsTrigger value="ai">
              <Brain className="w-4 h-4 mr-2" />
              AI Analysis
            </TabsTrigger>
            <TabsTrigger value="search">
              <Search className="w-4 h-4 mr-2" />
              Search
            </TabsTrigger>
          </TabsList>

          {/* Allen Cell Tab */}
          <TabsContent value="allen">
            <Card>
              <CardHeader>
                <CardTitle>Allen Cell Explorer</CardTitle>
                <CardDescription>Load cell data from Allen Cell database</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex gap-2">
                  <Input
                    placeholder="Enter Cell ID"
                    value={cellId}
                    onChange={(e) => setCellId(e.target.value)}
                  />
                  <Button disabled={allenLoading || !apiKey}>
                    {allenLoading ? 'Loading...' : 'Load Cell'}
                  </Button>
                </div>
                
                {cellData && (
                  <div className="bg-gray-100 p-4 rounded">
                    <h3 className="font-bold mb-2">Cell Data Loaded</h3>
                    <pre className="text-xs overflow-auto">
                      {JSON.stringify(cellData, null, 2)}
                    </pre>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Upload Tab */}
          <TabsContent value="upload">
            <Card>
              <CardHeader>
                <CardTitle>Upload 3D Mesh</CardTitle>
                <CardDescription>Upload STL, OBJ, or PLY files</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <Input
                    type="file"
                    accept=".stl,.obj,.ply"
                    onChange={handleFileUpload}
                  />
                </div>
                
                {meshLoading && <p>Loading mesh...</p>}
                
                {meshData && (
                  <div className="bg-gray-100 p-4 rounded">
                    <h3 className="font-bold mb-2">Mesh Loaded</h3>
                    <p className="text-sm">Vertices: {meshData.positions.length / 3}</p>
                    {meshData.normals && <p className="text-sm">Normals: {meshData.normals.length / 3}</p>}
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* AI Analysis Tab */}
          <TabsContent value="ai">
            <Card>
              <CardHeader>
                <CardTitle>AI-Powered Analysis</CardTitle>
                <CardDescription>Segment and classify cells using HuggingFace models</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <Input
                    type="file"
                    accept="image/*"
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (file) handleImageAnalysis(file);
                    }}
                  />
                </div>
                
                {aiLoading && <p>Analyzing image...</p>}
                
                {aiResult && (
                  <div className="bg-gray-100 p-4 rounded">
                    <h3 className="font-bold mb-2">Analysis Results</h3>
                    <pre className="text-xs overflow-auto max-h-64">
                      {JSON.stringify(aiResult, null, 2)}
                    </pre>
                  </div>
                )}
                
                <div className="grid grid-cols-2 gap-2">
                  <Button variant="outline" disabled={!apiKey}>
                    Segment Cells
                  </Button>
                  <Button variant="outline" disabled={!apiKey}>
                    Classify Cell Type
                  </Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Search Tab */}
          <TabsContent value="search">
            <Card>
              <CardHeader>
                <CardTitle>Search Cell Databases</CardTitle>
                <CardDescription>Search across multiple cell image databases</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex gap-2">
                  <Input
                    placeholder="Search for cell types, structures..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                  />
                  <Button disabled={!apiKey}>
                    <Search className="w-4 h-4" />
                  </Button>
                </div>
                
                <div className="text-sm text-gray-600">
                  Search across: Allen Cell, Cell Image Library, IDR
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>

        {/* 3D Viewer would be embedded here */}
        <Card className="mt-6">
          <CardHeader>
            <CardTitle>3D Visualization</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="bg-gray-200 rounded h-96 flex items-center justify-center">
              <p className="text-gray-500">3D Viewer Canvas (Embed Advanced Cell Viewer component here)</p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default CompleteCellApp;