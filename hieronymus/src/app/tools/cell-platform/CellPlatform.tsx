'use client';

import React, { useState, useRef } from 'react';
import { useAllenCell, useHuggingFaceModel, useMeshData, useIDRData } from '@/lib/cell-data/useCellData';
import { Search, Upload, Brain, Database, AlertCircle, CheckCircle } from 'lucide-react';

// ── Inline primitives (no shadcn dependency) ──────────────────────────────────

function Card({ children, className = '' }: { children: React.ReactNode; className?: string }) {
  return <div className={`bg-[#252526] border border-[#3a3a3a] rounded-lg ${className}`}>{children}</div>;
}
function CardHeader({ children }: { children: React.ReactNode }) {
  return <div className="px-4 pt-4 pb-2 border-b border-[#3a3a3a]">{children}</div>;
}
function CardTitle({ children }: { children: React.ReactNode }) {
  return <h3 className="text-white font-semibold text-sm">{children}</h3>;
}
function CardDesc({ children }: { children: React.ReactNode }) {
  return <p className="text-[#858585] text-xs mt-0.5">{children}</p>;
}
function CardContent({ children }: { children: React.ReactNode }) {
  return <div className="p-4 space-y-3">{children}</div>;
}
function Input({ placeholder, value, onChange, type = 'text', className = '' }: any) {
  return (
    <input type={type} placeholder={placeholder} value={value} onChange={onChange}
      className={`bg-[#1e1e1e] border border-[#3a3a3a] text-[#d4d4d4] placeholder-[#555] rounded px-3 py-1.5 text-sm outline-none focus:border-[#007acc] ${className}`} />
  );
}
function Button({ children, onClick, disabled, variant = 'primary', className = '' }: any) {
  const base = 'flex items-center gap-1.5 px-3 py-1.5 rounded text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed';
  const styles: Record<string, string> = {
    primary: 'bg-[#007acc] hover:bg-[#0098ff] text-white',
    outline: 'border border-[#3a3a3a] text-[#d4d4d4] hover:bg-[#2a2a2a]',
  };
  return <button className={`${base} ${styles[variant] ?? styles.primary} ${className}`} onClick={onClick} disabled={disabled}>{children}</button>;
}

// ── Tab structure ─────────────────────────────────────────────────────────────

type Tab = 'allen' | 'upload' | 'ai' | 'search';

export default function CellPlatform() {
  const [apiKey, setApiKey]           = useState('');
  const [tab, setTab]                 = useState<Tab>('allen');
  const [cellId, setCellId]           = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<string | null>(null);
  const [aiFile, setAiFile]           = useState<File | null>(null);
  const fileRef  = useRef<HTMLInputElement>(null);
  const aiRef    = useRef<HTMLInputElement>(null);

  const { cellData, loading: allenLoading, error: allenError } = useAllenCell(cellId || null, apiKey);
  const { segmentCell, classifyCell, detectCells, result: aiResult, loading: aiLoading, error: aiError } = useHuggingFaceModel(apiKey);
  const { meshData, loadSTL, loadOBJ, loading: meshLoading, error: meshError } = useMeshData(apiKey);
  const { projects, loadProjects, loading: idrLoading } = useIDRData(apiKey);

  const handleMeshUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (file.name.endsWith('.stl')) await loadSTL(file).catch(() => {});
    else if (file.name.endsWith('.obj')) await loadOBJ(file).catch(() => {});
  };

  const handleAIUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setAiFile(file);
  };

  const runSegmentation = async () => {
    if (!aiFile) return;
    const blob = new Blob([await aiFile.arrayBuffer()], { type: aiFile.type });
    await segmentCell(blob).catch(() => {});
  };

  const runClassification = async () => {
    if (!aiFile) return;
    const blob = new Blob([await aiFile.arrayBuffer()], { type: aiFile.type });
    await classifyCell(blob).catch(() => {});
  };

  const handleSearch = async () => {
    if (!searchQuery) return;
    setSearchResults(null);
    try {
      const res = await fetch(`/api/image-proxy?db=BBBC&dataset=BBBC007&image=${encodeURIComponent(searchQuery)}`);
      const json = await res.json();
      setSearchResults(json.error ? `Not found: ${json.error}` : `Found: ${json.width}×${json.height}px image`);
    } catch { setSearchResults('Search failed'); }
  };

  const TABS: Array<{ id: Tab; label: string; Icon: any }> = [
    { id: 'allen', label: 'Allen Cell', Icon: Database },
    { id: 'upload', label: 'Upload Mesh', Icon: Upload },
    { id: 'ai', label: 'AI Analysis', Icon: Brain },
    { id: 'search', label: 'Search', Icon: Search },
  ];

  return (
    <div className="min-h-screen bg-[#1e1e1e] text-[#d4d4d4] font-mono text-sm">
      {/* Header */}
      <div className="px-6 py-4 bg-[#252526] border-b border-[#3a3a3a]">
        <h1 className="text-xl font-bold text-white">Cell Visualization Platform</h1>
        <p className="text-[#858585] text-xs mt-1">Advanced 3D cell rendering with AI-powered analysis · HuggingFace integration</p>
      </div>

      <div className="max-w-5xl mx-auto p-4 space-y-4">

        {/* API Key */}
        <Card>
          <CardHeader>
            <CardTitle>Configuration</CardTitle>
            <CardDesc>HuggingFace API key enables AI segmentation, classification, and feature extraction</CardDesc>
          </CardHeader>
          <CardContent>
            <div className="flex gap-2 items-center">
              <Input type="password" placeholder="HuggingFace API Key (hf_…)" value={apiKey}
                onChange={(e: any) => setApiKey(e.target.value)} className="flex-1 max-w-md" />
              {apiKey && <CheckCircle size={16} className="text-[#4caf50]" />}
            </div>
            {!apiKey && (
              <p className="text-[#ffa500] text-xs flex items-center gap-1 mt-1">
                <AlertCircle size={12} /> Without a key, AI features are disabled. BBBC007 local images always work.
              </p>
            )}
          </CardContent>
        </Card>

        {/* Tab bar */}
        <div className="flex gap-1 border-b border-[#3a3a3a]">
          {TABS.map(({ id, label, Icon }) => (
            <button key={id} onClick={() => setTab(id)}
              className={`flex items-center gap-1.5 px-4 py-2 text-xs border-b-2 transition-colors ${
                tab === id ? 'border-[#007acc] text-white' : 'border-transparent text-[#858585] hover:text-[#d4d4d4]'
              }`}>
              <Icon size={13} /> {label}
            </button>
          ))}
        </div>

        {/* Allen Cell tab */}
        {tab === 'allen' && (
          <Card>
            <CardHeader>
              <CardTitle>Allen Cell Explorer</CardTitle>
              <CardDesc>Load cell morphology data from the Allen Cell Institute API</CardDesc>
            </CardHeader>
            <CardContent>
              <div className="flex gap-2">
                <Input placeholder="Cell ID (e.g. 1234)" value={cellId}
                  onChange={(e: any) => setCellId(e.target.value)} className="flex-1 max-w-xs" />
                <Button disabled={allenLoading || !apiKey} onClick={() => {}}>
                  {allenLoading ? 'Loading…' : 'Load Cell'}
                </Button>
              </div>
              {allenError && <div className="text-[#f44336] text-xs">{allenError.message}</div>}
              {cellData && (
                <pre className="bg-[#1e1e1e] border border-[#3a3a3a] rounded p-3 text-xs overflow-auto max-h-64">
                  {JSON.stringify(cellData, null, 2)}
                </pre>
              )}
              {!apiKey && (
                <div className="text-[#858585] text-xs border border-[#3a3a3a] rounded p-3">
                  Enter a HuggingFace API key above to enable live Allen Cell queries.
                  <br />Local BBBC007 images are always available via the Microscopy Viewer tool.
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {/* Upload tab */}
        {tab === 'upload' && (
          <Card>
            <CardHeader>
              <CardTitle>Upload 3D Mesh</CardTitle>
              <CardDesc>Upload STL or OBJ files — parsed client-side, vertices displayed below</CardDesc>
            </CardHeader>
            <CardContent>
              <div className="flex gap-2">
                <Button variant="outline" onClick={() => fileRef.current?.click()}>
                  <Upload size={12} /> Choose STL / OBJ
                </Button>
                <input ref={fileRef} type="file" accept=".stl,.obj,.ply" className="hidden" onChange={handleMeshUpload} />
              </div>
              {meshLoading && <div className="text-[#ffa500] text-xs">Parsing mesh…</div>}
              {meshError && <div className="text-[#f44336] text-xs">{meshError.message}</div>}
              {meshData && (
                <div className="bg-[#1e1e1e] border border-[#3a3a3a] rounded p-3 space-y-1 text-xs">
                  <div className="text-[#4ec9b0] font-semibold">Mesh loaded</div>
                  <div>Vertices: {(meshData.positions.length / 3).toLocaleString()}</div>
                  {meshData.normals && <div>Normals: {(meshData.normals.length / 3).toLocaleString()}</div>}
                  {meshData.indices && <div>Faces: {(meshData.indices.length / 3).toLocaleString()}</div>}
                  <div className="text-[#858585]">Open the 3D Cell Viewer tool to render this mesh interactively.</div>
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {/* AI Analysis tab */}
        {tab === 'ai' && (
          <Card>
            <CardHeader>
              <CardTitle>AI-Powered Cell Analysis</CardTitle>
              <CardDesc>
                Segmentation: facebook/mask2former-swin-large-coco-instance ·
                Classification: microsoft/resnet-50 ·
                Detection: facebook/detr-resnet-50
              </CardDesc>
            </CardHeader>
            <CardContent>
              <div className="flex gap-2 items-center">
                <Button variant="outline" onClick={() => aiRef.current?.click()}>
                  <Upload size={12} /> Select image
                </Button>
                <input ref={aiRef} type="file" accept="image/*" className="hidden" onChange={handleAIUpload} />
                {aiFile && <span className="text-[#858585] text-xs">{aiFile.name}</span>}
              </div>

              <div className="flex gap-2">
                <Button onClick={runSegmentation} disabled={!apiKey || !aiFile || aiLoading}>
                  {aiLoading ? 'Running…' : 'Segment Cells'}
                </Button>
                <Button variant="outline" onClick={runClassification} disabled={!apiKey || !aiFile || aiLoading}>
                  Classify Cell Type
                </Button>
              </div>

              {!apiKey && (
                <div className="text-[#ffa500] text-xs flex items-center gap-1">
                  <AlertCircle size={12} /> API key required for HuggingFace inference
                </div>
              )}
              {aiError && <div className="text-[#f44336] text-xs">{aiError.message}</div>}
              {aiResult && (
                <div className="space-y-2">
                  <div className="text-[#4ec9b0] text-xs font-semibold">Analysis Results</div>
                  <pre className="bg-[#1e1e1e] border border-[#3a3a3a] rounded p-3 text-xs overflow-auto max-h-64">
                    {JSON.stringify(aiResult, null, 2)}
                  </pre>
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {/* Search tab */}
        {tab === 'search' && (
          <Card>
            <CardHeader>
              <CardTitle>Search Cell Databases</CardTitle>
              <CardDesc>Search BBBC007 local dataset · Allen Cell · Cell Image Library · IDR</CardDesc>
            </CardHeader>
            <CardContent>
              <div className="flex gap-2">
                <Input placeholder="Image name or cell type… (e.g. A9 p10d.tif)"
                  value={searchQuery} onChange={(e: any) => setSearchQuery(e.target.value)}
                  className="flex-1" />
                <Button onClick={handleSearch} disabled={!searchQuery}>
                  <Search size={12} /> Search
                </Button>
              </div>

              {searchResults && (
                <div className={`text-xs p-2 rounded border ${
                  searchResults.startsWith('Found') ? 'border-[#4caf50] text-[#4ec9b0]' : 'border-[#f44336] text-[#f44336]'
                }`}>{searchResults}</div>
              )}

              <div>
                <Button variant="outline" onClick={() => loadProjects().catch(() => {})} disabled={idrLoading}>
                  {idrLoading ? 'Loading…' : 'Load IDR Projects'}
                </Button>
              </div>

              {projects.length > 0 && (
                <div className="bg-[#1e1e1e] border border-[#3a3a3a] rounded p-3 text-xs">
                  <div className="text-[#4ec9b0] mb-1">IDR Projects ({projects.length})</div>
                  <pre className="overflow-auto max-h-40">{JSON.stringify(projects.slice(0, 5), null, 2)}</pre>
                </div>
              )}

              <div className="text-[#555] text-xs">
                Local BBBC007 images: A9 p5d/p7d/p9d/p10d (DAPI), p5f/p7f/p9f/p10f (fluorescence),
                f96/f9620/f113 series — all served via /api/image-proxy
              </div>
            </CardContent>
          </Card>
        )}

        {/* 3D Viewer embed note */}
        <Card>
          <CardHeader>
            <CardTitle>3D Visualization</CardTitle>
            <CardDesc>Full interactive viewer with OrbitControls, organelle overlays, and slicing</CardDesc>
          </CardHeader>
          <CardContent>
            <div className="bg-[#111] rounded h-32 flex items-center justify-center border border-[#3a3a3a]">
              <a href="/tools/cell-viewer-3d"
                className="px-4 py-2 bg-[#007acc] hover:bg-[#0098ff] text-white rounded text-sm font-medium">
                Open 3D Cell Viewer →
              </a>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
