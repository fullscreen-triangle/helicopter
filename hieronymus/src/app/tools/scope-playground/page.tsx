'use client';

import React, { useReducer, useRef, useCallback, useEffect, useState } from 'react';
import { Play, RotateCcw, Download, ChevronDown, ChevronRight, FileCode2, FolderOpen, Folder } from 'lucide-react';
import { compile } from '@/lib/scope-compiler';
import { runScope, ScopeResult } from '@/lib/scope-runtime/runtime';
import { SCOPE_FOLDERS, ScopeScript, ScopeFolder } from '@/lib/scope-scripts';
import Canvas2D from './components/visualise/Canvas2D';
import SpectralPowerChart from './components/charts/SpectralPowerChart';
import EntropyTrajectoryChart from './components/charts/EntropyTrajectoryChart';
import UncertaintyBar from './components/charts/UncertaintyBar';
import ScaleHistogram from './components/charts/ScaleHistogram';
import EntropySphere from './components/threed/EntropySphere';
import DistanceTube from './components/threed/DistanceTube';
import ScaleFieldSurface from './components/threed/ScaleFieldSurface';
import PointCloud from './components/threed/PointCloud';

interface ImagePayload { data: Float32Array; width: number; height: number; synthetic?: boolean; }

// ── State ────────────────────────────────────────────────────────────────────

interface State {
  activeScriptId: string;
  source: string;
  running: boolean;
  result: ScopeResult | null;
  log: string[];
  error: string | null;
  activeTab: 'visualise' | 'charts' | '3d';
  activeVisMode: string;
  activeChartMode: string;
  active3DMode: string;
}

type Action =
  | { type: 'OPEN_SCRIPT'; script: ScopeScript }
  | { type: 'SET_SOURCE'; source: string }
  | { type: 'RUN_START' }
  | { type: 'RUN_OK'; result: ScopeResult; log: string[] }
  | { type: 'RUN_ERR'; error: string; log: string[] }
  | { type: 'CLEAR' }
  | { type: 'SET_TAB'; tab: State['activeTab'] }
  | { type: 'SET_VIS'; mode: string }
  | { type: 'SET_CHART'; mode: string }
  | { type: 'SET_3D'; mode: string };

const FIRST = SCOPE_FOLDERS[0].scripts[0];

const INITIAL: State = {
  activeScriptId: FIRST.id,
  source: FIRST.source,
  running: false,
  result: null,
  log: [],
  error: null,
  activeTab: 'visualise',
  activeVisMode: 'raw_image',
  activeChartMode: 'spectral_power',
  active3DMode: 'point_cloud',
};

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case 'OPEN_SCRIPT':
      return { ...state, activeScriptId: action.script.id, source: action.script.source, result: null, log: [], error: null };
    case 'SET_SOURCE':   return { ...state, source: action.source };
    case 'RUN_START':    return { ...state, running: true, error: null, log: [] };
    case 'RUN_OK':       return { ...state, running: false, result: action.result, log: action.log, error: null };
    case 'RUN_ERR':      return { ...state, running: false, error: action.error, log: action.log };
    case 'CLEAR':        return { ...state, result: null, log: [], error: null };
    case 'SET_TAB':      return { ...state, activeTab: action.tab };
    case 'SET_VIS':      return { ...state, activeVisMode: action.mode, activeTab: 'visualise' };
    case 'SET_CHART':    return { ...state, activeChartMode: action.mode, activeTab: 'charts' };
    case 'SET_3D':       return { ...state, active3DMode: action.mode, activeTab: '3d' };
    default:             return state;
  }
}

// ── Main page ────────────────────────────────────────────────────────────────

export default function ScopePlayground() {
  const [state, dispatch] = useReducer(reducer, INITIAL);
  const [preloadImage, setPreloadImage] = useState<ImagePayload | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Fetch preview image for the active script whenever the source changes.
  // Parse the first load() call directly from the source text to avoid
  // running the full compiler just for a preview.
  useEffect(() => {
    const m = state.source.match(/load\s*\(\s*db\s*=\s*"([^"]+)"\s*,\s*dataset\s*=\s*"([^"]+)"\s*,\s*image\s*=\s*"([^"]+)"/);
    if (!m) return;
    const [, db, dataset, image] = m;
    const params = new URLSearchParams({ db, dataset, image });
    fetch(`/api/image-proxy?${params}`)
      .then(r => r.json())
      .then(json => {
        if (!json.error) setPreloadImage({
          data: new Float32Array(json.data as number[]),
          width: json.width, height: json.height, synthetic: json.synthetic ?? false,
        });
      }).catch(() => {});
  }, [state.source]);

  const run = useCallback(async () => {
    dispatch({ type: 'RUN_START' });
    const log: string[] = [];
    try {
      const cr = compile(state.source);
      log.push(...cr.log);
      if (!cr.ok || !cr.program) {
        dispatch({ type: 'RUN_ERR', error: cr.errors[0]?.kind ?? 'CompileError', log });
        return;
      }
      const program = cr.program;

      // Fetch one real image per morphism (keyed by morphism name)
      const morphismImages: Record<string, ImagePayload> = {};
      let primaryPayload: ImagePayload | null = null;

      for (const morphism of program.morphisms) {
        const frame = morphism.expr.observe.frame;
        if (frame.kind !== 'LoadRef') continue;
        const params = new URLSearchParams({ db: frame.db, dataset: frame.dataset, image: frame.image });
        log.push(`[FETCH]    loading ${frame.dataset}/${frame.image} (${morphism.name})`);
        const res = await fetch(`/api/image-proxy?${params}`);
        const json = await res.json();
        if (json.error) throw new Error(json.error);
        const payload: ImagePayload = {
          data: new Float32Array(json.data as number[]),
          width: json.width,
          height: json.height,
        };
        if (json.synthetic) log.push(`[FETCH]    ⚠ ${morphism.name}: synthetic fallback`);
        else log.push(`[FETCH]    ${morphism.name}: ${json.width}×${json.height} real image`);
        morphismImages[morphism.name] = payload;
        if (!primaryPayload) primaryPayload = payload;
      }

      // Fallback: no LoadRef morphisms — use synthetic
      if (!primaryPayload) {
        const W = 256, H = 256, data = new Float32Array(W * H);
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
          const d1 = (x-80)**2+(y-128)**2, d2 = (x-176)**2+(y-128)**2;
          data[y*W+x] = Math.max(0, Math.min(1, 0.05+0.9*Math.exp(-d1/1250)+0.9*Math.exp(-d2/968)));
        }
        primaryPayload = { data, width: W, height: H };
        log.push(`[FETCH]    synthetic 256×256 image`);
      }

      setPreloadImage(primaryPayload);
      log.push(`[RUNTIME]  running pipeline...`);
      const result = await runScope(program, primaryPayload, morphismImages);
      log.push(...result.log.filter(l => !log.includes(l)));
      const vis = result.visualData.activeVisMode;
      dispatch({ type: 'RUN_OK', result, log });
      if (vis) {
        const tabFor = visToTab(vis);
        if (tabFor === 'visualise') dispatch({ type: 'SET_VIS', mode: vis });
        else if (tabFor === 'charts') dispatch({ type: 'SET_CHART', mode: vis });
        else if (tabFor === '3d') dispatch({ type: 'SET_3D', mode: vis });
      }
    } catch (err) {
      log.push(`[ERROR]    ${err instanceof Error ? err.message : String(err)}`);
      dispatch({ type: 'RUN_ERR', error: String(err), log });
    }
  }, [state.source]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') { e.preventDefault(); run(); }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [run]);

  const { result, log, error, running, activeTab } = state;

  // Find active script for breadcrumb
  const activeFolder = SCOPE_FOLDERS.find(f => f.scripts.some(s => s.id === state.activeScriptId));
  const activeScript = activeFolder?.scripts.find(s => s.id === state.activeScriptId);

  return (
    <div className="flex flex-col h-screen bg-[#1e1e1e] text-[#d4d4d4] font-mono text-sm select-none overflow-hidden">

      {/* ── Top bar ── */}
      <div className="flex items-center gap-2 px-3 h-9 bg-[#3c3c3c] border-b border-[#555] shrink-0">
        <span className="text-white font-semibold tracking-wide text-xs">SCOPE</span>
        <span className="text-[#555]">·</span>
        {activeFolder && <span className="text-[#858585] text-xs">{activeFolder.label}</span>}
        {activeScript && <><span className="text-[#555]">/</span><span className="text-[#4ec9b0] text-xs">{activeScript.filename}</span></>}
        <div className="flex-1" />
        <button onClick={() => dispatch({ type: 'CLEAR' })}
          className="flex items-center gap-1 px-2 py-0.5 bg-[#2d2d2d] border border-[#555] rounded hover:bg-[#3a3a3a] text-xs">
          <RotateCcw size={11} /> Clear
        </button>
        {result && (
          <button onClick={() => {
            const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' });
            const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
            a.download = `${activeScript?.filename ?? 'scope'}.json`; a.click();
          }} className="flex items-center gap-1 px-2 py-0.5 bg-[#2d2d2d] border border-[#555] rounded hover:bg-[#3a3a3a] text-xs">
            <Download size={11} /> Export
          </button>
        )}
        <button onClick={run} disabled={running}
          className="flex items-center gap-1 px-3 py-0.5 bg-[#007acc] hover:bg-[#0098ff] disabled:opacity-50 rounded text-white text-xs font-semibold">
          <Play size={11} fill="white" /> {running ? 'Running…' : 'Run ▶'}
        </button>
      </div>

      {/* ── Three-column body ── */}
      <div className="flex flex-1 overflow-hidden">

        {/* ── Col 1: File explorer ── */}
        <FileExplorer
          folders={SCOPE_FOLDERS}
          activeId={state.activeScriptId}
          onOpen={script => dispatch({ type: 'OPEN_SCRIPT', script })}
        />

        {/* ── Col 2: Editor ── */}
        <div className="flex flex-col border-r border-[#3a3a3a]" style={{ width: '40%', minWidth: 280 }}>
          <div className="px-3 py-1 bg-[#252526] text-[#858585] text-xs border-b border-[#3a3a3a] flex items-center gap-2">
            <FileCode2 size={12} />
            <span>{activeScript?.filename ?? 'untitled.scope'}</span>
            <span className="flex-1" />
            <span className="text-[#555]">Ctrl+Enter to run</span>
          </div>
          <textarea
            ref={textareaRef}
            value={state.source}
            onChange={e => dispatch({ type: 'SET_SOURCE', source: e.target.value })}
            spellCheck={false}
            className="flex-1 bg-[#1e1e1e] text-[#d4d4d4] resize-none outline-none p-4 leading-relaxed text-xs font-mono"
            style={{ tabSize: 2 }}
          />
        </div>

        {/* ── Col 3: Output panel ── */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Tab bar */}
          <div className="flex bg-[#252526] border-b border-[#3a3a3a] shrink-0">
            {(['visualise','charts','3d'] as const).map(tab => (
              <button key={tab} onClick={() => dispatch({ type: 'SET_TAB', tab })}
                className={`px-4 py-2 text-xs capitalize border-b-2 ${
                  activeTab === tab ? 'border-[#007acc] text-white' : 'border-transparent text-[#858585] hover:text-[#d4d4d4]'}`}>
                {tab === '3d' ? '3D' : tab.charAt(0).toUpperCase() + tab.slice(1)}
              </button>
            ))}
          </div>

          {/* Tab content */}
          <div className="flex-1 overflow-auto p-3">
            {activeTab === 'visualise' && (
              <VisualiseTab result={result} mode={state.activeVisMode}
                onMode={m => dispatch({ type: 'SET_VIS', mode: m })} preloadImage={preloadImage} />
            )}
            {activeTab === 'charts' && (
              <ChartsTab result={result} preloadImage={preloadImage} />
            )}
            {activeTab === '3d' && (
              <ThreeDTab result={result} mode={state.active3DMode}
                onMode={m => dispatch({ type: 'SET_3D', mode: m })} preloadImage={preloadImage} />
            )}
          </div>

          {/* Results strip */}
          {result && (
            <div className="shrink-0 border-t border-[#3a3a3a] bg-[#252526] px-3 py-1.5 text-xs space-y-1">
              <div className="flex gap-3 flex-wrap">
                {result.distance !== null && (
                  <span className="text-[#4ec9b0]">d={result.distance.toFixed(3)} ±{result.uncertainty?.toFixed(3)} µm
                    {result.relativeUncertainty !== null && ` (${(result.relativeUncertainty*100).toFixed(2)}%)`}
                  </span>
                )}
                <span>Sk={result.sEntropy.sk.toFixed(3)} St={result.sEntropy.st.toFixed(3)} Se={result.sEntropy.se.toFixed(3)} Σ={result.sEntropy.sum.toFixed(10)} {Math.abs(result.sEntropy.sum-1)<1e-10?'✓':'⚠'}</span>
                <span>SNR={result.snr.toFixed(1)} CRLB={result.crlbPixels.toFixed(3)}px C={result.channelCapacity.toFixed(2)}bits</span>
              </div>
              {result.goalStatus.length > 0 && (
                <div className="flex gap-1.5 flex-wrap">
                  {result.goalStatus.map((g, i) => (
                    <span key={i} className={`px-1.5 py-0.5 rounded text-xs ${g.passed?'bg-[#1a3a1a] text-[#4caf50]':'bg-[#3a1a1a] text-[#f44336]'}`}>
                      {g.passed?'✓':'✗'} {g.metric} {g.op} {g.threshold}{g.unit}
                    </span>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* ── Console ── */}
      <div className="shrink-0 h-32 border-t border-[#3a3a3a] bg-[#1a1a1a] overflow-y-auto">
        <div className="px-3 py-0.5 bg-[#252526] text-[#858585] text-xs border-b border-[#3a3a3a]">CONSOLE</div>
        <div className="px-3 py-1 text-xs leading-5 font-mono">
          {log.length === 0 && !error && <span className="text-[#555]">Run a script with Ctrl+Enter or the Run ▶ button.</span>}
          {error && <span className="text-[#f44336]">{error}</span>}
          {log.map((line, i) => <div key={i} className={logLineColor(line)}>{line}</div>)}
        </div>
      </div>
    </div>
  );
}

// ── File Explorer ─────────────────────────────────────────────────────────────

function FileExplorer({ folders, activeId, onOpen }: {
  folders: ScopeFolder[];
  activeId: string;
  onOpen: (s: ScopeScript) => void;
}) {
  const [open, setOpen] = useState<Set<string>>(() => new Set([folders[0].id]));

  const toggle = (id: string) => {
    setOpen(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  };

  return (
    <div className="flex flex-col bg-[#252526] border-r border-[#3a3a3a] overflow-y-auto shrink-0" style={{ width: 220 }}>
      <div className="px-3 py-1.5 text-[#858585] text-xs uppercase tracking-widest border-b border-[#3a3a3a]">
        Explorer
      </div>
      <div className="flex-1 py-1">
        {folders.map(folder => {
          const isOpen = open.has(folder.id);
          return (
            <div key={folder.id}>
              {/* Folder row */}
              <button
                onClick={() => toggle(folder.id)}
                className="w-full flex items-center gap-1.5 px-2 py-1 text-xs hover:bg-[#2a2d2e] text-[#cccccc]"
              >
                {isOpen ? <ChevronDown size={12} className="text-[#858585] shrink-0" /> : <ChevronRight size={12} className="text-[#858585] shrink-0" />}
                {isOpen ? <FolderOpen size={13} className="text-[#e8ab55] shrink-0" /> : <Folder size={13} className="text-[#e8ab55] shrink-0" />}
                <span className="truncate">{folder.label}</span>
              </button>

              {/* Script rows */}
              {isOpen && folder.scripts.map(script => {
                const active = script.id === activeId;
                return (
                  <button
                    key={script.id}
                    onClick={() => onOpen(script)}
                    title={script.description}
                    className={`w-full flex items-center gap-1.5 pl-8 pr-2 py-1 text-xs truncate
                      ${active ? 'bg-[#094771] text-white' : 'text-[#858585] hover:bg-[#2a2d2e] hover:text-[#cccccc]'}`}
                  >
                    <FileCode2 size={12} className={active ? 'text-[#4ec9b0] shrink-0' : 'text-[#858585] shrink-0'} />
                    <span className="truncate">{script.filename}</span>
                  </button>
                );
              })}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Output tabs ───────────────────────────────────────────────────────────────

const VIS_MODES = ['raw_image','scale_field','segmentation','distance_map','geodesic','overlay'] as const;
const SCENE_MODES = ['scale_field','point_cloud','entropy_sphere','distance_tube','partition_tree'] as const;

function VisualiseTab({ result, mode, onMode, preloadImage }: {
  result: ScopeResult | null; mode: string; onMode: (m: string) => void;
  preloadImage: ImagePayload | null;
}) {
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-1">
        {VIS_MODES.map(m => (
          <button key={m} onClick={() => onMode(m)}
            className={`px-2 py-0.5 text-xs rounded ${mode===m?'bg-[#007acc] text-white':'bg-[#2d2d2d] hover:bg-[#37373d]'}`}>
            {m.replace(/_/g,' ')}
          </button>
        ))}
      </div>
      <Canvas2D result={result} mode={mode} preloadImage={preloadImage} />
    </div>
  );
}

function ChartsTab({ result, preloadImage }: { result: ScopeResult | null; preloadImage: ImagePayload | null; }) {
  const hasImage = !!(result?.visualData?.rawImage ?? preloadImage);
  return (
    <div className="space-y-3">
      {hasImage && (
        <div className="flex gap-3 items-start">
          <div className="shrink-0" style={{ width: 160 }}>
            <div className="text-[#858585] text-xs mb-1">Cell image</div>
            <Canvas2D result={result} mode={result?.visualData?.activeVisMode ?? 'raw_image'} preloadImage={preloadImage} />
          </div>
          {result && (
            <div className="shrink-0" style={{ width: 160 }}>
              <div className="text-[#858585] text-xs mb-1">Segmentation</div>
              <Canvas2D result={result} mode="segmentation" preloadImage={preloadImage} />
            </div>
          )}
        </div>
      )}
      {!result && <div className="text-[#555] text-xs py-2">Run a script to see charts.</div>}
      {result && (
        <>
          <SpectralPowerChart data={result.chartData.spectralPower} exponent={result.chartData.powerLawExponent} />
          <EntropyTrajectoryChart data={result.chartData.entropyTrajectory} />
          <UncertaintyBar data={result.chartData.uncertaintyBar} />
          <ScaleHistogram data={result.chartData.scaleHistogram} mean={result.chartData.alphaMean} />
        </>
      )}
    </div>
  );
}

function ThreeDTab({ result, mode, onMode, preloadImage }: {
  result: ScopeResult | null; mode: string; onMode: (m: string) => void;
  preloadImage: ImagePayload | null;
}) {
  const imgData   = result?.visualData?.rawImage ?? preloadImage?.data ?? null;
  const imgWidth  = result?.visualData?.width    ?? preloadImage?.width  ?? 0;
  const imgHeight = result?.visualData?.height   ?? preloadImage?.height ?? 0;
  const scaleField = result?.visualData?.scaleField ?? null;
  const segMask    = result?.visualData?.segmentationMask ?? null;
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-1">
        {SCENE_MODES.map(m => (
          <button key={m} onClick={() => onMode(m)}
            className={`px-2 py-0.5 text-xs rounded ${mode===m?'bg-[#007acc] text-white':'bg-[#2d2d2d] hover:bg-[#37373d]'}`}>
            {m.replace(/_/g,' ')}
          </button>
        ))}
      </div>
      {mode==='point_cloud' && imgData && <PointCloud imageData={imgData} width={imgWidth} height={imgHeight} scaleField={scaleField} segMask={segMask} pixelSizeµm={0.1} zScale={5} />}
      {mode==='point_cloud' && !imgData && <div className="text-[#555] text-xs py-8 text-center">Loading image…</div>}
      {mode==='scale_field' && result && <ScaleFieldSurface result={result} />}
      {mode==='scale_field' && !result && <div className="text-[#555] text-xs py-8 text-center">Run a script to see the scale field.</div>}
      {mode==='entropy_sphere' && result && <EntropySphere sk={result.sEntropy.sk} st={result.sEntropy.st} se={result.sEntropy.se} />}
      {mode==='entropy_sphere' && !result && <div className="text-[#555] text-xs py-8 text-center">Run a script to see the entropy sphere.</div>}
      {mode==='distance_tube' && result && <DistanceTube result={result} />}
      {mode==='distance_tube' && !result && <div className="text-[#555] text-xs py-8 text-center">Run a script to see the distance tube.</div>}
      {mode==='partition_tree' && <div className="text-[#555] text-xs py-4 text-center">Partition tree — coming in Phase 5.</div>}
    </div>
  );
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function logLineColor(line: string): string {
  if (line.startsWith('[COMPILE]'))  return 'text-[#4ec9b0]';
  if (line.startsWith('[MEASURE]'))  return 'text-[#569cd6]';
  if (line.startsWith('[ASSIGN]'))   return 'text-[#9cdcfe]';
  if (line.startsWith('[EXECUTE]'))  return 'text-[#d4d4d4]';
  if (line.startsWith('[EMIT]'))     return 'text-[#ce9178]';
  if (line.startsWith('[GOAL]'))     return line.includes('✓') ? 'text-[#4caf50]' : 'text-[#f44336]';
  if (line.startsWith('[TYPE WARNING]')) return 'text-[#ffa500]';
  if (line.startsWith('[TYPE ERROR]'))   return 'text-[#f44336]';
  if (line.startsWith('[PARSE ERROR]'))  return 'text-[#f44336]';
  if (line.startsWith('[FETCH]'))    return 'text-[#858585]';
  if (line.startsWith('[VISUALISE]'))return 'text-[#c586c0]';
  if (line.startsWith('[ERROR]'))    return 'text-[#f44336]';
  return 'text-[#858585]';
}

function visToTab(mode: string): 'visualise' | 'charts' | '3d' {
  const vis    = ['scale_field','segmentation','distance_map','geodesic','overlay','raw_image'];
  const charts = ['spectral_power','entropy_trajectory','uncertainty_bar','scale_histogram','channel_capacity'];
  const d3     = ['point_cloud','entropy_sphere','distance_tube','partition_tree'];
  if (vis.includes(mode))    return 'visualise';
  if (charts.includes(mode)) return 'charts';
  if (d3.includes(mode))     return '3d';
  return 'visualise';
}
