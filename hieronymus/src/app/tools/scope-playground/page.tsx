'use client';

import React, { useReducer, useRef, useCallback, useEffect, useState } from 'react';
import { Play, RotateCcw, Download, ChevronDown } from 'lucide-react';
import { compile } from '@/lib/scope-compiler';
import { runScope, ScopeResult } from '@/lib/scope-runtime/runtime';
import { SCOPE_EXAMPLES } from '@/lib/scope-examples';
import Canvas2D from './components/visualise/Canvas2D';
import SpectralPowerChart from './components/charts/SpectralPowerChart';
import EntropyTrajectoryChart from './components/charts/EntropyTrajectoryChart';
import UncertaintyBar from './components/charts/UncertaintyBar';
import ScaleHistogram from './components/charts/ScaleHistogram';
import EntropySphere from './components/threed/EntropySphere';
import DistanceTube from './components/threed/DistanceTube';
import ScaleFieldSurface from './components/threed/ScaleFieldSurface';

// ── State ────────────────────────────────────────────────────────────────────

interface State {
  source: string;
  running: boolean;
  result: ScopeResult | null;
  log: string[];
  error: string | null;
  activeTab: 'dataset' | 'visualise' | 'charts' | '3d';
  activeVisMode: string;
  activeChartMode: string;
  active3DMode: string;
}

type Action =
  | { type: 'SET_SOURCE'; source: string }
  | { type: 'RUN_START' }
  | { type: 'RUN_OK'; result: ScopeResult; log: string[] }
  | { type: 'RUN_ERR'; error: string; log: string[] }
  | { type: 'CLEAR' }
  | { type: 'SET_TAB'; tab: State['activeTab'] }
  | { type: 'SET_VIS'; mode: string }
  | { type: 'SET_CHART'; mode: string }
  | { type: 'SET_3D'; mode: string };

function reducer(state: State, action: Action): State {
  switch (action.type) {
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

const INITIAL: State = {
  source: SCOPE_EXAMPLES[0].source,
  running: false,
  result: null,
  log: [],
  error: null,
  activeTab: 'dataset',
  activeVisMode: 'geodesic',
  activeChartMode: 'spectral_power',
  active3DMode: 'scale_field',
};

// ── Main page ────────────────────────────────────────────────────────────────

export default function ScopePlayground() {
  const [state, dispatch] = useReducer(reducer, INITIAL);
  const [examplesOpen, setExamplesOpen] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const run = useCallback(async () => {
    dispatch({ type: 'RUN_START' });
    const log: string[] = [];

    try {
      // 1. Compile
      const cr = compile(state.source);
      log.push(...cr.log);
      if (!cr.ok || !cr.program) {
        dispatch({ type: 'RUN_ERR', error: cr.errors[0]?.kind ?? 'CompileError', log });
        return;
      }

      // 2. Fetch image
      const program = cr.program;
      const firstLoad = program.morphisms.find(
        m => m.expr.observe.frame.kind === 'LoadRef'
      )?.expr.observe.frame;
      let imagePayload: { data: Float32Array; width: number; height: number };

      if (firstLoad?.kind === 'LoadRef') {
        const params = new URLSearchParams({
          db: firstLoad.db,
          dataset: firstLoad.dataset,
          image: firstLoad.image,
        });
        log.push(`[FETCH]    loading ${firstLoad.dataset}/${firstLoad.image}`);
        const res = await fetch(`/api/image-proxy?${params}`);
        const json = await res.json();
        if (json.error) throw new Error(json.error);
        const raw = json.data as number[];
        imagePayload = {
          data: new Float32Array(raw),
          width: json.width as number,
          height: json.height as number,
        };
        if (json.synthetic) log.push(`[FETCH]    ⚠ using synthetic fallback image`);
        else log.push(`[FETCH]    ${json.width}×${json.height} pixels loaded`);
      } else {
        // No load() — generate synthetic
        const W = 256, H = 256;
        const data = new Float32Array(W * H);
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
          const d1 = (x-80)**2 + (y-128)**2, d2 = (x-176)**2 + (y-128)**2;
          data[y*W+x] = Math.max(0, Math.min(1,
            0.05 + 0.9*Math.exp(-d1/1250) + 0.9*Math.exp(-d2/968)));
        }
        imagePayload = { data, width: W, height: H };
        log.push(`[FETCH]    synthetic 256×256 image`);
      }

      // 3. Run
      log.push(`[RUNTIME]  running five-phase pipeline...`);
      const result = await runScope(program, imagePayload);
      log.push(...result.log.filter(l => !log.includes(l)));

      // Auto-switch to the visualise mode requested by the last visualise() step
      const vis = result.visualData.activeVisMode;
      if (vis) {
        const tabFor = visToTab(vis);
        dispatch({ type: 'RUN_OK', result, log });
        if (tabFor === 'visualise') dispatch({ type: 'SET_VIS', mode: vis });
        else if (tabFor === 'charts') dispatch({ type: 'SET_CHART', mode: vis });
        else if (tabFor === '3d')    dispatch({ type: 'SET_3D', mode: vis });
      } else {
        dispatch({ type: 'RUN_OK', result, log });
      }
    } catch (err) {
      log.push(`[ERROR]    ${err instanceof Error ? err.message : String(err)}`);
      dispatch({ type: 'RUN_ERR', error: String(err), log });
    }
  }, [state.source]);

  // Keyboard shortcut Ctrl+Enter
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') { e.preventDefault(); run(); }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [run]);

  const { result, log, error, running, activeTab } = state;

  return (
    <div className="flex flex-col h-screen bg-[#1e1e1e] text-[#d4d4d4] font-mono text-sm select-none">

      {/* ── Top bar ── */}
      <div className="flex items-center gap-3 px-4 h-10 bg-[#3c3c3c] border-b border-[#555] shrink-0">
        <span className="text-white font-semibold tracking-wide">SCOPE PLAYGROUND</span>
        <div className="flex-1" />

        {/* Examples dropdown */}
        <div className="relative">
          <button
            onClick={() => setExamplesOpen(o => !o)}
            className="flex items-center gap-1 px-3 py-1 bg-[#2d2d2d] border border-[#555] rounded hover:bg-[#3a3a3a] text-xs"
          >
            Examples <ChevronDown size={12} />
          </button>
          {examplesOpen && (
            <div className="absolute right-0 mt-1 w-72 bg-[#252526] border border-[#555] rounded shadow-xl z-50">
              {SCOPE_EXAMPLES.map(ex => (
                <button
                  key={ex.id}
                  onClick={() => {
                    dispatch({ type: 'SET_SOURCE', source: ex.source });
                    dispatch({ type: 'CLEAR' });
                    setExamplesOpen(false);
                  }}
                  className="w-full text-left px-3 py-2 hover:bg-[#37373d] text-xs border-b border-[#3a3a3a] last:border-0"
                >
                  <div className="text-[#4ec9b0] font-semibold">{ex.title}</div>
                  <div className="text-[#858585] mt-0.5">{ex.description}</div>
                </button>
              ))}
            </div>
          )}
        </div>

        <button
          onClick={() => dispatch({ type: 'CLEAR' })}
          className="flex items-center gap-1 px-3 py-1 bg-[#2d2d2d] border border-[#555] rounded hover:bg-[#3a3a3a] text-xs"
        >
          <RotateCcw size={12} /> Clear
        </button>

        {result && (
          <button
            onClick={() => {
              const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' });
              const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
              a.download = 'scope-result.json'; a.click();
            }}
            className="flex items-center gap-1 px-3 py-1 bg-[#2d2d2d] border border-[#555] rounded hover:bg-[#3a3a3a] text-xs"
          >
            <Download size={12} /> Export JSON
          </button>
        )}

        <button
          onClick={run}
          disabled={running}
          className="flex items-center gap-1 px-4 py-1 bg-[#007acc] hover:bg-[#0098ff] disabled:opacity-50 rounded text-white text-xs font-semibold"
        >
          <Play size={12} fill="white" />
          {running ? 'Running...' : 'Run ▶'}
        </button>
      </div>

      {/* ── Main split ── */}
      <div className="flex flex-1 overflow-hidden">

        {/* ── Left: code editor ── */}
        <div className="w-[45%] flex flex-col border-r border-[#3a3a3a]">
          <div className="px-3 py-1 bg-[#252526] text-[#858585] text-xs border-b border-[#3a3a3a]">
            SCOPE — Ctrl+Enter to run
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

        {/* ── Right: tabbed panel ── */}
        <div className="flex-1 flex flex-col overflow-hidden">

          {/* Tab bar */}
          <div className="flex bg-[#252526] border-b border-[#3a3a3a] shrink-0">
            {(['dataset','visualise','charts','3d'] as const).map(tab => (
              <button
                key={tab}
                onClick={() => dispatch({ type: 'SET_TAB', tab })}
                className={`px-4 py-2 text-xs capitalize border-b-2 ${
                  activeTab === tab
                    ? 'border-[#007acc] text-white'
                    : 'border-transparent text-[#858585] hover:text-[#d4d4d4]'
                }`}
              >
                {tab === '3d' ? '3D' : tab.charAt(0).toUpperCase() + tab.slice(1)}
              </button>
            ))}
            <div className="flex-1" />
          </div>

          {/* Tab content */}
          <div className="flex-1 overflow-auto p-3">

            {activeTab === 'dataset' && (
              <DatasetTab />
            )}

            {activeTab === 'visualise' && (
              <VisualiseTab
                result={result}
                mode={state.activeVisMode}
                onMode={m => dispatch({ type: 'SET_VIS', mode: m })}
              />
            )}

            {activeTab === 'charts' && (
              <ChartsTab
                result={result}
                mode={state.activeChartMode}
                onMode={m => dispatch({ type: 'SET_CHART', mode: m })}
              />
            )}

            {activeTab === '3d' && (
              <ThreeDTab
                result={result}
                mode={state.active3DMode}
                onMode={m => dispatch({ type: 'SET_3D', mode: m })}
              />
            )}
          </div>

          {/* Results panel */}
          {result && (
            <div className="shrink-0 border-t border-[#3a3a3a] bg-[#252526] px-4 py-2 text-xs space-y-1">
              <div className="flex gap-4 flex-wrap">
                {result.distance !== null && (
                  <span className="text-[#4ec9b0]">
                    d = {result.distance.toFixed(3)} ± {result.uncertainty?.toFixed(3)} µm
                    {result.relativeUncertainty !== null && ` (${(result.relativeUncertainty * 100).toFixed(2)}%)`}
                  </span>
                )}
                <span>S_k={result.sEntropy.sk.toFixed(3)} S_t={result.sEntropy.st.toFixed(3)} S_e={result.sEntropy.se.toFixed(3)} Σ={result.sEntropy.sum.toFixed(12)} {Math.abs(result.sEntropy.sum - 1) < 1e-10 ? '✓' : '⚠'}</span>
                <span>SNR={result.snr.toFixed(1)} CRLB={result.crlbPixels.toFixed(3)}px C={result.channelCapacity.toFixed(2)}bits</span>
              </div>
              {result.goalStatus.length > 0 && (
                <div className="flex gap-2 flex-wrap">
                  {result.goalStatus.map((g, i) => (
                    <span key={i} className={`px-2 py-0.5 rounded text-xs ${g.passed ? 'bg-[#1a3a1a] text-[#4caf50]' : 'bg-[#3a1a1a] text-[#f44336]'}`}>
                      {g.passed ? '✓' : '✗'} {g.metric} {g.op} {g.threshold}{g.unit}
                    </span>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* ── Console ── */}
      <div className="shrink-0 h-36 border-t border-[#3a3a3a] bg-[#1a1a1a] overflow-y-auto">
        <div className="px-3 py-1 bg-[#252526] text-[#858585] text-xs border-b border-[#3a3a3a]">CONSOLE</div>
        <div className="px-3 py-1 text-xs leading-5 font-mono">
          {log.length === 0 && !error && (
            <span className="text-[#555]">Run a program with Ctrl+Enter or the Run ▶ button.</span>
          )}
          {error && <span className="text-[#f44336]">{error}</span>}
          {log.map((line, i) => (
            <div key={i} className={logLineColor(line)}>{line}</div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── Sub-tabs ─────────────────────────────────────────────────────────────────

function DatasetTab() {
  const datasets = [
    { id: 'BBBC039', label: 'BBBC039 — HeLa cells (Hoechst + Actin)', res: '0.1 µm/px', size: '1024×1024', ch: 'DAPI, Actin' },
    { id: 'BBBC006', label: 'BBBC006 — CHO cells (Tubulin)',          res: '0.063 µm/px', size: '696×520',   ch: 'Tubulin' },
    { id: 'BBBC008', label: 'BBBC008 — Drosophila (GFP + DAPI)',      res: '0.08 µm/px',  size: '1024×1024', ch: 'GFP, DAPI' },
    { id: 'AllenCell', label: 'Allen Cell — 3D structures',           res: '0.065 µm/px', size: '1×z-stack', ch: 'GFP, DAPI' },
    { id: 'IDR',      label: 'IDR — Time-lapse',                      res: '0.1 µm/px',   size: 'varies',    ch: 'GFP, DAPI, RFP' },
    { id: 'OpenCell', label: 'OpenCell — Protein localisation',       res: '0.08 µm/px',  size: '1024×1024', ch: 'GFP, DAPI' },
  ];
  const [sel, setSel] = useState('BBBC039');
  const info = datasets.find(d => d.id === sel)!;

  return (
    <div className="space-y-3 text-xs">
      <div className="space-y-1">
        {datasets.map(d => (
          <button key={d.id} onClick={() => setSel(d.id)}
            className={`w-full text-left px-3 py-2 rounded ${sel === d.id ? 'bg-[#37373d] border border-[#007acc]' : 'hover:bg-[#2a2a2a]'}`}>
            {d.label}
          </button>
        ))}
      </div>
      <div className="border border-[#3a3a3a] rounded p-3 space-y-1 text-[#858585]">
        <div><span className="text-[#d4d4d4]">Selected: </span>{info.id}</div>
        <div><span className="text-[#d4d4d4]">Resolution: </span>{info.res}</div>
        <div><span className="text-[#d4d4d4]">Size: </span>{info.size}</div>
        <div><span className="text-[#d4d4d4]">Channels: </span>{info.ch}</div>
      </div>
      <p className="text-[#555] leading-relaxed">
        Images are fetched at runtime via <code className="text-[#4ec9b0]">/api/image-proxy</code>.
        If the database is unreachable, a synthetic 256×256 two-nucleus image is used as fallback.
      </p>
    </div>
  );
}

const VIS_MODES = ['raw_image','scale_field','segmentation','distance_map','geodesic','overlay'] as const;
const CHART_MODES = ['spectral_power','entropy_trajectory','uncertainty_bar','scale_histogram','channel_capacity'] as const;
const SCENE_MODES = ['scale_field','point_cloud','entropy_sphere','distance_tube','partition_tree'] as const;

function VisualiseTab({ result, mode, onMode }: {
  result: ScopeResult | null; mode: string; onMode: (m: string) => void;
}) {
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-1">
        {VIS_MODES.map(m => (
          <button key={m} onClick={() => onMode(m)}
            className={`px-2 py-0.5 text-xs rounded ${mode === m ? 'bg-[#007acc] text-white' : 'bg-[#2d2d2d] hover:bg-[#37373d]'}`}>
            {m.replace('_', ' ')}
          </button>
        ))}
      </div>
      <Canvas2D result={result} mode={mode} />
    </div>
  );
}

function ChartsTab({ result, mode, onMode }: {
  result: ScopeResult | null; mode: string; onMode: (m: string) => void;
}) {
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-1">
        {CHART_MODES.map(m => (
          <button key={m} onClick={() => onMode(m)}
            className={`px-2 py-0.5 text-xs rounded ${mode === m ? 'bg-[#007acc] text-white' : 'bg-[#2d2d2d] hover:bg-[#37373d]'}`}>
            {m.replace(/_/g, ' ')}
          </button>
        ))}
      </div>
      {!result && <div className="text-[#555] text-xs py-8 text-center">Run a program to see charts.</div>}
      {result && mode === 'spectral_power'     && <SpectralPowerChart data={result.chartData.spectralPower} exponent={result.chartData.powerLawExponent} />}
      {result && mode === 'entropy_trajectory' && <EntropyTrajectoryChart data={result.chartData.entropyTrajectory} />}
      {result && mode === 'uncertainty_bar'    && <UncertaintyBar data={result.chartData.uncertaintyBar} />}
      {result && mode === 'scale_histogram'    && <ScaleHistogram data={result.chartData.scaleHistogram} mean={result.chartData.powerLawExponent} />}
      {result && mode === 'channel_capacity'   && <SpectralPowerChart data={result.chartData.spectralPower} exponent={result.chartData.powerLawExponent} />}
    </div>
  );
}

function ThreeDTab({ result, mode, onMode }: {
  result: ScopeResult | null; mode: string; onMode: (m: string) => void;
}) {
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-1">
        {SCENE_MODES.map(m => (
          <button key={m} onClick={() => onMode(m)}
            className={`px-2 py-0.5 text-xs rounded ${mode === m ? 'bg-[#007acc] text-white' : 'bg-[#2d2d2d] hover:bg-[#37373d]'}`}>
            {m.replace('_', ' ')}
          </button>
        ))}
      </div>
      {!result && <div className="text-[#555] text-xs py-8 text-center">Run a program to see 3D visualisation.</div>}
      {result && mode === 'scale_field'    && <ScaleFieldSurface result={result} />}
      {result && mode === 'entropy_sphere' && <EntropySphere sk={result.sEntropy.sk} st={result.sEntropy.st} se={result.sEntropy.se} />}
      {result && mode === 'distance_tube'  && <DistanceTube result={result} />}
      {result && (mode === 'point_cloud' || mode === 'partition_tree') && (
        <div className="text-[#555] text-xs py-4 text-center">
          {mode === 'point_cloud' ? 'Point cloud — switch to Scale Field or Distance Tube.' : 'Partition tree — coming in a later example.'}
        </div>
      )}
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
  const vis  = ['scale_field','segmentation','distance_map','geodesic','overlay','raw_image'];
  const charts = ['spectral_power','entropy_trajectory','uncertainty_bar','scale_histogram','channel_capacity'];
  const d3   = ['point_cloud','entropy_sphere','distance_tube','partition_tree'];
  if (vis.includes(mode))    return 'visualise';
  if (charts.includes(mode)) return 'charts';
  if (d3.includes(mode))     return '3d';
  return 'visualise';
}
