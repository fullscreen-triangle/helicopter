'use client';

import React, { useState, useRef, useCallback, useMemo, useEffect } from 'react';
import {
  Files, Search, GitBranch, Play, Blocks, Settings, ChevronRight, ChevronDown,
  X, Circle, FileCode2, FileJson, FileText, Folder, FolderOpen,
  Terminal as TerminalIcon, AlertCircle, Bell, PanelBottomClose, Check,
  Eye, Code2, Trash2, RefreshCw,
} from 'lucide-react';
import { compileScope } from '@/lib/scope-compiler';
import { executeReal } from '@/lib/scope-runtime/real-executor';
import { getScopeExample } from '@/lib/scope-examples';
import AnalysisEditor from '@/components/analysis/AnalysisEditor';
import { EntropyBarChart, MeasurementChart } from './ScopeIDECharts';

const theme = {
  titlebar: '#3c3c3c',
  activitybar: '#333333',
  activitybarFg: '#858585',
  activitybarFgActive: '#ffffff',
  sidebar: '#252526',
  sidebarFg: '#cccccc',
  sidebarHeader: '#bbbbbb',
  editor: '#1e1e1e',
  editorFg: '#d4d4d4',
  tabBar: '#252526',
  tabActive: '#1e1e1e',
  tabInactive: '#2d2d2d',
  tabFg: '#969696',
  tabFgActive: '#ffffff',
  border: '#3c3c3c',
  accent: '#0e639c',
  accentBright: '#007acc',
  statusBar: '#007acc',
  statusFg: '#ffffff',
  panel: '#1e1e1e',
  gutter: '#858585',
  lineActive: '#2a2d2e',
  selection: '#264f78',
};

// In-memory file system for SCOPE scripts
const initialFiles = {
  examples: {
    type: 'folder',
    children: {
      'tutorial_01_hello.scope': {
        type: 'file',
        lang: 'scope',
        content: getScopeExample('ex1-nuclear-separation')?.source ?? '',
      },
      'tutorial_02_measurement.scope': {
        type: 'file',
        lang: 'scope',
        content: getScopeExample('ex3-confidence-threshold')?.source ?? '',
      },
      'tutorial_03_anaphase.scope': {
        type: 'file',
        lang: 'scope',
        content: getScopeExample('ex5-anaphase')?.source ?? '',
      },
      'tutorial_03_nuclear_separation.scope': {
        type: 'file',
        lang: 'scope',
        content: `scope nuclear_separation_measurement {
  channels {
    sync dapi at 0.1 µm/pixel
    cell METAPHASE bounds (-0.8e-6, 0.8e-6) action measure_separation
  }

  coordinate_space {
    field 100 x 100 µm
    depth 10
    lambda_s 0.10
    lambda_t 0.05
  }

  goal {
    distance_uncertainty < 0.5 µm
  }

  rule conservation(dna_mass) {
    invariant: "DAPI-stained area conserved"
    epsilon: 0.008
  }

  measure_separation = observe(load(db="BBBC", dataset="BBBC007", image="A9 p10d.tif"), n = 10)
    |> visualise(scale_field)
    |> catalyze(conservation(dna_mass))
    |> access(nucleus_a)
    |> access(nucleus_b)
    |> visualise(segmentation)
    |> measure_distance(nucleus_a, nucleus_b)
    |> visualise(geodesic)

  dispatch {
    when METAPHASE do execute(measure_separation)
  }
}`,
      },
      'tutorial_04_load_bbbc007.scope': {
        type: 'file',
        lang: 'scope',
        content: `scope load_bbbc007_hela {
  coordinate_space {
    field 100 x 100 µm
    depth 10
    lambda_s 0.10
    lambda_t 0.05
  }

  goal {
    distance_uncertainty < 0.5 µm
    snr > 6.0
  }

  rule conservation(dna_mass) {
    invariant: "DAPI-stained area conserved across field"
    epsilon: 0.008
  }

  analyze_cells = observe(load(db="BBBC", dataset="BBBC007", image="A9 p9d.tif"), n = 10)
    |> visualise(scale_field)
    |> catalyze(conservation(dna_mass))
    |> catalyze(phase_lock(chromatin), confidence = 0.85)
    |> access(nucleus_a)
    |> access(nucleus_b)
    |> visualise(segmentation)
    |> measure_distance(nucleus_a, nucleus_b)
    |> visualise(spectral_power)
    |> visualise(entropy_trajectory)
}`,
      },
      'tutorial_05_f96_spindle.scope': {
        type: 'file',
        lang: 'scope',
        content: `scope f96_spindle_measurement {
  coordinate_space {
    field 64 x 64 µm
    depth 10
    lambda_s 0.08
    lambda_t 0.03
  }

  goal {
    distance_uncertainty < 0.2 µm
  }

  rule symmetry(bilateral) {
    invariant: "mitotic spindle has bilateral symmetry"
    epsilon: 0.006
  }

  rule conservation(dna_mass) {
    invariant: "DAPI-stained area conserved"
    epsilon: 0.008
  }

  spindle = observe(load(db="BBBC", dataset="BBBC007", image="17P1_POS0006_D_1UL.tif"), n = 10)
    |> visualise(scale_field)
    |> catalyze(symmetry(bilateral), confidence = 0.8)
    |> catalyze(conservation(dna_mass))
    |> access(nucleus_a, threshold = 0.7)
    |> access(nucleus_b, threshold = 0.7)
    |> visualise(segmentation)
    |> measure_distance(nucleus_a, nucleus_b)
    |> visualise(distance_map)
    |> visualise(scale_histogram)
}`,
      },
      'tutorial_06_dual_channel.scope': {
        type: 'file',
        lang: 'scope',
        content: `scope f113_dual_channel {
  channels {
    sync gfp  at 0.08 µm/pixel
    sync dapi at 0.08 µm/pixel
  }

  coordinate_space {
    field 82 x 82 µm
    depth 10
    lambda_s 0.09
    lambda_t 0.04
  }

  goal {
    distance_uncertainty < 0.15 µm
    crlb_pixels < 0.1
  }

  rule conservation(dna_mass) {
    invariant: "DAPI-stained area conserved across channels"
    epsilon: 0.008
  }

  dapi_channel = observe(load(db="BBBC", dataset="BBBC007", image="AS_09125_040701150004_A02f00d0.tif"), n = 10)
    |> visualise(scale_field)
    |> catalyze(conservation(dna_mass))
    |> access(nucleus_a)
    |> access(nucleus_b)
    |> visualise(segmentation)
    |> measure_distance(nucleus_a, nucleus_b)
    |> visualise(uncertainty_bar)
    |> visualise(entropy_trajectory)
}`,
      },
    },
  },
  datasets: {
    type: 'folder',
    children: {
      'BBBC007_v1_images': { type: 'folder', children: {} },
      'AICS-24-part06': { type: 'folder', children: {} },
      'human_HT29_colon_cancer': { type: 'folder', children: {} },
    },
  },
  'my-analysis': {
    type: 'folder',
    children: {
      'analysis.scope': {
        type: 'file',
        lang: 'scope',
        content: `scope my_analysis {
    channels {
        sync acquisition at 10000000
        cell METAPHASE bounds (-0.8, 0.8) action measure
    }

    coordinate_space {
        field 100.0 x 100.0 um
        depth 1000
        lambda_s 0.10
        lambda_t 0.05
    }

    morphisms {
        measure =
            observe(frame, n=1000)
            |> measure_distance(nucleus_a, nucleus_b)
    }

    dispatch {
        when METAPHASE do execute(measure)
    }
}`,
      },
    },
  },
};

const fileIcon = (name: string) => {
  if (name.endsWith('.scope')) return { Icon: FileCode2, color: '#00d4ff' };
  if (name.endsWith('.json')) return { Icon: FileJson, color: '#cbcb41' };
  if (name.endsWith('.md')) return { Icon: FileText, color: '#519aba' };
  return { Icon: FileText, color: '#858585' };
};

const getNode = (tree: any, path: string[]) => {
  let n: any = { children: tree };
  for (const p of path) {
    n = n.children[p];
    if (!n) return null;
  }
  return n;
};

function Tree({ tree, path = [], depth = 0, expanded, toggle, activePath, openFile }: any) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const entries = (Object.entries(tree) as [string, any][]).sort((a, b) =>
    a[1].type !== b[1].type ? (a[1].type === 'folder' ? -1 : 1) : a[0].localeCompare(b[0])
  );

  return (
    <>
      {entries.map(([name, node]) => {
        const fullPath = [...path, name];
        const key = fullPath.join('/');
        const isFolder = node.type === 'folder';
        const isOpen = expanded.has(key);
        const isActive = activePath === key;
        const { Icon, color } = isFolder
          ? { Icon: isOpen ? FolderOpen : Folder, color: '#90a4ae' }
          : fileIcon(name);

        return (
          <div key={key}>
            <button
              onClick={() => (isFolder ? toggle(key) : openFile(fullPath))}
              className="flex w-full items-center gap-1 py-0.5 pr-2 text-left text-[13px] leading-relaxed transition-colors"
              style={{
                paddingLeft: 8 + depth * 12,
                color: theme.sidebarFg,
                background: isActive ? theme.lineActive : 'transparent',
              }}
              onMouseEnter={(e) => {
                if (!isActive) e.currentTarget.style.background = '#2a2d2e';
              }}
              onMouseLeave={(e) => {
                if (!isActive) e.currentTarget.style.background = 'transparent';
              }}
            >
              {isFolder ? (
                isOpen ? (
                  <ChevronDown size={14} className="shrink-0 opacity-70" />
                ) : (
                  <ChevronRight size={14} className="shrink-0 opacity-70" />
                )
              ) : (
                <span className="w-[14px] shrink-0" />
              )}
              <Icon size={15} className="shrink-0" style={{ color }} />
              <span className="truncate">{name}</span>
            </button>
            {isFolder && isOpen && (
              <Tree
                tree={node.children}
                path={fullPath}
                depth={depth + 1}
                expanded={expanded}
                toggle={toggle}
                activePath={activePath}
                openFile={openFile}
              />
            )}
          </div>
        );
      })}
    </>
  );
}

function Editor({ value, onChange, onCursor }) {
  const gutterRef = useRef(null);
  const lines = value.split('\n');

  const syncScroll = (e) => {
    if (gutterRef.current) gutterRef.current.scrollTop = e.target.scrollTop;
  };

  const handleCursor = (e) => {
    const upto = e.target.value.slice(0, e.target.selectionStart);
    onCursor({ ln: upto.split('\n').length, col: upto.length - upto.lastIndexOf('\n') });
  };

  return (
    <div className="flex min-h-0 flex-1" style={{ background: theme.editor }}>
      <div
        ref={gutterRef}
        className="select-none overflow-hidden py-3 text-right font-mono text-[13px] leading-[1.5]"
        style={{ color: theme.gutter, minWidth: 52, paddingRight: 16 }}
      >
        {lines.map((_, i) => (
          <div key={i}>{i + 1}</div>
        ))}
      </div>
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onScroll={syncScroll}
        onKeyUp={handleCursor}
        onClick={handleCursor}
        spellCheck={false}
        className="min-h-0 flex-1 resize-none border-0 bg-transparent py-3 pr-4 font-mono text-[13px] leading-[1.5] outline-none"
        style={{ color: theme.editorFg, tabSize: 2, caretColor: '#fff' }}
      />
    </div>
  );
}

function OutputColumn({ compiled, logs, onRun, onClear, outputTab, setOutputTab, canvasRef, chartData }: any) {
  const tab = outputTab;
  const setTab = setOutputTab;
  const tabs = [
    { id: 'console', label: 'Execution Log', Icon: TerminalIcon },
    { id: 'image', label: 'Visualization', Icon: Eye },
    { id: 'entropy', label: 'S-Entropy', Icon: Blocks },
    { id: 'measurements', label: 'Measurements', Icon: Code2 },
    { id: 'quality', label: 'Quality Metrics', Icon: Blocks },
    { id: 'spectral', label: 'Spectral Data', Icon: Blocks },
    { id: 'compiled', label: 'Compiled IR', Icon: Code2 },
  ];

  const levelColor = {
    log: '#d4d4d4',
    info: '#9cdcfe',
    warn: '#dcdcaa',
    error: '#f48771',
  };

  return (
    <div
      className="flex min-w-0 flex-1 flex-col"
      style={{ background: theme.editor, borderLeft: `1px solid ${theme.border}` }}
    >
      <div className="flex h-9 shrink-0 items-center justify-between pr-2" style={{ background: theme.tabInactive }}>
        <div className="flex h-full">
          {tabs.map(({ id, label, Icon }) => {
            const active = tab === id;
            return (
              <button
                key={id}
                onClick={() => setTab(id)}
                className="relative flex items-center gap-1.5 px-3 text-[12px] transition-colors"
                style={{
                  color: active ? theme.tabFgActive : theme.tabFg,
                  background: active ? theme.tabActive : 'transparent',
                }}
              >
                <Icon size={13} /> {label}
                {id === 'console' && logs.length > 0 && (
                  <span className="rounded-full px-1.5 text-[10px]" style={{ background: theme.accent, color: '#fff' }}>
                    {logs.length}
                  </span>
                )}
                {active && <span className="absolute left-0 top-0 h-0.5 w-full" style={{ background: theme.accentBright }} />}
              </button>
            );
          })}
        </div>
        <div className="flex items-center gap-1">
          {tab === 'console' && (
            <button
              onClick={onClear}
              title="Clear console"
              className="flex h-6 w-6 items-center justify-center rounded"
              style={{ color: theme.tabFg }}
            >
              <Trash2 size={14} />
            </button>
          )}
          <button
            onClick={onRun}
            title="Re-run"
            className="flex h-6 items-center gap-1 rounded px-2 text-[12px]"
            style={{ background: theme.accent, color: '#fff' }}
          >
            <RefreshCw size={12} /> Run
          </button>
        </div>
      </div>

      <div className="min-h-0 flex-1">
        {tab === 'console' && (
          <div className="h-full overflow-y-auto p-2 font-mono text-[12px] leading-relaxed">
            {logs.length === 0 ? (
              <div className="px-1 pt-1" style={{ color: '#5a5a5a' }}>
                Execution log appears here.
              </div>
            ) : (
              logs.map((l, i) => (
                <div key={i} className="border-b px-1 py-1" style={{ color: levelColor[l.level] || '#d4d4d4', borderColor: '#2a2a2a' }}>
                  <span className="mr-2 opacity-50">[{l.level}]</span>
                  {l.message}
                </div>
              ))
            )}
          </div>
        )}
        {tab === 'compiled' && (
          <pre className="h-full overflow-auto p-3 font-mono text-[12px] leading-[1.5]" style={{ color: theme.editorFg }}>
            {compiled || '(compiled output appears here)'}
          </pre>
        )}
        {tab === 'image' && (
          <div className="h-full overflow-auto p-3 flex items-center justify-center" style={{ background: theme.editor }}>
            <canvas
              ref={canvasRef}
              className="border border-gray-600 rounded"
              style={{ maxWidth: '100%', maxHeight: '100%' }}
            />
          </div>
        )}
        {tab === 'entropy' && chartData?.entropyChart && (
          <div className="h-full overflow-auto p-4" style={{ background: theme.editor }}>
            <div className="mb-6">
              <h3 className="text-lg font-bold mb-4" style={{ color: theme.editorFg }}>S-Entropy Evolution</h3>
              <EntropyBarChart phases={chartData.entropyChart.phases} />
            </div>
          </div>
        )}
        {tab === 'quality' && chartData?.qualityMetrics && (
          <div className="h-full overflow-auto p-4" style={{ background: theme.editor }}>
            <div className="mb-6">
              <h3 className="text-lg font-bold mb-4" style={{ color: theme.editorFg }}>Quality Metrics</h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 rounded" style={{ background: theme.tabInactive }}>
                  <div className="text-sm opacity-75" style={{ color: theme.editorFg }}>Sharpness</div>
                  <div className="text-2xl font-bold text-blue-400">{(chartData.qualityMetrics.sharpness * 100).toFixed(1)}%</div>
                </div>
                <div className="p-4 rounded" style={{ background: theme.tabInactive }}>
                  <div className="text-sm opacity-75" style={{ color: theme.editorFg }}>Noise</div>
                  <div className="text-2xl font-bold text-yellow-400">{(chartData.qualityMetrics.noise * 100).toFixed(1)}%</div>
                </div>
                <div className="p-4 rounded" style={{ background: theme.tabInactive }}>
                  <div className="text-sm opacity-75" style={{ color: theme.editorFg }}>Coherence</div>
                  <div className="text-2xl font-bold text-green-400">{(chartData.qualityMetrics.coherence * 100).toFixed(1)}%</div>
                </div>
                <div className="p-4 rounded" style={{ background: theme.tabInactive }}>
                  <div className="text-sm opacity-75" style={{ color: theme.editorFg }}>Visibility</div>
                  <div className="text-2xl font-bold text-purple-400">{(chartData.qualityMetrics.visibility * 100).toFixed(1)}%</div>
                </div>
              </div>
            </div>
          </div>
        )}
        {tab === 'spectral' && chartData?.spectralData && (
          <div className="h-full overflow-auto p-4" style={{ background: theme.editor }}>
            <div className="mb-6">
              <h3 className="text-lg font-bold mb-4" style={{ color: theme.editorFg }}>Spectral Decomposition</h3>
              <svg width="100%" height="300" style={{ background: theme.tabInactive, borderRadius: '4px' }}>
                {chartData.spectralData.wavelengths.map((wl: number, idx: number) => {
                  const intensity = chartData.spectralData.intensities[idx];
                  const x = 50 + (idx * 120);
                  const barHeight = intensity * 200;
                  const colors = ['#4a90e2', '#50c878', '#ffd700', '#ff6b6b'];
                  return (
                    <g key={idx}>
                      <rect x={x} y={250 - barHeight} width="80" height={barHeight} fill={colors[idx]} opacity="0.8" rx="4" />
                      <text x={x + 40} y="280" fontSize="12" fill={theme.editorFg} textAnchor="middle">{wl}nm</text>
                      <text x={x + 40} y={230 - barHeight} fontSize="11" fill={colors[idx]} textAnchor="middle" fontWeight="bold">{(intensity * 100).toFixed(0)}%</text>
                    </g>
                  );
                })}
              </svg>
            </div>
          </div>
        )}
        {tab === 'measurements' && chartData?.measurementChart && (
          <div className="h-full overflow-auto p-4" style={{ background: theme.editor }}>
            <h3 className="text-lg font-bold mb-4" style={{ color: theme.editorFg }}>{chartData.measurementChart.title}</h3>
            {chartData.measurementChart.measurements.length > 0 ? (
              <MeasurementChart measurements={chartData.measurementChart.measurements} />
            ) : (
              <div style={{ color: theme.editorFg, opacity: 0.6 }}>No measurements in this program</div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default function ScopeIDEMain() {
  const [files, setFiles] = useState(initialFiles);
  const [expanded, setExpanded] = useState(new Set(['examples', 'datasets', 'my-analysis']));
  const [openTabs, setOpenTabs] = useState([['examples', 'tutorial_01_hello.scope']]);
  const [activeTab, setActiveTab] = useState('examples/tutorial_01_hello.scope');
  const [dirty, setDirty] = useState(new Set());
  const [sidebar, setSidebar] = useState(true);
  const [cursor, setCursor] = useState({ ln: 1, col: 1 });
  const [activity, setActivity] = useState('files');

  const [compiled, setCompiled] = useState('');
  const [logs, setLogs] = useState([]);
  const [outputTab, setOutputTab] = useState<'console' | 'image' | 'entropy' | 'measurements' | 'quality' | 'spectral' | 'compiled'>('console');
  const [visualizationData, setVisualizationData] = useState<any>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const splitRef = useRef(null);
  const dragging = useRef(false);
  const [editorWidth, setEditorWidth] = useState(55);

  const run = useCallback(() => {
    const activePathArr = openTabs.find((t) => t.join('/') === activeTab);
    if (!activePathArr) return;

    const activeNode = getNode(files, activePathArr);
    if (!activeNode) return;

    const newLogs = [];
    const log = (msg) => newLogs.push({ level: 'log', message: msg });

    try {
      log('Compiling SCOPE program...');
      const compiledProgram = compileScope(activeNode.content);

      if (!compiledProgram.ok) {
        log('❌ Compilation failed:');
        compiledProgram.errors.forEach((err) => {
          log(`  ${err.kind}: ${(err as any).message || JSON.stringify(err)}`);
        });
        setLogs(newLogs);
        return;
      }

      log('✓ Compilation successful');
      log(`Program: ${compiledProgram.program?.name ?? '(unknown)'}`);
      log(`Morphisms: ${compiledProgram.program?.morphisms?.length || 0}`);

      if (compiledProgram.warnings.length > 0) {
        log('⚠ Warnings:');
        compiledProgram.warnings.forEach((warn) => {
          log(`  ${warn.kind}`);
        });
      }

      setCompiled(JSON.stringify(compiledProgram.program, null, 2));

      log('');
      log('Executing SCOPE program...');

      executeReal(compiledProgram.program as any)
        .then((result) => {
          if (result.success) {
            result.logs.forEach((l) => log(l));

            log('');
            log('═══ RESULT ═══');
            log(`Structure: ${result.structure}`);
            if (result.distance) {
              log(`Distance: ${result.distance.toFixed(3)} µm ± ${result.uncertainty?.toFixed(3)} µm`);
            }
            log(
              `Position: (${result.position.x.toFixed(1)}, ${result.position.y.toFixed(1)}, ${result.position.z.toFixed(1)}) µm`
            );
            log(
              `S-Entropy: S_k=${result.s_entropy.S_k.toFixed(3)} S_t=${result.s_entropy.S_t.toFixed(3)} S_e=${result.s_entropy.S_e.toFixed(3)}`
            );

            // Store visualization data
            if (result.coordinateField) {
              setVisualizationData(result);
              // Auto-switch to image tab if measurements present
              if (result.measurements && result.measurements.length > 0) {
                setOutputTab('image');
              }
            }
          } else {
            log(`❌ Execution failed`);
          }
          setLogs([...newLogs]);
        })
        .catch((err) => {
          log(`❌ Execution error: ${err instanceof Error ? err.message : String(err)}`);
          setLogs([...newLogs]);
        });

      setLogs(newLogs);
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : String(error);
      log(`❌ Error: ${errorMsg}`);
      setLogs(newLogs);
    }
  }, [files, openTabs, activeTab]);

  // Render visualization when data changes
  useEffect(() => {
    if (visualizationData && canvasRef.current && outputTab === 'image') {
      try {
        const { visualizeCoordinateField } = require('@/lib/scope-runtime/visualize-field');
        const hasMeasure = visualizationData.measurements && visualizationData.measurements.length > 0;
        visualizeCoordinateField(visualizationData.coordinateField, canvasRef.current, {
          width: 512,
          height: 512,
          showGrid: true,
          showMeasurements: true,
          programName: visualizationData.programName || 'Unknown',
          hasMeasure,
          measurements: visualizationData.measurements,
        });
      } catch (e) {
        // Fallback: show text
        const ctx = canvasRef.current.getContext('2d');
        if (ctx) {
          ctx.fillStyle = '#d4d4d4';
          ctx.font = '14px monospace';
          ctx.fillText('Visualizing coordinate field...', 10, 30);
        }
      }
    }
  }, [visualizationData, outputTab]);

  useEffect(() => {
    const move = (e) => {
      if (!dragging.current || !splitRef.current) return;
      const r = splitRef.current.getBoundingClientRect();
      const pct = ((e.clientX - r.left) / r.width) * 100;
      setEditorWidth(Math.min(80, Math.max(25, pct)));
    };
    const up = () => {
      dragging.current = false;
      document.body.style.cursor = '';
    };
    window.addEventListener('mousemove', move);
    window.addEventListener('mouseup', up);
    return () => {
      window.removeEventListener('mousemove', move);
      window.removeEventListener('mouseup', up);
    };
  }, []);

  const toggleFolder = useCallback((key) => {
    setExpanded((prev) => {
      const n = new Set(prev);
      n.has(key) ? n.delete(key) : n.add(key);
      return n;
    });
  }, []);

  const openFile = useCallback((pathArr) => {
    const key = pathArr.join('/');
    setOpenTabs((prev) => (prev.some((t) => t.join('/') === key) ? prev : [...prev, pathArr]));
    setActiveTab(key);
  }, []);

  const closeTab = useCallback(
    (key, e) => {
      e.stopPropagation();
      setOpenTabs((prev) => {
        const next = prev.filter((t) => t.join('/') !== key);
        if (activeTab === key) setActiveTab(next.length ? next[next.length - 1].join('/') : null);
        return next;
      });
      setDirty((prev) => {
        const n = new Set(prev);
        n.delete(key);
        return n;
      });
    },
    [activeTab]
  );

  const activePathArr = useMemo(
    () => openTabs.find((t) => t.join('/') === activeTab) || null,
    [openTabs, activeTab]
  );
  const activeNode = activePathArr ? getNode(files, activePathArr) : null;

  const updateContent = useCallback(
    (val) => {
      if (!activePathArr) return;
      setFiles((prev) => {
        const next = structuredClone(prev);
        getNode(next, activePathArr).content = val;
        return next;
      });
      setDirty((prev) => new Set(prev).add(activeTab));
    },
    [activePathArr, activeTab]
  );

  const activities = [
    { id: 'files', Icon: Files, label: 'Explorer' },
    { id: 'search', Icon: Search, label: 'Search' },
    { id: 'git', Icon: GitBranch, label: 'Source Control' },
  ];

  return (
    <div
      className="flex h-screen w-screen flex-col overflow-hidden text-sm"
      style={{ background: theme.editor, color: theme.editorFg, border: `1px solid ${theme.border}` }}
    >
      {/* Title bar */}
      <div className="flex h-9 shrink-0 items-center justify-between px-3" style={{ background: theme.titlebar }}>
        <div className="flex items-center gap-2">
          <span className="h-3 w-3 rounded-full" style={{ background: '#ff5f56' }} />
          <span className="h-3 w-3 rounded-full" style={{ background: '#ffbd2e' }} />
          <span className="h-3 w-3 rounded-full" style={{ background: '#27c93f' }} />
        </div>
        <span className="text-xs" style={{ color: '#cccccc' }}>SCOPE Sandbox — Microscopy Analysis</span>
        <div className="w-12" />
      </div>

      <div className="flex min-h-0 flex-1">
        {/* Activity bar */}
        <div className="flex w-12 shrink-0 flex-col items-center py-2" style={{ background: theme.activitybar }}>
          <div className="flex flex-col items-center gap-1">
            {activities.map(({ id, Icon, label }) => {
              const active = activity === id;
              return (
                <button
                  key={id}
                  title={label}
                  onClick={() => {
                    if (active) setSidebar((s) => !s);
                    else {
                      setActivity(id);
                      setSidebar(true);
                    }
                  }}
                  className="relative flex h-11 w-12 items-center justify-center transition-colors"
                  style={{ color: active ? theme.activitybarFgActive : theme.activitybarFg }}
                >
                  {active && (
                    <span
                      className="absolute left-0 top-1/2 h-6 w-0.5 -translate-y-1/2"
                      style={{ background: '#ffffff' }}
                    />
                  )}
                  <Icon size={24} strokeWidth={1.5} />
                </button>
              );
            })}
          </div>
        </div>

        {/* Sidebar */}
        {sidebar && (
          <div
            className="flex w-60 shrink-0 flex-col overflow-hidden"
            style={{ background: theme.sidebar, borderRight: `1px solid ${theme.border}` }}
          >
            <div
              className="flex h-9 shrink-0 items-center px-4 text-[11px] font-medium uppercase tracking-wider"
              style={{ color: theme.sidebarHeader }}
            >
              {activities.find((a) => a.id === activity)?.label}
            </div>
            <div className="min-h-0 flex-1 overflow-y-auto pb-2">
              {activity === 'files' ? (
                <Tree tree={files} expanded={expanded} toggle={toggleFolder} activePath={activeTab} openFile={openFile} />
              ) : (
                <div className="px-4 py-6 text-[13px]" style={{ color: theme.tabFg }}>
                  {activities.find((a) => a.id === activity)?.label} panel
                </div>
              )}
            </div>
          </div>
        )}

        {/* Editor + Output split */}
        <div ref={splitRef} className="flex min-w-0 flex-1">
          {/* Editor column */}
          <div className="flex min-w-0 flex-col" style={{ width: `${editorWidth}%` }}>
            <div className="flex h-9 shrink-0 items-stretch overflow-x-auto" style={{ background: theme.tabInactive }}>
              {openTabs.map((pathArr) => {
                const key = pathArr.join('/');
                const name = pathArr[pathArr.length - 1];
                const active = key === activeTab;
                const isDirty = dirty.has(key);
                const { Icon, color } = fileIcon(name);
                return (
                  <div
                    key={key}
                    onClick={() => setActiveTab(key)}
                    className="group flex cursor-pointer items-center gap-2 border-r px-3 text-[13px]"
                    style={{
                      background: active ? theme.tabActive : theme.tabInactive,
                      color: active ? theme.tabFgActive : theme.tabFg,
                      borderColor: theme.border,
                      borderTop: active ? `1px solid ${theme.accentBright}` : '1px solid transparent',
                    }}
                  >
                    <Icon size={15} style={{ color }} />
                    <span className="whitespace-nowrap">{name}</span>
                    <button
                      onClick={(e) => closeTab(key, e)}
                      className="flex h-5 w-5 items-center justify-center rounded"
                      style={{ color: active ? theme.tabFgActive : theme.tabFg }}
                    >
                      {isDirty ? (
                        <Circle size={9} fill="currentColor" className="group-hover:hidden" />
                      ) : null}
                      <X size={15} className={isDirty ? 'hidden group-hover:block' : 'opacity-0 group-hover:opacity-100'} />
                    </button>
                  </div>
                );
              })}
            </div>

            {activePathArr && (
              <div className="flex h-6 shrink-0 items-center gap-1 px-4 text-[12px]" style={{ background: theme.editor, color: theme.tabFg }}>
                {activePathArr.map((p, i) => (
                  <span key={i} className="flex items-center gap-1">
                    {i > 0 && <ChevronRight size={12} className="opacity-60" />}
                    {p}
                  </span>
                ))}
              </div>
            )}

            {activeNode ? (
              <Editor value={activeNode.content} onChange={updateContent} onCursor={setCursor} />
            ) : (
              <div
                className="flex min-h-0 flex-1 items-center justify-center text-sm"
                style={{ background: theme.editor, color: '#5a5a5a' }}
              >
                Select a file to start editing
              </div>
            )}
          </div>

          {/* Splitter */}
          <div
            onMouseDown={() => {
              dragging.current = true;
              document.body.style.cursor = 'col-resize';
            }}
            className="w-1 shrink-0 cursor-col-resize transition-colors hover:opacity-100"
            style={{ background: theme.border }}
            title="Drag to resize"
          />

          {/* Output column */}
          <OutputColumn
            compiled={compiled}
            logs={logs}
            onRun={run}
            onClear={() => setLogs([])}
            outputTab={outputTab}
            setOutputTab={setOutputTab}
            canvasRef={canvasRef}
            chartData={visualizationData}
          />
        </div>
      </div>

      {/* Status bar */}
      <div
        className="flex h-6 shrink-0 items-center justify-between px-3 text-[12px]"
        style={{ background: theme.statusBar, color: theme.statusFg }}
      >
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1">Ln {cursor.ln}, Col {cursor.col}</span>
        </div>
        <div className="flex items-center gap-3">
          <span>{activeNode ? 'SCOPE' : '—'}</span>
        </div>
      </div>
    </div>
  );
}
