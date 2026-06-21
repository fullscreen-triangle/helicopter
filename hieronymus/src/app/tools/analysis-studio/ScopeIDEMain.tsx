'use client';

import React, { useState, useRef, useCallback, useMemo, useEffect } from 'react';
import {
  Files, Search, GitBranch, Play, ChevronRight, ChevronDown,
  X, Circle, FileCode2, FileJson, FileText, Folder, FolderOpen,
  Terminal as TerminalIcon, BarChart2, Image as ImageIcon, Trash2, RefreshCw,
} from 'lucide-react';
import { compile } from '@/lib/scope-compiler';
import { runScope } from '@/lib/scope-runtime/runtime';
import type { ScopeResult } from '@/lib/scope-runtime/result-types';
import SpectralPowerChart from '@/app/tools/scope-playground/components/charts/SpectralPowerChart';
import EntropyTrajectoryChart from '@/app/tools/scope-playground/components/charts/EntropyTrajectoryChart';
import UncertaintyBar from '@/app/tools/scope-playground/components/charts/UncertaintyBar';
import ScaleHistogram from '@/app/tools/scope-playground/components/charts/ScaleHistogram';

// ─────────────────────────────────────────────────────────────────────────────
// Theme
// ─────────────────────────────────────────────────────────────────────────────
const T = {
  titlebar:   '#3c3c3c', activitybar: '#333333', activitybarFg: '#858585',
  activitybarFgActive: '#ffffff', sidebar: '#252526', sidebarFg: '#cccccc',
  sidebarHeader: '#bbbbbb', editor: '#1e1e1e', editorFg: '#d4d4d4',
  tabBar: '#252526', tabActive: '#1e1e1e', tabInactive: '#2d2d2d',
  tabFg: '#969696', tabFgActive: '#ffffff', border: '#3c3c3c',
  accent: '#0e639c', accentBright: '#007acc', statusBar: '#007acc',
  statusFg: '#ffffff', gutter: '#858585', lineActive: '#2a2d2e',
};

// ─────────────────────────────────────────────────────────────────────────────
// Tutorial scripts — each one builds on the previous
// ─────────────────────────────────────────────────────────────────────────────
const TUTORIAL_SCRIPTS: Record<string, string> = {

'tutorial_01_observe.scope': `// Tutorial 01 — Observe
// The simplest possible SCOPE program.
// Load one image, measure the scale field α(x,y), emit position.
// Introduces: observe(), coordinate_space, depth.

scope hello_microscopy {
  coordinate_space {
    field 100 x 100 µm
    depth 8
    lambda_s 0.10
    lambda_t 0.05
  }

  hello = observe(load(db="BBBC", dataset="BBBC007", image="A9 p10d.tif"), n = 8)
    |> visualise(scale_field)
}`,

'tutorial_02_scale_field.scope': `// Tutorial 02 — Scale Field
// Compute the spectral metric α(x,y) and inspect its statistics.
// Introduces: visualise(scale_field), goal/SNR criterion.

scope scale_field_analysis {
  coordinate_space {
    field 100 x 100 µm
    depth 10
    lambda_s 0.10
    lambda_t 0.05
  }

  goal {
    snr > 4.0
  }

  measure = observe(load(db="BBBC", dataset="BBBC007", image="A9 p9d.tif"), n = 10)
    |> visualise(scale_field)
}`,

'tutorial_03_segmentation.scope': `// Tutorial 03 — Segmentation
// Access nucleus_a and nucleus_b from a single DAPI image.
// Introduces: access(), visualise(segmentation), conservation rule.

scope nucleus_segmentation {
  coordinate_space {
    field 100 x 100 µm
    depth 10
    lambda_s 0.10
    lambda_t 0.05
  }

  rule conservation(dna_mass) {
    invariant: "DAPI-stained area conserved"
    epsilon: 0.008
  }

  segment = observe(load(db="BBBC", dataset="BBBC007", image="A9 p10d.tif"), n = 10)
    |> catalyze(conservation(dna_mass))
    |> access(nucleus_a)
    |> access(nucleus_b)
    |> visualise(segmentation)
}`,

'tutorial_04_distance.scope': `// Tutorial 04 — Nuclear Separation
// Measure the geodesic distance between two nuclei.
// Introduces: measure_distance(), visualise(geodesic), uncertainty goal.

scope nuclear_separation {
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

  measure = observe(load(db="BBBC", dataset="BBBC007", image="A9 p10d.tif"), n = 10)
    |> catalyze(conservation(dna_mass))
    |> access(nucleus_a)
    |> access(nucleus_b)
    |> measure_distance(nucleus_a, nucleus_b)
    |> visualise(geodesic)
}`,

'tutorial_05_confidence.scope': `// Tutorial 05 — Confidence & Catalysis
// Add a bilateral symmetry constraint with reduced confidence.
// Introduces: catalyze(..., confidence=…), multiple constraints, S-entropy effect.

scope spindle_with_confidence {
  coordinate_space {
    field 64 x 64 µm
    depth 10
    lambda_s 0.08
    lambda_t 0.03
  }

  goal {
    distance_uncertainty < 0.2 µm
    snr > 5.0
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
    |> catalyze(symmetry(bilateral), confidence = 0.80)
    |> catalyze(conservation(dna_mass))
    |> access(nucleus_a, threshold = 0.70)
    |> access(nucleus_b, threshold = 0.70)
    |> measure_distance(nucleus_a, nucleus_b)
    |> visualise(distance_map)
}`,

'tutorial_06_dispatch.scope': `// Tutorial 06 — Dispatch & Cell Classification
// Use a dispatch block to select a morphism based on cell phase.
// Introduces: channels, dispatch, when/do, full five-phase pipeline.

scope phase_dispatch {
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

'tutorial_07_full_pipeline.scope': `// Tutorial 07 — Full Pipeline
// All five phases: COMPILE → ASSIGN → MEASURE → EXECUTE → EMIT.
// Two constraints, dispatch, goals, CRLB criterion.
// Uses f9620 image for a different cell morphology.

scope full_pipeline_f9620 {
  channels {
    sync dapi at 0.08 µm/pixel
    cell METAPHASE bounds (-0.8e-6, 0.8e-6) action analyze
  }

  coordinate_space {
    field 82 x 82 µm
    depth 10
    lambda_s 0.09
    lambda_t 0.04
  }

  goal {
    distance_uncertainty < 0.3 µm
    snr > 5.0
    crlb_pixels < 0.15
  }

  rule conservation(dna_mass) {
    invariant: "DAPI-stained area conserved"
    epsilon: 0.008
  }

  rule phase_lock(chromatin) {
    invariant: "chromatin condensation consistent with metaphase"
    epsilon: 0.012
  }

  analyze = observe(load(db="BBBC", dataset="BBBC007", image="20P1_POS0002_D_1UL.tif"), n = 10)
    |> visualise(scale_field)
    |> catalyze(conservation(dna_mass))
    |> catalyze(phase_lock(chromatin), confidence = 0.85)
    |> access(nucleus_a)
    |> access(nucleus_b)
    |> visualise(segmentation)
    |> measure_distance(nucleus_a, nucleus_b)
    |> visualise(geodesic)

  dispatch {
    when METAPHASE do execute(analyze)
  }
}`,
};

// ─────────────────────────────────────────────────────────────────────────────
// File system
// ─────────────────────────────────────────────────────────────────────────────
const initialFiles: any = {
  examples: {
    type: 'folder',
    children: Object.fromEntries(
      Object.entries(TUTORIAL_SCRIPTS).map(([name, content]) => [
        name,
        { type: 'file', lang: 'scope', content },
      ])
    ),
  },
  datasets: {
    type: 'folder',
    children: {
      'BBBC007_v1_images': { type: 'folder', children: {
        'A9_p10d.tif': { type: 'file', lang: 'tif', content: '// HeLa A9 p10 DAPI — 512×512, 0.1 µm/px' },
        'A9_p9d.tif':  { type: 'file', lang: 'tif', content: '// HeLa A9 p9 DAPI — 512×512, 0.1 µm/px' },
        '17P1_POS0006_D_1UL.tif': { type: 'file', lang: 'tif', content: '// f96 spindle DAPI — 512×512, 0.08 µm/px' },
        '20P1_POS0002_D_1UL.tif': { type: 'file', lang: 'tif', content: '// f9620 fused DAPI — 512×512, 0.08 µm/px' },
      }},
      'AICS-24-part06': { type: 'folder', children: {} },
      'human_HT29_colon_cancer': { type: 'folder', children: {} },
    },
  },
  'my-analysis': {
    type: 'folder',
    children: {
      'analysis.scope': {
        type: 'file', lang: 'scope',
        content: `// My Analysis — start from tutorial_04 and customise
scope my_analysis {
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

  measure = observe(load(db="BBBC", dataset="BBBC007", image="A9 p10d.tif"), n = 10)
    |> catalyze(conservation(dna_mass))
    |> access(nucleus_a)
    |> access(nucleus_b)
    |> measure_distance(nucleus_a, nucleus_b)
    |> visualise(geodesic)
}`,
      },
    },
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
const fileIcon = (name: string) => {
  if (name.endsWith('.scope')) return { Icon: FileCode2, color: '#00d4ff' };
  if (name.endsWith('.json')) return { Icon: FileJson, color: '#cbcb41' };
  if (name.endsWith('.tif') || name.endsWith('.tiff')) return { Icon: ImageIcon, color: '#4ec9b0' };
  return { Icon: FileText, color: '#858585' };
};

const getNode = (tree: any, path: string[]) => {
  let n: any = { children: tree };
  for (const p of path) { n = n.children[p]; if (!n) return null; }
  return n;
};

// ─────────────────────────────────────────────────────────────────────────────
// File tree
// ─────────────────────────────────────────────────────────────────────────────
function Tree({ tree, path = [], depth = 0, expanded, toggle, activePath, openFile }: any) {
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
              onClick={() => isFolder ? toggle(key) : openFile(fullPath)}
              className="flex w-full items-center gap-1 py-0.5 pr-2 text-left text-[13px] leading-relaxed"
              style={{ paddingLeft: 8 + depth * 12, color: T.sidebarFg, background: isActive ? T.lineActive : 'transparent' }}
              onMouseEnter={e => { if (!isActive) (e.currentTarget as HTMLElement).style.background = '#2a2d2e'; }}
              onMouseLeave={e => { if (!isActive) (e.currentTarget as HTMLElement).style.background = 'transparent'; }}
            >
              {isFolder
                ? isOpen ? <ChevronDown size={14} className="shrink-0 opacity-70" /> : <ChevronRight size={14} className="shrink-0 opacity-70" />
                : <span className="w-[14px] shrink-0" />}
              <Icon size={15} className="shrink-0" style={{ color }} />
              <span className="truncate">{name}</span>
            </button>
            {isFolder && isOpen && (
              <Tree tree={node.children} path={fullPath} depth={depth + 1}
                expanded={expanded} toggle={toggle} activePath={activePath} openFile={openFile} />
            )}
          </div>
        );
      })}
    </>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Editor
// ─────────────────────────────────────────────────────────────────────────────
function Editor({ value, onChange, onCursor }: any) {
  const gutterRef = useRef<HTMLDivElement>(null);
  const lines = value.split('\n');
  return (
    <div className="flex min-h-0 flex-1" style={{ background: T.editor }}>
      <div ref={gutterRef} className="select-none overflow-hidden py-3 text-right font-mono text-[13px] leading-[1.5]"
        style={{ color: T.gutter, minWidth: 52, paddingRight: 16 }}>
        {lines.map((_: any, i: number) => <div key={i}>{i + 1}</div>)}
      </div>
      <textarea value={value} onChange={e => onChange(e.target.value)}
        onScroll={e => { if (gutterRef.current) gutterRef.current.scrollTop = (e.target as HTMLTextAreaElement).scrollTop; }}
        onKeyUp={e => { const u = (e.target as HTMLTextAreaElement).value.slice(0, (e.target as HTMLTextAreaElement).selectionStart); onCursor({ ln: u.split('\n').length, col: u.length - u.lastIndexOf('\n') }); }}
        onClick={e => { const u = (e.target as HTMLTextAreaElement).value.slice(0, (e.target as HTMLTextAreaElement).selectionStart); onCursor({ ln: u.split('\n').length, col: u.length - u.lastIndexOf('\n') }); }}
        spellCheck={false}
        className="min-h-0 flex-1 resize-none border-0 bg-transparent py-3 pr-4 font-mono text-[13px] leading-[1.5] outline-none"
        style={{ color: T.editorFg, tabSize: 2, caretColor: '#fff' }}
      />
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Viridis helper (for inline canvas draws)
// ─────────────────────────────────────────────────────────────────────────────
const VIRIDIS: [number,number,number][] = [
  [68,1,84],[72,35,116],[64,67,135],[52,94,141],[41,120,142],
  [32,144,140],[34,167,132],[68,190,112],[121,209,81],[189,222,38],[253,231,37],
];
function viridis(t: number): [number,number,number] {
  const i = Math.min(VIRIDIS.length - 2, Math.max(0, Math.floor(t * (VIRIDIS.length - 1))));
  const f = t * (VIRIDIS.length - 1) - i;
  const a = VIRIDIS[i], b = VIRIDIS[i + 1];
  return [a[0]+f*(b[0]-a[0]), a[1]+f*(b[1]-a[1]), a[2]+f*(b[2]-a[2])];
}

// ─────────────────────────────────────────────────────────────────────────────
// ImagePanel — shows raw image + scale field + segmentation + geodesic annotations
// ─────────────────────────────────────────────────────────────────────────────
function ImagePanel({ result }: { result: ScopeResult | null }) {
  const rawRef  = useRef<HTMLCanvasElement>(null);
  const segRef  = useRef<HTMLCanvasElement>(null);
  const sfRef   = useRef<HTMLCanvasElement>(null);
  const geoRef  = useRef<HTMLCanvasElement>(null);
  const dmRef   = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!result) return;
    const { visualData } = result;
    const W = visualData.width, H = visualData.height;

    // 1. Raw grayscale
    const draw = (ref: React.RefObject<HTMLCanvasElement | null>, fn: (ctx: CanvasRenderingContext2D) => void) => {
      const c = ref.current; if (!c) return;
      c.width = W; c.height = H;
      const ctx = c.getContext('2d'); if (!ctx) return;
      fn(ctx);
    };

    draw(rawRef, ctx => {
      const id = ctx.createImageData(W, H);
      for (let i = 0; i < W * H; i++) {
        const v = Math.round(Math.max(0, Math.min(1, visualData.rawImage[i])) * 255);
        id.data[i*4]=v; id.data[i*4+1]=v; id.data[i*4+2]=v; id.data[i*4+3]=255;
      }
      ctx.putImageData(id, 0, 0);
    });

    // 2. Scale field viridis
    draw(sfRef, ctx => {
      const f = visualData.scaleField;
      if (!f) return;
      let mn = Infinity, mx = -Infinity;
      for (let i = 0; i < f.length; i++) { if (isFinite(f[i])) { if (f[i]<mn) mn=f[i]; if (f[i]>mx) mx=f[i]; } }
      const rng = (mx-mn)||1;
      const id = ctx.createImageData(W, H);
      for (let i = 0; i < W*H; i++) {
        const [r,g,b] = viridis((f[i]-mn)/rng);
        id.data[i*4]=r; id.data[i*4+1]=g; id.data[i*4+2]=b; id.data[i*4+3]=255;
      }
      ctx.putImageData(id, 0, 0);
    });

    // 3. Segmentation overlay
    draw(segRef, ctx => {
      // base raw
      const id0 = ctx.createImageData(W, H);
      for (let i = 0; i < W*H; i++) {
        const v = Math.round(Math.max(0,Math.min(1,visualData.rawImage[i]))*255);
        id0.data[i*4]=v; id0.data[i*4+1]=v; id0.data[i*4+2]=v; id0.data[i*4+3]=255;
      }
      ctx.putImageData(id0, 0, 0);
      // overlay
      const mask = visualData.segmentationMask;
      if (mask) {
        const mid = ctx.createImageData(W, H);
        for (let i = 0; i < W*H; i++) {
          if (mask[i]===1) { mid.data[i*4]=78; mid.data[i*4+1]=201; mid.data[i*4+2]=176; mid.data[i*4+3]=130; }
          else if (mask[i]===2) { mid.data[i*4]=197; mid.data[i*4+1]=134; mid.data[i*4+2]=192; mid.data[i*4+3]=130; }
        }
        const off = new OffscreenCanvas(W, H);
        off.getContext('2d')!.putImageData(mid, 0, 0);
        ctx.drawImage(off, 0, 0);
      }
      // contour
      const cont = visualData.segmentationContour;
      if (cont?.length) {
        ctx.fillStyle = '#4ec9b0';
        const step = Math.max(1, Math.floor(cont.length / 3000));
        for (let i = 0; i < cont.length; i += step) ctx.fillRect(cont[i][0], cont[i][1], 1, 1);
      }
      // label centroids
      const targets = Object.entries(result.visualData as any);
      // draw distance value annotation if available
      if (result.distance !== null) {
        ctx.fillStyle = 'rgba(0,0,0,0.65)';
        ctx.fillRect(2, H - 20, 200, 18);
        ctx.fillStyle = '#4ec9b0';
        ctx.font = 'bold 11px monospace';
        ctx.fillText(`d = ${result.distance.toFixed(3)} ± ${(result.uncertainty??0).toFixed(3)} µm`, 6, H - 6);
      }
    });

    // 4. Geodesic path
    draw(geoRef, ctx => {
      const id0 = ctx.createImageData(W, H);
      for (let i = 0; i < W*H; i++) {
        const v = Math.round(Math.max(0,Math.min(1,visualData.rawImage[i]))*255);
        id0.data[i*4]=v; id0.data[i*4+1]=v; id0.data[i*4+2]=v; id0.data[i*4+3]=255;
      }
      ctx.putImageData(id0, 0, 0);
      const path = visualData.geodesicPath;
      if (path?.length) {
        ctx.save();
        ctx.strokeStyle = '#ffd700'; ctx.lineWidth = 2;
        ctx.shadowColor = '#ffd700'; ctx.shadowBlur = 6;
        ctx.beginPath(); ctx.moveTo(path[0][0], path[0][1]);
        for (let i = 1; i < path.length; i++) ctx.lineTo(path[i][0], path[i][1]);
        ctx.stroke();
        for (const pt of [path[0], path[path.length-1]]) {
          ctx.fillStyle = '#ffd700'; ctx.beginPath(); ctx.arc(pt[0], pt[1], 4, 0, Math.PI*2); ctx.fill();
        }
        ctx.restore();
      }
      if (result.distance !== null) {
        ctx.fillStyle = 'rgba(0,0,0,0.65)';
        ctx.fillRect(2, H - 20, 220, 18);
        ctx.fillStyle = '#ffd700'; ctx.font = 'bold 11px monospace';
        ctx.fillText(`d = ${result.distance.toFixed(3)} ± ${(result.uncertainty??0).toFixed(3)} µm`, 6, H - 6);
      }
    });

    // 5. Distance map
    draw(dmRef, ctx => {
      const dm = visualData.distanceMap;
      if (!dm) { ctx.fillStyle='#0d1117'; ctx.fillRect(0,0,W,H); return; }
      let mn=Infinity, mx=-Infinity;
      for (let i=0;i<dm.length;i++) { if (isFinite(dm[i])) { if(dm[i]<mn)mn=dm[i]; if(dm[i]>mx)mx=dm[i]; } }
      const rng=(mx-mn)||1;
      const id=ctx.createImageData(W,H);
      for (let i=0;i<W*H;i++) {
        const [r,g,b]=viridis((dm[i]-mn)/rng);
        id.data[i*4]=r; id.data[i*4+1]=g; id.data[i*4+2]=b; id.data[i*4+3]=255;
      }
      ctx.putImageData(id,0,0);
    });

  }, [result]);

  if (!result) {
    return (
      <div className="flex h-full items-center justify-center" style={{ color: '#5a5a5a' }}>
        Run a program to see the processed image.
      </div>
    );
  }

  const panels = [
    { ref: rawRef,  label: 'Raw image' },
    { ref: sfRef,   label: 'Scale field α(x,y)' },
    { ref: segRef,  label: 'Segmentation + annotations' },
    { ref: geoRef,  label: 'Geodesic path' },
    { ref: dmRef,   label: 'Distance map T(x,y)' },
  ];

  return (
    <div className="h-full overflow-y-auto p-3 space-y-3">
      <div className="grid grid-cols-2 gap-3">
        {panels.map(({ ref, label }) => (
          <div key={label} className="space-y-1">
            <div className="text-[#858585] text-[10px]">{label}</div>
            <canvas ref={ref} className="w-full rounded border border-[#3a3a3a]"
              style={{ imageRendering: 'pixelated' }} />
          </div>
        ))}
        {/* S-entropy summary */}
        <div className="border border-[#3a3a3a] rounded p-3 space-y-2 text-[11px]">
          <div className="text-[#858585]">Partition state summary</div>
          <div className="space-y-1">
            {[
              ['S_k (knowledge)', result.sEntropy.sk, '#4ec9b0'],
              ['S_t (temporal)',  result.sEntropy.st, '#569cd6'],
              ['S_e (emission)',  result.sEntropy.se, '#c586c0'],
            ].map(([label, val, color]) => (
              <div key={label as string} className="flex items-center gap-2">
                <div className="text-[#858585] w-28">{label as string}</div>
                <div className="flex-1 h-2 rounded bg-[#1a1a1a]">
                  <div className="h-2 rounded" style={{ width: `${(val as number)*100}%`, background: color as string }} />
                </div>
                <div style={{ color: color as string }} className="w-12 text-right">{(val as number).toFixed(3)}</div>
              </div>
            ))}
            <div className="text-[#858585] pt-1">
              Σ = {result.sEntropy.sum.toFixed(12)} {Math.abs(result.sEntropy.sum-1)<1e-9?'✓':'⚠'}
            </div>
          </div>
          {result.distance !== null && (
            <div className="pt-1 space-y-0.5">
              <div className="text-[#4ec9b0]">d = {result.distance.toFixed(3)} µm</div>
              <div className="text-[#858585]">δd = ±{(result.uncertainty??0).toFixed(3)} µm ({((result.relativeUncertainty??0)*100).toFixed(2)}%)</div>
              <div className="text-[#858585]">SNR = {result.snr.toFixed(1)}  CRLB = {result.crlbPixels.toFixed(3)} px</div>
            </div>
          )}
          {result.goalStatus.length > 0 && (
            <div className="pt-1 space-y-0.5">
              {result.goalStatus.map((g, i) => (
                <div key={i} className={g.passed ? 'text-[#4caf50]' : 'text-[#f44336]'}>
                  {g.passed?'✓':'✗'} {g.metric} {g.op} {g.threshold} {g.unit}
                  <span className="text-[#858585]"> (got {g.actual.toPrecision(4)})</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// ChartsPanel — 10 charts per execution
// ─────────────────────────────────────────────────────────────────────────────
function ChartsPanel({ result, hfResult }: { result: ScopeResult | null; hfResult: HFResult | null }) {
  if (!result) {
    return (
      <div className="flex h-full items-center justify-center" style={{ color: '#5a5a5a' }}>
        Run a program to see charts.
      </div>
    );
  }

  const cd = result.chartData;

  // Chart 5: SNR vs CRLB scatter (inline SVG)
  const snrVal = result.snr;
  const crlbVal = result.crlbPixels;

  // Chart 6: Goal pass/fail bar
  const goals = result.goalStatus;

  // Chart 7: Partition state tree (bar per step)
  const psNodes = result.visualData.partitionStates ?? [];

  // Chart 8: Scale field radial profile (inline)
  const sf = result.visualData.scaleField;
  const W = result.visualData.width, H = result.visualData.height;
  const cx = Math.floor(W/2), cy = Math.floor(H/2);
  const radialBins = 16;
  const radialProfile: number[] = Array(radialBins).fill(0);
  const radialCounts: number[]  = Array(radialBins).fill(0);
  const maxR = Math.sqrt(cx*cx + cy*cy);
  if (sf) {
    for (let y=0; y<H; y++) for (let x=0; x<W; x++) {
      const r = Math.sqrt((x-cx)**2 + (y-cy)**2);
      const bin = Math.min(radialBins-1, Math.floor(r/maxR*radialBins));
      radialProfile[bin] += sf[y*W+x];
      radialCounts[bin]++;
    }
  }
  const radialMean = radialProfile.map((s,i) => radialCounts[i] ? s/radialCounts[i] : 0);
  const radialMax = Math.max(...radialMean, 0.001);

  // Chart 9: Uncertainty budget (stacked bar)
  const distVal = result.distance ?? 0;
  const uncVal  = result.uncertainty ?? 0;

  // Chart 10: Channel capacity arc
  const C = result.channelCapacity;

  return (
    <div className="h-full overflow-y-auto p-3 space-y-3">
      {/* Row 1 */}
      <div className="grid grid-cols-2 gap-3">
        <div>
          <div className="text-[#858585] text-[10px] mb-1">Chart 1 — Spectral power |û(k)|² vs k</div>
          <SpectralPowerChart data={cd.spectralPower} exponent={cd.powerLawExponent} />
        </div>
        <div>
          <div className="text-[#858585] text-[10px] mb-1">Chart 2 — Scale field α(x,y) histogram</div>
          <ScaleHistogram data={cd.scaleHistogram} mean={cd.alphaMean ?? cd.powerLawExponent} />
        </div>
      </div>

      {/* Row 2 */}
      <div className="grid grid-cols-2 gap-3">
        <div>
          <div className="text-[#858585] text-[10px] mb-1">Chart 3 — S-entropy trajectory</div>
          <EntropyTrajectoryChart data={cd.entropyTrajectory} />
        </div>
        <div>
          <div className="text-[#858585] text-[10px] mb-1">Chart 4 — Distance ± uncertainty</div>
          <UncertaintyBar data={cd.uncertaintyBar} />
        </div>
      </div>

      {/* Row 3 */}
      <div className="grid grid-cols-2 gap-3">
        {/* Chart 5: SNR / CRLB */}
        <InlineChart title="Chart 5 — SNR & CRLB">
          {(W, H) => {
            const mrgL=52, mrgB=30, mrgT=12, mrgR=12;
            const iW=W-mrgL-mrgR, iH=H-mrgT-mrgB;
            const vals = [{ label:'SNR', val:snrVal, max:20, color:'#4ec9b0' }, { label:'CRLB (px)', val:crlbVal, max:1, color:'#c586c0', invert:true }];
            return (
              <g transform={`translate(${mrgL},${mrgT})`}>
                <rect width={iW} height={iH} fill="#0d1117" />
                {vals.map(({ label, val, max, color, invert }, i) => {
                  const bw=iW/vals.length*0.55, x=iW/(vals.length)*(i+0.5)-bw/2;
                  const pct = invert ? Math.max(0,1-val/max) : Math.min(1,val/max);
                  const bh = pct * iH;
                  return (
                    <g key={label}>
                      <rect x={x} y={iH-bh} width={bw} height={bh} fill={color} opacity={0.8} />
                      <text x={x+bw/2} y={iH-bh-4} fill={color} fontSize={9} textAnchor="middle">{val.toFixed(2)}</text>
                      <text x={x+bw/2} y={iH+16} fill="#858585" fontSize={9} textAnchor="middle">{label}</text>
                    </g>
                  );
                })}
                <line x1={0} y1={iH*0.5} x2={iW} y2={iH*0.5} stroke="#3a3a3a" strokeDasharray="3,2" />
                <text x={-2} y={iH*0.5+4} fill="#555" fontSize={8} textAnchor="end">50%</text>
              </g>
            );
          }}
        </InlineChart>

        {/* Chart 6: Goal pass/fail */}
        <InlineChart title="Chart 6 — Goal criteria">
          {(W, H) => {
            const mrgL=8, mrgB=12, mrgT=12, mrgR=8;
            const iW=W-mrgL-mrgR, iH=H-mrgT-mrgB;
            if (goals.length === 0) return (
              <g transform={`translate(${mrgL},${mrgT})`}>
                <text x={iW/2} y={iH/2} fill="#555" fontSize={11} textAnchor="middle">No goal criteria</text>
              </g>
            );
            const rowH = iH / goals.length;
            return (
              <g transform={`translate(${mrgL},${mrgT})`}>
                {goals.map((g, i) => {
                  const barW = Math.min(iW, (g.actual / g.threshold) * iW * 0.8);
                  const color = g.passed ? '#4caf50' : '#f44336';
                  return (
                    <g key={i} transform={`translate(0,${i*rowH})`}>
                      <rect x={0} y={4} width={iW} height={rowH-8} fill="#0d1117" rx={2} />
                      <rect x={0} y={4} width={barW} height={rowH-8} fill={color} opacity={0.35} rx={2} />
                      <line x1={(g.threshold/g.threshold)*iW*0.8} y1={2} x2={(g.threshold/g.threshold)*iW*0.8} y2={rowH-2} stroke={color} strokeWidth={1.5} strokeDasharray="3,2" />
                      <text x={4} y={rowH/2+4} fill={color} fontSize={9}>{g.passed?'✓':'✗'} {g.metric} {g.op} {g.threshold}{g.unit}</text>
                      <text x={iW-4} y={rowH/2+4} fill="#858585" fontSize={9} textAnchor="end">{g.actual.toPrecision(3)}</text>
                    </g>
                  );
                })}
              </g>
            );
          }}
        </InlineChart>
      </div>

      {/* Row 4 */}
      <div className="grid grid-cols-2 gap-3">
        {/* Chart 7: Partition state S_k progression */}
        <InlineChart title="Chart 7 — S_k progression through steps">
          {(W, H) => {
            const mrgL=44, mrgB=36, mrgT=12, mrgR=12;
            const iW=W-mrgL-mrgR, iH=H-mrgT-mrgB;
            if (!psNodes.length) return <text x={W/2} y={H/2} fill="#555" fontSize={11} textAnchor="middle">No partition states</text>;
            const xScale = (i: number) => (i/(psNodes.length-1||1))*iW;
            const yScale = (v: number) => iH * (1-v);
            return (
              <g transform={`translate(${mrgL},${mrgT})`}>
                <rect width={iW} height={iH} fill="#0d1117" />
                {[['sk','#4ec9b0'],['st','#569cd6'],['se','#c586c0']].map(([k,c]) => (
                  <polyline key={k} fill="none" stroke={c} strokeWidth={1.5} opacity={0.9}
                    points={psNodes.map((n,i) => `${xScale(i)},${yScale((n as any)[k])}`).join(' ')} />
                ))}
                {psNodes.map((n, i) => (
                  <g key={i}>
                    <circle cx={xScale(i)} cy={yScale(n.sk)} r={2.5} fill="#4ec9b0" />
                    {i % Math.max(1,Math.floor(psNodes.length/4)) === 0 && (
                      <text x={xScale(i)} y={iH+14} fill="#858585" fontSize={7} textAnchor="middle"
                        transform={`rotate(-20,${xScale(i)},${iH+14})`}>{n.label.slice(0,12)}</text>
                    )}
                  </g>
                ))}
                {[0,0.25,0.5,0.75,1].map(v => (
                  <g key={v}>
                    <line x1={0} x2={iW} y1={yScale(v)} y2={yScale(v)} stroke="#1e2a2a" strokeDasharray="2,2" />
                    <text x={-4} y={yScale(v)+4} fill="#555" fontSize={8} textAnchor="end">{v.toFixed(2)}</text>
                  </g>
                ))}
              </g>
            );
          }}
        </InlineChart>

        {/* Chart 8: Radial profile of α */}
        <InlineChart title="Chart 8 — α radial profile (centre→edge)">
          {(W, H) => {
            const mrgL=44, mrgB=30, mrgT=12, mrgR=12;
            const iW=W-mrgL-mrgR, iH=H-mrgT-mrgB;
            const xS = (i: number) => (i/(radialBins-1))*iW;
            const yS = (v: number) => iH*(1-v/radialMax);
            const pts = radialMean.map((v,i) => `${xS(i)},${yS(v)}`).join(' ');
            return (
              <g transform={`translate(${mrgL},${mrgT})`}>
                <rect width={iW} height={iH} fill="#0d1117" />
                <polyline fill="none" stroke="#4ec9b0" strokeWidth={2} points={pts} />
                {radialMean.map((v,i) => <circle key={i} cx={xS(i)} cy={yS(v)} r={2} fill="#4ec9b0" />)}
                {[0,0.5,1].map(t => {
                  const v=t*radialMax;
                  return <g key={t}>
                    <line x1={0} x2={iW} y1={yS(v)} y2={yS(v)} stroke="#1e2a2a" strokeDasharray="2,2" />
                    <text x={-4} y={yS(v)+4} fill="#555" fontSize={8} textAnchor="end">{v.toFixed(2)}</text>
                  </g>;
                })}
                <text x={iW/2} y={iH+22} fill="#858585" fontSize={9} textAnchor="middle">radial bin (centre → edge)</text>
                <text transform="rotate(-90)" x={-iH/2} y={-32} fill="#858585" fontSize={9} textAnchor="middle">ᾱ</text>
              </g>
            );
          }}
        </InlineChart>
      </div>

      {/* Row 5 */}
      <div className="grid grid-cols-2 gap-3">
        {/* Chart 9: Uncertainty budget */}
        <InlineChart title="Chart 9 — Measurement uncertainty budget">
          {(W, H) => {
            const mrgL=52, mrgB=30, mrgT=12, mrgR=12;
            const iW=W-mrgL-mrgR, iH=H-mrgT-mrgB;
            const total = Math.max(distVal + uncVal*2, 0.001);
            const dBar = (distVal/total)*iH, uBar = (uncVal/total)*iH;
            const cx2 = iW/2;
            const bw = iW*0.3;
            return (
              <g transform={`translate(${mrgL},${mrgT})`}>
                <rect width={iW} height={iH} fill="#0d1117" />
                {/* distance bar */}
                <rect x={cx2-bw-4} y={iH-dBar} width={bw} height={dBar} fill="#569cd6" opacity={0.85} />
                <text x={cx2-bw/2-4} y={iH-dBar-4} fill="#569cd6" fontSize={9} textAnchor="middle">
                  {distVal.toFixed(3)} µm
                </text>
                <text x={cx2-bw/2-4} y={iH+18} fill="#858585" fontSize={9} textAnchor="middle">distance</text>
                {/* uncertainty bar */}
                <rect x={cx2+4} y={iH-uBar} width={bw} height={uBar} fill="#c586c0" opacity={0.85} />
                <text x={cx2+bw/2+4} y={iH-uBar-4} fill="#c586c0" fontSize={9} textAnchor="middle">
                  ±{uncVal.toFixed(3)}
                </text>
                <text x={cx2+bw/2+4} y={iH+18} fill="#858585" fontSize={9} textAnchor="middle">δd</text>
                {/* relative uncertainty */}
                {distVal > 0 && (
                  <text x={iW/2} y={12} fill="#dcdcaa" fontSize={9} textAnchor="middle">
                    {((uncVal/distVal)*100).toFixed(2)}% relative uncertainty
                  </text>
                )}
                {/* y axis */}
                {[0,0.5,1].map(t => {
                  const y = iH*(1-t);
                  return <g key={t}>
                    <line x1={0} x2={iW} y1={y} y2={y} stroke="#1e2a2a" strokeDasharray="2,2"/>
                    <text x={-4} y={y+4} fill="#555" fontSize={8} textAnchor="end">{(t*total).toFixed(2)}</text>
                  </g>;
                })}
              </g>
            );
          }}
        </InlineChart>

        {/* Chart 10: HF classification (live) or channel capacity (fallback) */}
        <InlineChart title="Chart 10 — HF cell classification / Channel capacity">
          {(W, H) => {
            if (hfResult?.classification?.length) {
              const cls = hfResult.classification;
              const mrgL = 8, mrgR = 8, mrgT = 12, mrgB = 8;
              const iW = W - mrgL - mrgR, iH = H - mrgT - mrgB;
              const rowH = iH / cls.length;
              const cols = ['#4ec9b0','#569cd6','#c586c0','#dcdcaa','#9cdcfe'];
              return (
                <g transform={`translate(${mrgL},${mrgT})`}>
                  {cls.map((c, i) => {
                    const bw = c.score * iW * 0.85;
                    const col = cols[i % cols.length];
                    return (
                      <g key={i} transform={`translate(0,${i * rowH})`}>
                        <rect x={0} y={2} width={bw} height={rowH - 4} fill={col} opacity={0.75} rx={2} />
                        <text x={4} y={rowH / 2 + 4} fill="#111" fontSize={8} fontWeight="bold">
                          {c.label.split(',')[0].slice(0, 24)}
                        </text>
                        <text x={iW} y={rowH / 2 + 4} fill={col} fontSize={9} textAnchor="end">
                          {(c.score * 100).toFixed(1)}%
                        </text>
                      </g>
                    );
                  })}
                </g>
              );
            }
            // Fallback: channel capacity arc
            const cx2=W/2, cy2=H/2, r=Math.min(W,H)*0.32;
            const maxC = 8, pct = Math.min(1, C/maxC);
            const arcPt = (frac: number) => {
              const a = -Math.PI/2 + frac*2*Math.PI;
              return { x: cx2 + r*Math.cos(a), y: cy2 + r*Math.sin(a) };
            };
            const p0 = arcPt(0), p1 = arcPt(pct);
            const largeArc = pct > 0.5 ? 1 : 0;
            return (
              <>
                <circle cx={cx2} cy={cy2} r={r} fill="none" stroke="#1e2a2a" strokeWidth={18} />
                <path d={`M ${p0.x} ${p0.y} A ${r} ${r} 0 ${largeArc} 1 ${p1.x} ${p1.y}`}
                  fill="none" stroke="#569cd6" strokeWidth={18} strokeLinecap="round" />
                <text x={cx2} y={cy2-8} fill="#d4d4d4" fontSize={22} textAnchor="middle" fontWeight="bold">{C.toFixed(2)}</text>
                <text x={cx2} y={cy2+14} fill="#858585" fontSize={11} textAnchor="middle">bits/px</text>
                <text x={cx2} y={H-10} fill="#858585" fontSize={9} textAnchor="middle">Shannon C = ½ log₂(1 + SNR²)</text>
              </>
            );
          }}
        </InlineChart>
      </div>

      {/* HF segmentation + DINOv2 features — shown when available */}
      {hfResult && (hfResult.segmentation?.length || hfResult.features?.length) ? (
        <div className="grid grid-cols-2 gap-3">
          {hfResult.segmentation?.length ? (
            <div className="bg-[#0d1117] border border-[#3a3a3a] rounded p-2">
              <div className="text-[#858585] text-[10px] mb-2">HF segmentation — mask2former instances</div>
              <div className="space-y-1">
                {hfResult.segmentation.map((s, i) => (
                  <div key={i} className="flex items-center gap-2 text-[11px]">
                    <div className="h-2 rounded flex-1 bg-[#1a1a1a]">
                      <div className="h-2 rounded bg-[#4ec9b0]" style={{ width: `${s.score * 100}%` }} />
                    </div>
                    <span className="text-[#4ec9b0] w-10 text-right">{(s.score*100).toFixed(1)}%</span>
                    <span className="text-[#858585] truncate w-28">{s.label}</span>
                  </div>
                ))}
              </div>
            </div>
          ) : <div />}

          {hfResult.features?.length ? (
            <InlineChart title="HF DINOv2 feature embedding (first 16 dims)">
              {(W, H) => {
                const feats = hfResult.features!;
                const mn = Math.min(...feats), mx = Math.max(...feats);
                const rng = (mx - mn) || 1;
                const bw = W / feats.length;
                const midY = H / 2;
                return (
                  <>
                    <line x1={0} x2={W} y1={midY} y2={midY} stroke="#2a2a2a" />
                    {feats.map((v, i) => {
                      const norm = (v - mn) / rng;
                      const barH = norm * (H * 0.42);
                      const col = norm > 0.5 ? '#569cd6' : '#c586c0';
                      return (
                        <rect key={i} x={i * bw + 1} y={midY - barH} width={bw - 2} height={Math.abs(barH)}
                          fill={col} opacity={0.8} rx={1} />
                      );
                    })}
                    <text x={W/2} y={H-4} fill="#555" fontSize={8} textAnchor="middle">dim index</text>
                  </>
                );
              }}
            </InlineChart>
          ) : <div />}
        </div>
      ) : null}
    </div>
  );
}

// Inline SVG chart wrapper
function InlineChart({ title, children }: { title: string; children: (W: number, H: number) => React.ReactNode }) {
  const W = 340, H = 200;
  return (
    <div className="bg-[#0d1117] border border-[#3a3a3a] rounded p-2">
      <div className="text-[#858585] text-[10px] mb-1">{title}</div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ height: H }}>
        {children(W, H)}
      </svg>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// HuggingFace inference — routed through /api/hf-inference (key stays server-side)
// ─────────────────────────────────────────────────────────────────────────────

export interface HFClassLabel { label: string; score: number; }
export interface HFResult {
  classification: HFClassLabel[];   // resnet-50 cell-type top-5
  segmentation:   Array<{ label: string; score: number; mask?: string }> | null;
  features:       number[] | null;  // DINOv2 CLS token (first 16 for display)
}

async function hfInfer(imageData: Float32Array, width: number, height: number): Promise<HFResult> {
  // Convert Float32Array grayscale → base64 PNG via OffscreenCanvas
  const toBase64 = async (): Promise<string> => {
    const canvas = new OffscreenCanvas(width, height);
    const ctx = canvas.getContext('2d')!;
    const id = ctx.createImageData(width, height);
    for (let i = 0; i < width * height; i++) {
      const v = Math.round(Math.max(0, Math.min(1, imageData[i])) * 255);
      id.data[i * 4] = v; id.data[i * 4 + 1] = v; id.data[i * 4 + 2] = v; id.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(id, 0, 0);
    const blob = await canvas.convertToBlob({ type: 'image/png' });
    return new Promise(resolve => {
      const fr = new FileReader();
      fr.onload = () => resolve((fr.result as string).split(',')[1]);
      fr.readAsDataURL(blob);
    });
  };

  const b64 = await toBase64();

  const call = async (model: string, inputs: unknown) => {
    const r = await fetch('/api/hf-inference', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, inputs }),
    });
    if (!r.ok) throw new Error(`HF ${model}: ${r.status}`);
    return r.json();
  };

  // Run classification and feature extraction in parallel; segmentation separately
  const [classRaw, featRaw] = await Promise.allSettled([
    call('microsoft/resnet-50', b64),
    call('facebook/dinov2-base', b64),
  ]);

  let segRaw: PromiseSettledResult<unknown>;
  try {
    segRaw = await Promise.resolve({ status: 'fulfilled' as const, value: await call('facebook/mask2former-swin-large-coco-instance', b64) });
  } catch (e) {
    segRaw = { status: 'rejected', reason: e };
  }

  const classification: HFClassLabel[] =
    classRaw.status === 'fulfilled' && Array.isArray(classRaw.value)
      ? (classRaw.value as HFClassLabel[]).slice(0, 5)
      : [];

  const segmentation =
    segRaw.status === 'fulfilled' && Array.isArray(segRaw.value)
      ? (segRaw.value as Array<{ label: string; score: number; mask?: string }>).slice(0, 8)
      : null;

  const features: number[] | null = (() => {
    if (featRaw.status !== 'fulfilled') return null;
    const v = featRaw.value;
    // DINOv2 returns { last_hidden_state: number[][] } or flat number[]
    if (Array.isArray(v)) return (v as number[]).slice(0, 16);
    if (v && typeof v === 'object' && 'last_hidden_state' in v) {
      const lhs = (v as { last_hidden_state: number[][] }).last_hidden_state;
      if (Array.isArray(lhs) && Array.isArray(lhs[0])) return lhs[0].slice(0, 16);
    }
    return null;
  })();

  return { classification, segmentation, features };
}

// ─────────────────────────────────────────────────────────────────────────────
// OutputColumn — 3 tabs: Console | Charts | Image
// ─────────────────────────────────────────────────────────────────────────────
function OutputColumn({ logs, onRun, onClear, outputTab, setOutputTab, result, hfResult, running }: {
  logs: Array<{ level: string; message: string }>;
  onRun: () => void;
  onClear: () => void;
  outputTab: string;
  setOutputTab: (t: string) => void;
  result: ScopeResult | null;
  hfResult: HFResult | null;
  running: boolean;
}) {
  const tabs = [
    { id: 'console', label: 'Console',  Icon: TerminalIcon },
    { id: 'charts',  label: 'Charts',   Icon: BarChart2 },
    { id: 'image',   label: 'Image',    Icon: ImageIcon },
  ];
  const levelColor: Record<string, string> = {
    log: '#d4d4d4', info: '#9cdcfe', warn: '#dcdcaa', error: '#f48771',
  };

  return (
    <div className="flex min-w-0 flex-1 flex-col" style={{ background: T.editor, borderLeft: `1px solid ${T.border}` }}>
      <div className="flex h-9 shrink-0 items-center justify-between pr-2" style={{ background: T.tabInactive }}>
        <div className="flex h-full">
          {tabs.map(({ id, label, Icon }) => {
            const active = outputTab === id;
            return (
              <button key={id} onClick={() => setOutputTab(id)}
                className="relative flex items-center gap-1.5 px-3 text-[12px] transition-colors"
                style={{ color: active ? T.tabFgActive : T.tabFg, background: active ? T.tabActive : 'transparent' }}>
                <Icon size={13} /> {label}
                {id === 'console' && logs.length > 0 && (
                  <span className="rounded-full px-1.5 text-[10px]" style={{ background: T.accent, color: '#fff' }}>
                    {logs.length}
                  </span>
                )}
                {active && <span className="absolute left-0 top-0 h-0.5 w-full" style={{ background: T.accentBright }} />}
              </button>
            );
          })}
        </div>
        <div className="flex items-center gap-1">
          {outputTab === 'console' && (
            <button onClick={onClear} title="Clear" className="flex h-6 w-6 items-center justify-center rounded"
              style={{ color: T.tabFg }}>
              <Trash2 size={14} />
            </button>
          )}
          <button onClick={onRun} disabled={running}
            className="flex h-6 items-center gap-1 rounded px-2 text-[12px] disabled:opacity-50"
            style={{ background: T.accent, color: '#fff' }}>
            {running ? <RefreshCw size={12} className="animate-spin" /> : <Play size={12} fill="white" />}
            {running ? 'Running…' : 'Run'}
          </button>
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-hidden">
        {outputTab === 'console' && (
          <div className="h-full overflow-y-auto p-2 font-mono text-[12px] leading-relaxed">
            {logs.length === 0
              ? <div className="px-1 pt-1" style={{ color: '#5a5a5a' }}>Press Run to execute the active script.</div>
              : logs.map((l, i) => (
                <div key={i} className="border-b px-1 py-0.5" style={{ color: levelColor[l.level] ?? '#d4d4d4', borderColor: '#2a2a2a' }}>
                  <span className="mr-2 opacity-40">[{l.level}]</span>{l.message}
                </div>
              ))
            }
          </div>
        )}
        {outputTab === 'charts'  && <ChartsPanel result={result} hfResult={hfResult} />}
        {outputTab === 'image'   && <ImagePanel result={result} />}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Main component
// ─────────────────────────────────────────────────────────────────────────────
export default function ScopeIDEMain() {
  const [files, setFiles]       = useState(initialFiles);
  const [expanded, setExpanded] = useState(new Set(['examples', 'datasets', 'my-analysis']));
  const [openTabs, setOpenTabs] = useState([['examples', 'tutorial_01_observe.scope']]);
  const [activeTab, setActiveTab] = useState('examples/tutorial_01_observe.scope');
  const [dirty, setDirty]       = useState(new Set<string>());
  const [sidebar, setSidebar]   = useState(true);
  const [cursor, setCursor]     = useState({ ln: 1, col: 1 });
  const [activity, setActivity] = useState('files');

  const [logs, setLogs]           = useState<Array<{ level: string; message: string }>>([]);
  const [outputTab, setOutputTab] = useState('console');
  const [result, setResult]       = useState<ScopeResult | null>(null);
  const [running, setRunning]     = useState(false);
  const [hfResult, setHfResult]   = useState<HFResult | null>(null);

  const splitRef  = useRef<HTMLDivElement>(null);
  const dragging  = useRef(false);
  const [editorWidth, setEditorWidth] = useState(45);

  // ── Run ──────────────────────────────────────────────────────────────────
  const run = useCallback(async () => {
    const activePathArr = openTabs.find(t => t.join('/') === activeTab);
    if (!activePathArr) return;
    const activeNode = getNode(files, activePathArr);
    if (!activeNode || activeNode.type !== 'file' || activeNode.lang !== 'scope') return;

    setRunning(true);
    setResult(null);
    const newLogs: Array<{ level: string; message: string }> = [];
    const log = (msg: string, level = 'log') => newLogs.push({ level, message: msg });

    try {
      // 1. Compile
      log('Compiling SCOPE program…');
      const cr = compile(activeNode.content);

      if (!cr.ok || !cr.program) {
        log('❌ Compilation failed:', 'error');
        cr.errors.forEach(e => log(`  ${e.kind}: ${(e as any).message ?? JSON.stringify(e)}`, 'error'));
        setLogs([...newLogs]);
        setRunning(false);
        return;
      }
      log(`✓ Compiled: ${cr.program.name}  (${cr.program.morphisms.length} morphism${cr.program.morphisms.length!==1?'s':''})`);
      if (cr.warnings.length) cr.warnings.forEach(w => log(`  ⚠ ${w.kind}`, 'warn'));

      // 2. Fetch image
      const firstLoad = cr.program.morphisms
        .map(m => m.expr.observe.frame)
        .find(f => f.kind === 'LoadRef');

      let imagePayload: { data: Float32Array; width: number; height: number };

      if (firstLoad?.kind === 'LoadRef') {
        log(`Fetching ${firstLoad.dataset}/${firstLoad.image}…`);
        const params = new URLSearchParams({ db: firstLoad.db, dataset: firstLoad.dataset, image: firstLoad.image });
        const res = await fetch(`/api/image-proxy?${params}`);
        const json = await res.json();
        if (json.error) throw new Error(json.error);
        imagePayload = { data: new Float32Array(json.data as number[]), width: json.width as number, height: json.height as number };
        log(`✓ Image: ${json.width}×${json.height}px${json.synthetic ? ' (synthetic fallback)' : ''}`);
      } else {
        const W2 = 256, H2 = 256;
        const data = new Float32Array(W2 * H2);
        for (let y = 0; y < H2; y++) for (let x = 0; x < W2; x++) {
          const d1=(x-80)**2+(y-128)**2, d2=(x-176)**2+(y-128)**2;
          data[y*W2+x] = Math.max(0, Math.min(1, 0.05 + 0.9*Math.exp(-d1/1250) + 0.9*Math.exp(-d2/968)));
        }
        imagePayload = { data, width: W2, height: H2 };
        log('Using synthetic 256×256 image');
      }

      // 3. Run 5-phase pipeline
      log('Running SCOPE pipeline (COMPILE → ASSIGN → MEASURE → EXECUTE → EMIT)…');
      setLogs([...newLogs]);

      const scopeResult = await runScope(cr.program, imagePayload);

      // Append pipeline logs
      scopeResult.log.forEach(l => log(l));
      log('');
      log('═══ RESULT ═══');
      if (scopeResult.distance !== null) {
        log(`d = ${scopeResult.distance.toFixed(4)} µm  δd = ±${(scopeResult.uncertainty??0).toFixed(4)} µm  (${((scopeResult.relativeUncertainty??0)*100).toFixed(2)}%)`);
      }
      log(`SNR = ${scopeResult.snr.toFixed(2)}  CRLB = ${scopeResult.crlbPixels.toFixed(4)} px  C = ${scopeResult.channelCapacity.toFixed(3)} bits/px`);
      log(`S_k=${scopeResult.sEntropy.sk.toFixed(4)}  S_t=${scopeResult.sEntropy.st.toFixed(4)}  S_e=${scopeResult.sEntropy.se.toFixed(4)}  Σ=${scopeResult.sEntropy.sum.toFixed(12)}`);
      if (scopeResult.goalStatus.length) {
        log('Goals:');
        scopeResult.goalStatus.forEach(g => log(`  ${g.passed?'✓':'✗'} ${g.metric} ${g.op} ${g.threshold} ${g.unit}  (got ${g.actual.toPrecision(4)})`));
      }

      setResult(scopeResult);
      setLogs([...newLogs]);

      // 4. HuggingFace inference (non-blocking — enrich after render)
      log('Running HuggingFace inference (classify + segment + features)…');
      setLogs([...newLogs]);
      setHfResult(null);
      hfInfer(imagePayload.data, imagePayload.width, imagePayload.height)
        .then(hf => {
          setHfResult(hf);
          setLogs(prev => [
            ...prev,
            { level: 'info', message: `HF classification: ${hf.classification.map(c => `${c.label} ${(c.score*100).toFixed(1)}%`).join(', ') || 'no results'}` },
            ...(hf.segmentation ? [{ level: 'info', message: `HF segmentation: ${hf.segmentation.length} instance(s) detected` }] : []),
            ...(hf.features     ? [{ level: 'info', message: `HF DINOv2 features: ${hf.features.length} dims extracted` }]          : []),
          ]);
        })
        .catch(err => {
          setLogs(prev => [...prev, { level: 'warn', message: `HF inference skipped: ${err instanceof Error ? err.message : String(err)}` }]);
        });

      // Auto-switch to image tab when done
      setOutputTab('image');
    } catch (err) {
      log(`❌ ${err instanceof Error ? err.message : String(err)}`, 'error');
      setLogs([...newLogs]);
    } finally {
      setRunning(false);
    }
  }, [files, openTabs, activeTab]);

  // Ctrl+Enter shortcut
  useEffect(() => {
    const h = (e: KeyboardEvent) => { if ((e.ctrlKey||e.metaKey) && e.key==='Enter') { e.preventDefault(); run(); } };
    window.addEventListener('keydown', h);
    return () => window.removeEventListener('keydown', h);
  }, [run]);

  // Splitter drag
  useEffect(() => {
    const move = (e: MouseEvent) => {
      if (!dragging.current || !splitRef.current) return;
      const r = splitRef.current.getBoundingClientRect();
      setEditorWidth(Math.min(75, Math.max(20, ((e.clientX - r.left) / r.width) * 100)));
    };
    const up = () => { dragging.current = false; document.body.style.cursor = ''; };
    window.addEventListener('mousemove', move);
    window.addEventListener('mouseup', up);
    return () => { window.removeEventListener('mousemove', move); window.removeEventListener('mouseup', up); };
  }, []);

  const toggleFolder = useCallback((key: string) => {
    setExpanded(prev => { const n = new Set(prev); n.has(key) ? n.delete(key) : n.add(key); return n; });
  }, []);

  const openFile = useCallback((pathArr: string[]) => {
    const key = pathArr.join('/');
    setOpenTabs(prev => prev.some(t => t.join('/')===key) ? prev : [...prev, pathArr]);
    setActiveTab(key);
  }, []);

  const closeTab = useCallback((key: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setOpenTabs(prev => {
      const next = prev.filter(t => t.join('/')!==key);
      if (activeTab===key) setActiveTab(next.length ? next[next.length-1].join('/') : '');
      return next;
    });
    setDirty(prev => { const n = new Set(prev); n.delete(key); return n; });
  }, [activeTab]);

  const activePathArr = useMemo(() => openTabs.find(t => t.join('/')===activeTab) ?? null, [openTabs, activeTab]);
  const activeNode    = activePathArr ? getNode(files, activePathArr) : null;

  const updateContent = useCallback((val: string) => {
    if (!activePathArr) return;
    setFiles((prev: typeof initialFiles) => { const next = structuredClone(prev); getNode(next, activePathArr!).content = val; return next; });
    setDirty(prev => new Set(prev).add(activeTab));
  }, [activePathArr, activeTab]);

  const activities = [
    { id: 'files',  Icon: Files,     label: 'Explorer' },
    { id: 'search', Icon: Search,    label: 'Search' },
    { id: 'git',    Icon: GitBranch, label: 'Source Control' },
  ];

  return (
    <div className="flex h-screen w-screen flex-col overflow-hidden text-sm"
      style={{ background: T.editor, color: T.editorFg, border: `1px solid ${T.border}` }}>

      {/* Title bar */}
      <div className="flex h-9 shrink-0 items-center justify-between px-3" style={{ background: T.titlebar }}>
        <div className="flex items-center gap-2">
          <span className="h-3 w-3 rounded-full" style={{ background: '#ff5f56' }} />
          <span className="h-3 w-3 rounded-full" style={{ background: '#ffbd2e' }} />
          <span className="h-3 w-3 rounded-full" style={{ background: '#27c93f' }} />
        </div>
        <span className="text-xs" style={{ color: '#cccccc' }}>SCOPE Analysis Studio — Microscopy Measurement Framework</span>
        <div className="w-12" />
      </div>

      <div className="flex min-h-0 flex-1">
        {/* Activity bar */}
        <div className="flex w-12 shrink-0 flex-col items-center py-2" style={{ background: T.activitybar }}>
          {activities.map(({ id, Icon, label }) => {
            const active = activity === id;
            return (
              <button key={id} title={label}
                onClick={() => { if (active) setSidebar(s=>!s); else { setActivity(id); setSidebar(true); } }}
                className="relative flex h-11 w-12 items-center justify-center transition-colors"
                style={{ color: active ? T.activitybarFgActive : T.activitybarFg }}>
                {active && <span className="absolute left-0 top-1/2 h-6 w-0.5 -translate-y-1/2" style={{ background: '#fff' }} />}
                <Icon size={24} strokeWidth={1.5} />
              </button>
            );
          })}
        </div>

        {/* Sidebar */}
        {sidebar && (
          <div className="flex w-60 shrink-0 flex-col overflow-hidden"
            style={{ background: T.sidebar, borderRight: `1px solid ${T.border}` }}>
            <div className="flex h-9 shrink-0 items-center px-4 text-[11px] font-medium uppercase tracking-wider"
              style={{ color: T.sidebarHeader }}>
              {activities.find(a => a.id===activity)?.label}
            </div>
            <div className="min-h-0 flex-1 overflow-y-auto pb-2">
              {activity === 'files'
                ? <Tree tree={files} expanded={expanded} toggle={toggleFolder} activePath={activeTab} openFile={openFile} />
                : <div className="px-4 py-6 text-[13px]" style={{ color: T.tabFg }}>{activities.find(a=>a.id===activity)?.label} panel</div>
              }
            </div>
          </div>
        )}

        {/* Editor + Output split */}
        <div ref={splitRef} className="flex min-w-0 flex-1">
          {/* Editor */}
          <div className="flex min-w-0 flex-col" style={{ width: `${editorWidth}%` }}>
            {/* Tab bar */}
            <div className="flex h-9 shrink-0 items-stretch overflow-x-auto" style={{ background: T.tabInactive }}>
              {openTabs.map(pathArr => {
                const key = pathArr.join('/');
                const name = pathArr[pathArr.length-1];
                const active = key === activeTab;
                const isDirty = dirty.has(key);
                const { Icon, color } = fileIcon(name);
                return (
                  <div key={key} onClick={() => setActiveTab(key)}
                    className="group flex cursor-pointer items-center gap-2 border-r px-3 text-[13px]"
                    style={{ background: active ? T.tabActive : T.tabInactive, color: active ? T.tabFgActive : T.tabFg,
                      borderColor: T.border, borderTop: active ? `1px solid ${T.accentBright}` : '1px solid transparent' }}>
                    <Icon size={15} style={{ color }} />
                    <span className="whitespace-nowrap">{name}</span>
                    <button onClick={e => closeTab(key, e)}
                      className="flex h-5 w-5 items-center justify-center rounded"
                      style={{ color: active ? T.tabFgActive : T.tabFg }}>
                      {isDirty ? <Circle size={9} fill="currentColor" className="group-hover:hidden" /> : null}
                      <X size={15} className={isDirty ? 'hidden group-hover:block' : 'opacity-0 group-hover:opacity-100'} />
                    </button>
                  </div>
                );
              })}
            </div>
            {/* Breadcrumb */}
            {activePathArr && (
              <div className="flex h-6 shrink-0 items-center gap-1 px-4 text-[12px]"
                style={{ background: T.editor, color: T.tabFg }}>
                {activePathArr.map((p, i) => (
                  <span key={i} className="flex items-center gap-1">
                    {i > 0 && <ChevronRight size={12} className="opacity-60" />}{p}
                  </span>
                ))}
              </div>
            )}
            {activeNode?.lang === 'scope'
              ? <Editor value={activeNode.content} onChange={updateContent} onCursor={setCursor} />
              : <div className="flex min-h-0 flex-1 items-center justify-center text-sm"
                  style={{ background: T.editor, color: '#5a5a5a' }}>
                  {activeNode ? activeNode.content : 'Select a .scope file to start editing'}
                </div>
            }
          </div>

          {/* Splitter */}
          <div onMouseDown={() => { dragging.current = true; document.body.style.cursor = 'col-resize'; }}
            className="w-1 shrink-0 cursor-col-resize" style={{ background: T.border }} />

          {/* Output */}
          <OutputColumn logs={logs} onRun={run} onClear={() => setLogs([])}
            outputTab={outputTab} setOutputTab={setOutputTab} result={result} hfResult={hfResult} running={running} />
        </div>
      </div>

      {/* Status bar */}
      <div className="flex h-6 shrink-0 items-center justify-between px-3 text-[12px]"
        style={{ background: T.statusBar, color: T.statusFg }}>
        <span>Ln {cursor.ln}, Col {cursor.col}</span>
        <span>{activeNode?.lang === 'scope' ? 'SCOPE · Ctrl+Enter to run' : '—'}</span>
      </div>
    </div>
  );
}
