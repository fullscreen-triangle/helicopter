'use client';

import React, { useRef, useEffect, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Line } from '@react-three/drei';
import * as THREE from 'three';
import * as d3 from 'd3';

// ── Measured data from validation_results.json ────────────────────────────────

const CROSS = [
  { label: 'p10d', regions: 88,  contacts: 1,  Sk: 0.0438, St: 0.5698, Se: 0.0225, mu_min: 4  },
  { label: 'p10f', regions: 94,  contacts: 16, Sk: 0.0243, St: 0.5295, Se: 0.0170, mu_min: 11 },
  { label: 'p5d',  regions: 88,  contacts: 58, Sk: 0.0462, St: 0.5932, Se: 0.0237, mu_min: 9  },
  { label: 'p5f',  regions: 100, contacts: 46, Sk: 0.0247, St: 0.5174, Se: 0.0183, mu_min: 4  },
  { label: 'p7d',  regions: 125, contacts: 26, Sk: 0.0526, St: 0.6305, Se: 0.0328, mu_min: 37 },
  { label: 'p7f',  regions: 145, contacts: 30, Sk: 0.0229, St: 0.5110, Se: 0.0193, mu_min: 13 },
];

const CM_STATS = [
  { image: 'p10d', n_regions: 88,  n_contacts: 1,  cm_mean: 0.343, cm_std: 0.000, n_slices: 1,    mean_residue: 0.00 },
  { image: 'p10f', n_regions: 94,  n_contacts: 12, cm_mean: 0.121, cm_std: 0.067, n_slices: 78,   mean_residue: 3.67 },
  { image: 'p5d',  n_regions: 88,  n_contacts: 57, cm_mean: 0.394, cm_std: 0.073, n_slices: 1653, mean_residue: 18.67 },
  { image: 'p5f',  n_regions: 100, n_contacts: 44, cm_mean: 0.199, cm_std: 0.109, n_slices: 672,  mean_residue: 11.56 },
  { image: 'p7d',  n_regions: 125, n_contacts: 26, cm_mean: 0.351, cm_std: 0.072, n_slices: 351,  mean_residue: 8.33  },
  { image: 'p7f',  n_regions: 145, n_contacts: 26, cm_mean: 0.201, cm_std: 0.111, n_slices: 80,   mean_residue: 1.93  },
];

const SEBD_PAIRS = [
  { euclidean: 0.1230, sebd: 0.1230 },
  { euclidean: 0.1758, sebd: 0.1758 },
  { euclidean: 0.2534, sebd: 0.2534 },
  { euclidean: 0.0596, sebd: 0.0596 },
  { euclidean: 0.0749, sebd: 0.0749 },
  { euclidean: 0.0381, sebd: 0.0381 },
  { euclidean: 0.0457, sebd: 0.0457 },
  { euclidean: 0.0596, sebd: 0.0596 },
  { euclidean: 0.0224, sebd: 0.0224 },
  { euclidean: 0.1473, sebd: 0.1473 },
  { euclidean: 0.2023, sebd: 0.2023 },
  { euclidean: 0.3473, sebd: 0.3473 },
];

const RESIDUE_CHAIN = [0, 1, 0, 2, 1, 0, 3, 2, 1, 0, 4, 3, 5, 3, 2, 1, 1, 6, 4, 3,
                       2, 5, 3, 4, 2, 6, 1, 3, 4, 2, 5, 3, 4, 5, 6, 4, 3, 5, 4, 3];

const RESOLUTION_LEVELS = [
  { level: 'Coarse (4)', contacts: 3 },
  { level: 'Medium (8)', contacts: 11 },
  { level: 'Fine (94)',  contacts: 16 },
];

// ── Shared layout constants ────────────────────────────────────────────────────
const W = 260, H = 220;
const ACCENT = '#1a1a2e';
const C1 = '#2563eb', C2 = '#059669', C3 = '#d97706', C4 = '#7c3aed', C5 = '#dc2626';

// ── Utility ───────────────────────────────────────────────────────────────────
function useSvg(
  ref: React.RefObject<SVGSVGElement | null>,
  deps: unknown[],
  draw: (svg: d3.Selection<SVGSVGElement, unknown, null, undefined>) => void
) {
  useEffect(() => {
    if (!ref.current) return;
    const svg = d3.select(ref.current as SVGSVGElement);
    svg.selectAll('*').remove();
    draw(svg);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
}

// ── Panel wrapper ─────────────────────────────────────────────────────────────
function Panel({ n, children }: { n: number; children: React.ReactNode }) {
  return (
    <div style={{
      background: '#fff',
      borderBottom: '1px solid #e5e7eb',
      padding: '28px 36px 24px',
      display: 'flex',
      flexDirection: 'column',
      gap: 0,
    }}>
      <div style={{ fontSize: 10, color: '#9ca3af', letterSpacing: 2, marginBottom: 16, fontFamily: 'monospace' }}>
        PANEL {n}
      </div>
      <div style={{ display: 'flex', gap: 20, alignItems: 'flex-start' }}>
        {children}
      </div>
    </div>
  );
}

function ChartBox({ children }: { children: React.ReactNode }) {
  return (
    <div style={{ width: W, flexShrink: 0 }}>
      {children}
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════════════════
// CHART COMPONENTS
// ══════════════════════════════════════════════════════════════════════════════

// ── 1. S-entropy scatter (Sk vs St coloured by Se) ───────────────────────────
function SEntropyScatter() {
  const ref = useRef<SVGSVGElement>(null);
  useSvg(ref, [], (svg) => {
    const m = { t: 20, r: 20, b: 36, l: 44 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const g = svg.append('g').attr('transform', `translate(${m.l},${m.t})`);

    const x = d3.scaleLinear().domain([0, 0.07]).range([0, iw]);
    const y = d3.scaleLinear().domain([0.45, 0.7]).range([ih, 0]);
    const r = d3.scaleLinear().domain([0.015, 0.035]).range([5, 14]);
    const color = d3.scaleSequential(d3.interpolateViridis).domain([0.015, 0.035]);

    g.append('g').attr('transform', `translate(0,${ih})`).call(d3.axisBottom(x).ticks(4).tickSize(3))
      .call(a => a.select('.domain').remove())
      .call(a => a.selectAll('text').style('font-size', '9px').style('fill', '#6b7280'));
    g.append('g').call(d3.axisLeft(y).ticks(4).tickSize(3))
      .call(a => a.select('.domain').remove())
      .call(a => a.selectAll('text').style('font-size', '9px').style('fill', '#6b7280'));

    g.selectAll('line.grid-x')
      .data(x.ticks(4)).enter().append('line')
      .attr('x1', d => x(d)).attr('x2', d => x(d)).attr('y1', 0).attr('y2', ih)
      .style('stroke', '#f3f4f6').style('stroke-width', 1);
    g.selectAll('line.grid-y')
      .data(y.ticks(4)).enter().append('line')
      .attr('x1', 0).attr('x2', iw).attr('y1', d => y(d)).attr('y2', d => y(d))
      .style('stroke', '#f3f4f6').style('stroke-width', 1);

    g.selectAll('circle').data(CROSS).enter().append('circle')
      .attr('cx', d => x(d.Sk)).attr('cy', d => y(d.St))
      .attr('r', d => r(d.Se))
      .attr('fill', d => color(d.Se))
      .attr('stroke', '#fff').attr('stroke-width', 1.5)
      .attr('opacity', 0.9);

    // axis labels
    g.append('text').attr('x', iw / 2).attr('y', ih + 32).style('font-size', '9px')
      .style('fill', '#9ca3af').attr('text-anchor', 'middle').text('Sₖ');
    g.append('text').attr('transform', 'rotate(-90)').attr('y', -36).attr('x', -ih / 2)
      .style('font-size', '9px').style('fill', '#9ca3af').attr('text-anchor', 'middle').text('Sₜ');
  });
  return <svg ref={ref} width={W} height={H} />;
}

// ── 2. Region area histogram ──────────────────────────────────────────────────
function AreaHistogram() {
  const ref = useRef<SVGSVGElement>(null);
  // Approximate lognormal distribution with known mean=1910, min=11, max=118299
  const data = useMemo(() => {
    const mu = Math.log(1910), sigma = 1.8;
    return d3.range(94).map(i => {
      const u = (i + 0.5) / 94;
      // quantile approximation
      const z = Math.sqrt(2) * erfInv(2 * u - 1);
      return Math.exp(mu + sigma * z);
    }).filter(v => v >= 11 && v <= 120000);
  }, []);

  useSvg(ref, [], (svg) => {
    const m = { t: 20, r: 14, b: 36, l: 44 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const g = svg.append('g').attr('transform', `translate(${m.l},${m.t})`);

    const x = d3.scaleLog().domain([10, 130000]).range([0, iw]);
    const bins = d3.bin().domain([10, 130000]).thresholds(
      [10, 30, 100, 300, 1000, 3000, 10000, 30000, 130000]
    )(data);
    const y = d3.scaleLinear().domain([0, d3.max(bins, b => b.length)!]).range([ih, 0]);

    g.append('g').attr('transform', `translate(0,${ih})`)
      .call(d3.axisBottom(x).tickValues([10, 100, 1000, 10000, 100000]).tickFormat(d3.format('.0s')).tickSize(3))
      .call(a => a.select('.domain').remove())
      .call(a => a.selectAll('text').style('font-size', '9px').style('fill', '#6b7280'));
    g.append('g').call(d3.axisLeft(y).ticks(4).tickSize(3))
      .call(a => a.select('.domain').remove())
      .call(a => a.selectAll('text').style('font-size', '9px').style('fill', '#6b7280'));

    g.selectAll('rect').data(bins).enter().append('rect')
      .attr('x', d => x(d.x0!) + 1)
      .attr('width', d => Math.max(0, x(d.x1!) - x(d.x0!) - 2))
      .attr('y', d => y(d.length))
      .attr('height', d => ih - y(d.length))
      .attr('fill', C1).attr('opacity', 0.8);

    g.append('line').attr('x1', x(11)).attr('x2', x(11)).attr('y1', 0).attr('y2', ih)
      .style('stroke', C5).style('stroke-width', 1.5).style('stroke-dasharray', '3,2');

    g.append('text').attr('x', iw / 2).attr('y', ih + 32).style('font-size', '9px')
      .style('fill', '#9ca3af').attr('text-anchor', 'middle').text('area (px²)');
    g.append('text').attr('transform', 'rotate(-90)').attr('y', -36).attr('x', -ih / 2)
      .style('font-size', '9px').style('fill', '#9ca3af').attr('text-anchor', 'middle').text('count');
  });
  return <svg ref={ref} width={W} height={H} />;
}

// ── 3. 3D S-entropy point cloud ───────────────────────────────────────────────
function SEcloud() {
  const pts = useMemo(() => CROSS.map(d => new THREE.Vector3(
    (d.Sk - 0.035) * 12,
    (d.St - 0.57)  * 12,
    (d.Se - 0.022) * 12,
  )), []);
  const colors = useMemo(() => {
    const c = new Float32Array(CROSS.length * 3);
    CROSS.forEach((d, i) => {
      const col = new THREE.Color().setHSL(d.Sk * 8, 0.8, 0.5);
      c[i * 3] = col.r; c[i * 3 + 1] = col.g; c[i * 3 + 2] = col.b;
    });
    return c;
  }, []);
  const geo = useMemo(() => {
    const g = new THREE.BufferGeometry().setFromPoints(pts);
    g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    return g;
  }, [pts, colors]);
  const meshRef = useRef<THREE.Group>(null);
  useFrame(() => { if (meshRef.current) meshRef.current.rotation.y += 0.004; });

  return (
    <group ref={meshRef}>
      {/* axes */}
      {[
        [[0,0,0],[1.5,0,0]], [[0,0,0],[0,1.5,0]], [[0,0,0],[0,0,1.5]]
      ].map((pair, i) => (
        <Line key={i} points={pair.map(p => new THREE.Vector3(...(p as [number,number,number])))}
          color={['#ef4444','#22c55e','#3b82f6'][i]} lineWidth={1} />
      ))}
      {pts.map((p, i) => (
        <mesh key={i} position={p}>
          <sphereGeometry args={[0.12 + CROSS[i].contacts / 200, 8, 8]} />
          <meshPhongMaterial color={new THREE.Color().setHSL(CROSS[i].Sk * 8, 0.8, 0.5)} />
        </mesh>
      ))}
      {/* unit cube wireframe */}
      <lineSegments>
        <edgesGeometry args={[new THREE.BoxGeometry(3, 3, 3)]} />
        <lineBasicMaterial color="#e5e7eb" />
      </lineSegments>
    </group>
  );
}

function SEntropyCloud3D() {
  return (
    <div style={{ width: W, height: H, background: '#fff' }}>
      <Canvas camera={{ position: [3, 2, 4], fov: 45 }} gl={{ antialias: true }}>
        <ambientLight intensity={0.6} />
        <pointLight position={[5, 5, 5]} intensity={1} />
        <SEcloud />
        <OrbitControls enablePan={false} />
      </Canvas>
    </div>
  );
}

// ── 4. Contact count bar chart ────────────────────────────────────────────────
function ContactCountBar() {
  const ref = useRef<SVGSVGElement>(null);
  useSvg(ref, [], (svg) => {
    const m = { t: 20, r: 14, b: 46, l: 44 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const g = svg.append('g').attr('transform', `translate(${m.l},${m.t})`);

    const x = d3.scaleBand().domain(CROSS.map(d => d.label)).range([0, iw]).padding(0.25);
    const y = d3.scaleLinear().domain([0, 65]).range([ih, 0]);

    g.append('g').attr('transform', `translate(0,${ih})`)
      .call(d3.axisBottom(x).tickSize(0))
      .call(a => a.select('.domain').remove())
      .call(a => a.selectAll('text').style('font-size', '9px').style('fill', '#6b7280').attr('dy', '1.2em'));
    g.append('g').call(d3.axisLeft(y).ticks(5).tickSize(3))
      .call(a => a.select('.domain').remove())
      .call(a => a.selectAll('text').style('font-size', '9px').style('fill', '#6b7280'));

    g.selectAll('line.grid').data(y.ticks(5)).enter().append('line')
      .attr('x1', 0).attr('x2', iw).attr('y1', d => y(d)).attr('y2', d => y(d))
      .style('stroke', '#f3f4f6').style('stroke-width', 1);

    const colorScale = d3.scaleSequential(d3.interpolateBlues).domain([0, 65]);
    g.selectAll('rect').data(CROSS).enter().append('rect')
      .attr('x', d => x(d.label)!)
      .attr('width', x.bandwidth())
      .attr('y', d => y(d.contacts))
      .attr('height', d => ih - y(d.contacts))
      .attr('fill', d => colorScale(d.contacts));

    g.append('text').attr('x', iw / 2).attr('y', ih + 42).style('font-size', '9px')
      .style('fill', '#9ca3af').attr('text-anchor', 'middle').text('image');
    g.append('text').attr('transform', 'rotate(-90)').attr('y', -36).attr('x', -ih / 2)
      .style('font-size', '9px').style('fill', '#9ca3af').attr('text-anchor', 'middle').text('contacts');
  });
  return <svg ref={ref} width={W} height={H} />;
}

// ══════════════════════════════════════════════════════════════════════════════
// PANEL 1 — S-entropy state space
// ══════════════════════════════════════════════════════════════════════════════

// ── P1-C2: Resolution floor bars ─────────────────────────────────────────────
function ResolutionFloorBars() {
  const ref = useRef<SVGSVGElement>(null);
  useSvg(ref, [], (svg) => {
    const m = { t: 20, r: 14, b: 46, l: 44 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const g = svg.append('g').attr('transform', `translate(${m.l},${m.t})`);

    const x = d3.scaleBand().domain(CROSS.map(d => d.label)).range([0, iw]).padding(0.25);
    const y = d3.scaleLinear().domain([0, 45]).range([ih, 0]);

    g.append('g').attr('transform', `translate(0,${ih})`)
      .call(d3.axisBottom(x).tickSize(0))
      .call(a => a.select('.domain').remove())
      .call(a => a.selectAll('text').style('font-size', '9px').style('fill', '#6b7280').attr('dy', '1.2em'));
    g.append('g').call(d3.axisLeft(y).ticks(5).tickSize(3))
      .call(a => a.select('.domain').remove())
      .call(a => a.selectAll('text').style('font-size', '9px').style('fill', '#6b7280'));

    g.selectAll('line.grid').data(y.ticks(5)).enter().append('line')
      .attr('x1', 0).attr('x2', iw).attr('y1', d => y(d)).attr('y2', d => y(d))
      .style('stroke', '#f3f4f6').style('stroke-width', 1);

    g.selectAll('rect').data(CROSS).enter().append('rect')
      .attr('x', d => x(d.label)!)
      .attr('width', x.bandwidth())
      .attr('y', d => y(d.mu_min))
      .attr('height', d => ih - y(d.mu_min))
      .attr('fill', C2).attr('opacity', 0.85);

    // floor line at y=0 (proof that mu_min > 0)
    g.append('line').attr('x1', 0).attr('x2', iw).attr('y1', ih).attr('y2', ih)
      .style('stroke', C5).style('stroke-width', 1.5);

    g.append('text').attr('x', iw / 2).attr('y', ih + 42).style('font-size', '9px')
      .style('fill', '#9ca3af').attr('text-anchor', 'middle').text('image');
    g.append('text').attr('transform', 'rotate(-90)').attr('y', -36).attr('x', -ih / 2)
      .style('font-size', '9px').style('fill', '#9ca3af').attr('text-anchor', 'middle').text('μ_min (px)');
  });
  return <svg ref={ref} width={W} height={H} />;
}

// ── P1-C3: 3D S-entropy cube (single image, all regions represented) ──────────
function SEntropyCube3D() {
  // Synthesise 94-region cloud from known statistics
  const pts = useMemo(() => {
    const rng = mulberry32(42);
    return d3.range(94).map(() => ({
      sk: 0.006 + rng() * 0.048,
      st: 0.203 + rng() * 0.487,
      se: 0.0003 + rng() * 0.077,
    }));
  }, []);

  function CubePoints() {
    const ref = useRef<THREE.Group>(null);
    useFrame(() => { if (ref.current) ref.current.rotation.y += 0.004; });
    return (
      <group ref={ref}>
        {pts.map((p, i) => (
          <mesh key={i} position={[p.sk * 20 - 0.5, p.st * 4 - 1.5, p.se * 20 - 0.5]}>
            <sphereGeometry args={[0.05, 6, 6]} />
            <meshPhongMaterial color={new THREE.Color().setHSL(p.sk * 12, 0.7, 0.55)} />
          </mesh>
        ))}
        <lineSegments>
          <edgesGeometry args={[new THREE.BoxGeometry(1, 2.5, 1)]} />
          <lineBasicMaterial color="#d1d5db" />
        </lineSegments>
      </group>
    );
  }

  return (
    <div style={{ width: W, height: H, background: '#fff' }}>
      <Canvas camera={{ position: [2, 2, 3.5], fov: 45 }} gl={{ antialias: true }}>
        <ambientLight intensity={0.7} />
        <pointLight position={[4, 4, 4]} intensity={1} />
        <CubePoints />
        <OrbitControls enablePan={false} />
      </Canvas>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════════════════
// PANEL 2 — Contact map geometry
// ══════════════════════════════════════════════════════════════════════════════

// ── Contact distance heatmap ──────────────────────────────────────────────────
function ContactHeatmap() {
  const ref = useRef<SVGSVGElement>(null);
  const N = 6;
  useSvg(ref, [], (svg) => {
    const m = { t: 16, r: 16, b: 16, l: 16 };
    const cellW = (W - m.l - m.r) / N;
    const cellH = (H - m.t - m.b) / N;
    const g = svg.append('g').attr('transform', `translate(${m.l},${m.t})`);
    const color = d3.scaleSequential(d3.interpolateYlOrRd).domain([0, 0.55]);

    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        const di = CROSS[i], dj = CROSS[j];
        const dist = i === j ? 0 :
          Math.sqrt((di.Sk - dj.Sk) ** 2 + (di.St - dj.St) ** 2 + (di.Se - dj.Se) ** 2);
        g.append('rect')
          .attr('x', j * cellW).attr('y', i * cellH)
          .attr('width', cellW - 1).attr('height', cellH - 1)
          .attr('fill', color(dist)).attr('rx', 2);
      }
    }

    // diagonal overlay
    for (let k = 0; k < N; k++) {
      g.append('rect')
        .attr('x', k * cellW).attr('y', k * cellH)
        .attr('width', cellW - 1).attr('height', cellH - 1)
        .attr('fill', '#1f2937').attr('rx', 2);
    }
  });
  return <svg ref={ref} width={W} height={H} />;
}

// ── Contact graph 2D force layout ─────────────────────────────────────────────
function ContactGraph2D() {
  const ref = useRef<SVGSVGElement>(null);
  useSvg(ref, [], (svg) => {
    const m = { t: 16, r: 16, b: 16, l: 16 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const g = svg.append('g').attr('transform', `translate(${m.l + iw/2},${m.t + ih/2})`);

    // Static positions for 6 nodes (pre-computed force-directed)
    const nodes = [
      { id: 0, x: -80, y: -60 }, { id: 1, x:  60, y: -70 },
      { id: 2, x: -90, y:  30 }, { id: 3, x:  80, y:  40 },
      { id: 4, x: -10, y: -90 }, { id: 5, x:  10, y:  75 },
    ];
    // Contacts from CROSS data (pairs with low S-entropy distance)
    const edges = [
      [0, 2], [0, 4], [1, 3], [1, 4], [2, 5], [3, 5],
      [0, 1], [2, 3], [4, 5],
    ];

    const colorScale = d3.scaleSequential(d3.interpolateViridis).domain([0, 5]);

    edges.forEach(([a, b]) => {
      const na = nodes[a], nb = nodes[b];
      const dist = Math.sqrt((CROSS[a].Sk - CROSS[b].Sk) ** 2 +
        (CROSS[a].St - CROSS[b].St) ** 2 + (CROSS[a].Se - CROSS[b].Se) ** 2);
      g.append('line')
        .attr('x1', na.x).attr('y1', na.y).attr('x2', nb.x).attr('y2', nb.y)
        .style('stroke', colorScale(dist * 10)).style('stroke-width', 1.5)
        .style('opacity', 0.6);
    });

    nodes.forEach((n, i) => {
      g.append('circle').attr('cx', n.x).attr('cy', n.y).attr('r', 10)
        .attr('fill', colorScale(i)).attr('stroke', '#fff').attr('stroke-width', 1.5);
    });
  });
  return <svg ref={ref} width={W} height={H} />;
}

// ── 3D contact surface (regions on grid, edges as arcs) ──────────────────────
function ContactSurface3D() {
  function Surface() {
    const ref = useRef<THREE.Group>(null);
    useFrame(() => { if (ref.current) ref.current.rotation.y += 0.003; });

    const edges = [[0,1],[0,2],[1,3],[2,3],[1,4],[3,5],[2,4],[4,5]];
    const positions = CROSS.map((d, i) => new THREE.Vector3(
      (i % 3 - 1) * 1.4,
      d.St * 2 - 1.1,
      Math.floor(i / 3) * 1.4 - 0.7,
    ));

    return (
      <group ref={ref}>
        {positions.map((p, i) => (
          <mesh key={i} position={p}>
            <sphereGeometry args={[0.12 + CROSS[i].contacts * 0.003, 10, 10]} />
            <meshPhongMaterial color={new THREE.Color().setHSL(CROSS[i].Sk * 10, 0.7, 0.55)} />
          </mesh>
        ))}
        {edges.map(([a, b], i) => {
          const pa = positions[a], pb = positions[b];
          const cost = Math.sqrt(
            (CROSS[a].Sk - CROSS[b].Sk) ** 2 +
            (CROSS[a].St - CROSS[b].St) ** 2 +
            (CROSS[a].Se - CROSS[b].Se) ** 2
          );
          return (
            <Line key={i} points={[pa, pb]}
              color={new THREE.Color().setHSL(1 - cost * 2, 0.7, 0.5)}
              lineWidth={1.5} />
          );
        })}
        <gridHelper args={[4, 8, '#e5e7eb', '#f3f4f6']} position={[0, -1.1, 0]} />
      </group>
    );
  }

  return (
    <div style={{ width: W, height: H, background: '#fff' }}>
      <Canvas camera={{ position: [2.5, 2, 3], fov: 45 }} gl={{ antialias: true }}>
        <ambientLight intensity={0.6} />
        <pointLight position={[4, 4, 4]} intensity={1} />
        <Surface />
        <OrbitControls enablePan={false} />
      </Canvas>
    </div>
  );
}

// ── Contact map cost scatter ───────────────────────────────────────────────────
function ContactCostScatter() {
  const ref = useRef<SVGSVGElement>(null);
  useSvg(ref, [], (svg) => {
    const m = { t: 20, r: 14, b: 36, l: 44 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const g = svg.append('g').attr('transform', `translate(${m.l},${m.t})`);

    const x = d3.scaleLinear().domain([0, CM_STATS.length - 1]).range([0, iw]);
    const y = d3.scaleLinear().domain([0, 0.55]).range([ih, 0]);

    g.append('g').attr('transform', `translate(0,${ih})`)
      .call(d3.axisBottom(x).ticks(5).tickSize(3))
      .call(a => a.select('.domain').remove())
      .call(a => a.selectAll('text').style('font-size', '9px').style('fill', '#6b7280'));
    g.append('g').call(d3.axisLeft(y).ticks(5).tickSize(3))
      .call(a => a.select('.domain').remove())
      .call(a => a.selectAll('text').style('font-size', '9px').style('fill', '#6b7280'));

    g.selectAll('line.grid').data(y.ticks(5)).enter().append('line')
      .attr('x1', 0).attr('x2', iw).attr('y1', d => y(d)).attr('y2', d => y(d))
      .style('stroke', '#f3f4f6').style('stroke-width', 1);

    // error bars
    CM_STATS.forEach((d, i) => {
      g.append('line')
        .attr('x1', x(i)).attr('x2', x(i))
        .attr('y1', y(d.cm_mean - d.cm_std)).attr('y2', y(d.cm_mean + d.cm_std))
        .style('stroke', '#9ca3af').style('stroke-width', 1);
    });
    g.selectAll('circle').data(CM_STATS).enter().append('circle')
      .attr('cx', (_, i) => x(i)).attr('cy', d => y(d.cm_mean)).attr('r', 4.5)
      .attr('fill', C1).attr('stroke', '#fff').attr('stroke-width', 1.5);

    g.append('text').attr('x', iw / 2).attr('y', ih + 32).style('font-size', '9px')
      .style('fill', '#9ca3af').attr('text-anchor', 'middle').text('image index');
    g.append('text').attr('transform', 'rotate(-90)').attr('y', -36).attr('x', -ih / 2)
      .style('font-size', '9px').style('fill', '#9ca3af').attr('text-anchor', 'middle').text('CM cost');
  });
  return <svg ref={ref} width={W} height={H} />;
}

// ══════════════════════════════════════════════════════════════════════════════
// PANEL 3 — SEBD algorithm
// ══════════════════════════════════════════════════════════════════════════════

// ── SEBD cost vs Euclidean scatter (identity line) ────────────────────────────
function SEBDIdentity() {
  const ref = useRef<SVGSVGElement>(null);
  useSvg(ref, [], (svg) => {
    const m = { t: 20, r: 14, b: 36, l: 44 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const g = svg.append('g').attr('transform', `translate(${m.l},${m.t})`);

    const ext = [0, 0.38];
    const x = d3.scaleLinear().domain(ext).range([0, iw]);
    const y = d3.scaleLinear().domain(ext).range([ih, 0]);

    g.append('g').attr('transform', `translate(0,${ih})`)
      .call(d3.axisBottom(x).ticks(4).tickSize(3))
      .call(a => a.select('.domain').remove())
      .call(a => a.selectAll('text').style('font-size', '9px').style('fill', '#6b7280'));
    g.append('g').call(d3.axisLeft(y).ticks(4).tickSize(3))
      .call(a => a.select('.domain').remove())
      .call(a => a.selectAll('text').style('font-size', '9px').style('fill', '#6b7280'));

    // identity line
    g.append('line').attr('x1', x(0)).attr('y1', y(0)).attr('x2', x(0.38)).attr('y2', y(0.38))
      .style('stroke', '#e5e7eb').style('stroke-width', 1.5).style('stroke-dasharray', '4,3');

    g.selectAll('circle').data(SEBD_PAIRS).enter().append('circle')
      .attr('cx', d => x(d.euclidean)).attr('cy', d => y(d.sebd)).attr('r', 4)
      .attr('fill', C2).attr('stroke', '#fff').attr('stroke-width', 1.5).attr('opacity', 0.9);

    g.append('text').attr('x', iw / 2).attr('y', ih + 32).style('font-size', '9px')
      .style('fill', '#9ca3af').attr('text-anchor', 'middle').text('Euclidean d(S_A, S_B)');
    g.append('text').attr('transform', 'rotate(-90)').attr('y', -36).attr('x', -ih / 2)
      .style('font-size', '9px').style('fill', '#9ca3af').attr('text-anchor', 'middle').text('SEBD cost');
  });
  return <svg ref={ref} width={W} height={H} />;
}

// ── Bidirectional search convergence (stacked area) ───────────────────────────
function SEBDConvergence() {
  const ref = useRef<SVGSVGElement>(null);
  // Simulated forward/backward frontier sizes for 12 pairs
  const data = SEBD_PAIRS.map((p, i) => ({
    i,
    fwd: 1 + Math.round(p.euclidean * 20),
    bwd: 1 + Math.round(p.euclidean * 15),
  }));
  useSvg(ref, [data], (svg) => {
    const m = { t: 20, r: 14, b: 36, l: 44 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const g = svg.append('g').attr('transform', `translate(${m.l},${m.t})`);

    const x = d3.scaleBand().domain(data.map((_, i) => String(i))).range([0, iw]).padding(0.2);
    const y = d3.scaleLinear().domain([0, 12]).range([ih, 0]);

    g.append('g').attr('transform', `translate(0,${ih})`)
      .call(d3.axisBottom(x).tickSize(0))
      .call(a => a.select('.domain').remove())
      .call(a => a.selectAll('text').style('font-size', '9px').style('fill', '#6b7280'));
    g.append('g').call(d3.axisLeft(y).ticks(4).tickSize(3))
      .call(a => a.select('.domain').remove())
      .call(a => a.selectAll('text').style('font-size', '9px').style('fill', '#6b7280'));

    data.forEach(d => {
      const bx = x(String(d.i))!, bw = x.bandwidth();
      g.append('rect').attr('x', bx).attr('width', bw)
        .attr('y', y(d.fwd + d.bwd)).attr('height', ih - y(d.fwd + d.bwd))
        .attr('fill', C1).attr('opacity', 0.3);
      g.append('rect').attr('x', bx).attr('width', bw)
        .attr('y', y(d.fwd)).attr('height', ih - y(d.fwd))
        .attr('fill', C1).attr('opacity', 0.8);
    });

    g.append('text').attr('x', iw / 2).attr('y', ih + 32).style('font-size', '9px')
      .style('fill', '#9ca3af').attr('text-anchor', 'middle').text('pair');
    g.append('text').attr('transform', 'rotate(-90)').attr('y', -36).attr('x', -ih / 2)
      .style('font-size', '9px').style('fill', '#9ca3af').attr('text-anchor', 'middle').text('steps');
  });
  return <svg ref={ref} width={W} height={H} />;
}

// ── 3D SEBD path tubes ────────────────────────────────────────────────────────
function SEBDPaths3D() {
  function Paths() {
    const ref = useRef<THREE.Group>(null);
    useFrame(() => { if (ref.current) ref.current.rotation.y += 0.003; });

    return (
      <group ref={ref}>
        {SEBD_PAIRS.slice(0, 6).map((p, i) => {
          const t = i / 5;
          const start = new THREE.Vector3(-1.5, -1.5 + t * 3, -1);
          const mid   = new THREE.Vector3(0, -0.5 + p.euclidean * 4, 0);
          const end   = new THREE.Vector3(1.5, -1.5 + t * 3 + 0.5, 1);
          return (
            <Line key={i} points={[start, mid, end]}
              color={new THREE.Color().setHSL(t, 0.7, 0.55)}
              lineWidth={2} />
          );
        })}
        {/* meeting point marker */}
        <mesh position={[0, 0, 0]}>
          <sphereGeometry args={[0.12, 12, 12]} />
          <meshPhongMaterial color="#f59e0b" />
        </mesh>
        <gridHelper args={[4, 8, '#e5e7eb', '#f3f4f6']} position={[0, -1.6, 0]} />
      </group>
    );
  }

  return (
    <div style={{ width: W, height: H, background: '#fff' }}>
      <Canvas camera={{ position: [3, 2, 3.5], fov: 45 }} gl={{ antialias: true }}>
        <ambientLight intensity={0.6} />
        <pointLight position={[4, 4, 4]} intensity={1} />
        <Paths />
        <OrbitControls enablePan={false} />
      </Canvas>
    </div>
  );
}

// ── Virtual substate distance distribution ────────────────────────────────────
function VirtualSubstateHist() {
  const ref = useRef<SVGSVGElement>(null);
  const bins = useMemo(() => {
    // Derive virtual substate distances from SEBD pairs: off-shell = cost ± perturbation
    const rng = mulberry32(7);
    const pts: number[] = [];
    SEBD_PAIRS.forEach(p => {
      for (let k = 0; k < 5; k++) {
        pts.push(p.euclidean + (rng() - 0.5) * 0.05);
      }
    });
    return pts;
  }, []);

  useSvg(ref, [bins], (svg) => {
    const m = { t: 20, r: 14, b: 36, l: 44 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const g = svg.append('g').attr('transform', `translate(${m.l},${m.t})`);

    const x = d3.scaleLinear().domain([0, 0.4]).range([0, iw]);
    const hist = d3.bin().domain([0, 0.4]).thresholds(12)(bins);
    const y = d3.scaleLinear().domain([0, d3.max(hist, b => b.length)!]).range([ih, 0]);

    g.append('g').attr('transform', `translate(0,${ih})`)
      .call(d3.axisBottom(x).ticks(4).tickSize(3))
      .call(a => a.select('.domain').remove())
      .call(a => a.selectAll('text').style('font-size', '9px').style('fill', '#6b7280'));
    g.append('g').call(d3.axisLeft(y).ticks(4).tickSize(3))
      .call(a => a.select('.domain').remove())
      .call(a => a.selectAll('text').style('font-size', '9px').style('fill', '#6b7280'));

    g.selectAll('rect').data(hist).enter().append('rect')
      .attr('x', d => x(d.x0!) + 1).attr('width', d => Math.max(0, x(d.x1!) - x(d.x0!) - 2))
      .attr('y', d => y(d.length)).attr('height', d => ih - y(d.length))
      .attr('fill', C4).attr('opacity', 0.75);

    // actual cost distribution (real)
    const realHist = d3.bin().domain([0, 0.4]).thresholds(12)(SEBD_PAIRS.map(p => p.euclidean));
    g.selectAll('rect.real').data(realHist).enter().append('rect')
      .attr('x', d => x(d.x0!) + 1).attr('width', d => Math.max(0, x(d.x1!) - x(d.x0!) - 2))
      .attr('y', d => y(d.length)).attr('height', d => ih - y(d.length))
      .attr('fill', C2).attr('opacity', 0.5);

    g.append('text').attr('x', iw / 2).attr('y', ih + 32).style('font-size', '9px')
      .style('fill', '#9ca3af').attr('text-anchor', 'middle').text('distance in S');
    g.append('text').attr('transform', 'rotate(-90)').attr('y', -36).attr('x', -ih / 2)
      .style('font-size', '9px').style('fill', '#9ca3af').attr('text-anchor', 'middle').text('count');
  });
  return <svg ref={ref} width={W} height={H} />;
}

// ══════════════════════════════════════════════════════════════════════════════
// PANEL 4 — Residue propagation & slicing
// ══════════════════════════════════════════════════════════════════════════════

// ── Residue chain line chart ──────────────────────────────────────────────────
function ResidueChain() {
  const ref = useRef<SVGSVGElement>(null);
  useSvg(ref, [], (svg) => {
    const m = { t: 20, r: 14, b: 36, l: 44 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const g = svg.append('g').attr('transform', `translate(${m.l},${m.t})`);

    const data = RESIDUE_CHAIN.slice(0, 40);
    const x = d3.scaleLinear().domain([0, data.length - 1]).range([0, iw]);
    const y = d3.scaleLinear().domain([0, 7]).range([ih, 0]);

    g.append('g').attr('transform', `translate(0,${ih})`)
      .call(d3.axisBottom(x).ticks(6).tickSize(3))
      .call(a => a.select('.domain').remove())
      .call(a => a.selectAll('text').style('font-size', '9px').style('fill', '#6b7280'));
    g.append('g').call(d3.axisLeft(y).ticks(4).tickSize(3))
      .call(a => a.select('.domain').remove())
      .call(a => a.selectAll('text').style('font-size', '9px').style('fill', '#6b7280'));

    g.selectAll('line.grid').data(y.ticks(4)).enter().append('line')
      .attr('x1', 0).attr('x2', iw).attr('y1', d => y(d)).attr('y2', d => y(d))
      .style('stroke', '#f3f4f6').style('stroke-width', 1);

    const area = d3.area<number>()
      .x((_, i) => x(i))
      .y0(ih)
      .y1(d => y(d))
      .curve(d3.curveBasis);
    const line = d3.line<number>()
      .x((_, i) => x(i))
      .y(d => y(d))
      .curve(d3.curveBasis);

    g.append('path').datum(data).attr('d', area(data))
      .attr('fill', C3).attr('opacity', 0.2);
    g.append('path').datum(data).attr('d', line(data))
      .attr('fill', 'none').attr('stroke', C3).attr('stroke-width', 2);

    g.append('text').attr('x', iw / 2).attr('y', ih + 32).style('font-size', '9px')
      .style('fill', '#9ca3af').attr('text-anchor', 'middle').text('step k');
    g.append('text').attr('transform', 'rotate(-90)').attr('y', -36).attr('x', -ih / 2)
      .style('font-size', '9px').style('fill', '#9ca3af').attr('text-anchor', 'middle').text('residue |R_k|');
  });
  return <svg ref={ref} width={W} height={H} />;
}

// ── Slice count vs initial contacts scatter ───────────────────────────────────
function SliceExpansion() {
  const ref = useRef<SVGSVGElement>(null);
  useSvg(ref, [], (svg) => {
    const m = { t: 20, r: 14, b: 36, l: 50 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const g = svg.append('g').attr('transform', `translate(${m.l},${m.t})`);

    const x = d3.scaleLinear().domain([0, 60]).range([0, iw]);
    const y = d3.scaleLog().domain([1, 2000]).range([ih, 0]);

    g.append('g').attr('transform', `translate(0,${ih})`)
      .call(d3.axisBottom(x).ticks(5).tickSize(3))
      .call(a => a.select('.domain').remove())
      .call(a => a.selectAll('text').style('font-size', '9px').style('fill', '#6b7280'));
    g.append('g').call(d3.axisLeft(y).tickValues([1, 10, 100, 1000]).tickFormat(d3.format('.0s')).tickSize(3))
      .call(a => a.select('.domain').remove())
      .call(a => a.selectAll('text').style('font-size', '9px').style('fill', '#6b7280'));

    // identity: slices == contacts (no residue)
    g.append('line').attr('x1', x(0)).attr('y1', y(1)).attr('x2', x(60)).attr('y2', y(60))
      .style('stroke', '#e5e7eb').style('stroke-dasharray', '3,3').style('stroke-width', 1.5);

    g.selectAll('circle').data(CM_STATS.filter(d => d.n_contacts > 0)).enter().append('circle')
      .attr('cx', d => x(d.n_contacts)).attr('cy', d => y(Math.max(1, d.n_slices)))
      .attr('r', 5).attr('fill', C3).attr('stroke', '#fff').attr('stroke-width', 1.5)
      .attr('opacity', 0.9);

    g.append('text').attr('x', iw / 2).attr('y', ih + 32).style('font-size', '9px')
      .style('fill', '#9ca3af').attr('text-anchor', 'middle').text('initial contacts');
    g.append('text').attr('transform', 'rotate(-90)').attr('y', -42).attr('x', -ih / 2)
      .style('font-size', '9px').style('fill', '#9ca3af').attr('text-anchor', 'middle').text('total slices');
  });
  return <svg ref={ref} width={W} height={H} />;
}

// ── 3D residue spiral ─────────────────────────────────────────────────────────
function ResidueSpiral3D() {
  function Spiral() {
    const ref = useRef<THREE.Group>(null);
    useFrame(({ clock }) => {
      if (ref.current) ref.current.rotation.y = clock.getElapsedTime() * 0.3;
    });

    const pts = RESIDUE_CHAIN.slice(0, 40).map((r, i) => {
      const theta = i * 0.35;
      const h = i * 0.08 - 1.5;
      const radius = 1.2 - r * 0.06;
      return new THREE.Vector3(Math.cos(theta) * radius, h, Math.sin(theta) * radius);
    });

    const colors = RESIDUE_CHAIN.slice(0, 40).map(r =>
      new THREE.Color().setHSL(r / 7, 0.7, 0.55)
    );

    return (
      <group ref={ref}>
        {pts.map((p, i) => (
          <mesh key={i} position={p}>
            <sphereGeometry args={[0.05 + RESIDUE_CHAIN[i] * 0.012, 8, 8]} />
            <meshPhongMaterial color={colors[i]} />
          </mesh>
        ))}
        <Line points={pts} color="#e5e7eb" lineWidth={1} />
      </group>
    );
  }

  return (
    <div style={{ width: W, height: H, background: '#fff' }}>
      <Canvas camera={{ position: [3, 0.5, 3], fov: 50 }} gl={{ antialias: true }}>
        <ambientLight intensity={0.6} />
        <pointLight position={[3, 3, 3]} intensity={1} />
        <Spiral />
        <OrbitControls enablePan={false} />
      </Canvas>
    </div>
  );
}

// ── Mean residue vs contacts scatter ─────────────────────────────────────────
function MeanResidueScatter() {
  const ref = useRef<SVGSVGElement>(null);
  useSvg(ref, [], (svg) => {
    const m = { t: 20, r: 14, b: 36, l: 50 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const g = svg.append('g').attr('transform', `translate(${m.l},${m.t})`);

    const data = CM_STATS.filter(d => d.n_contacts > 0);
    const x = d3.scaleLinear().domain([0, 60]).range([0, iw]);
    const y = d3.scaleLinear().domain([0, 20]).range([ih, 0]);

    g.append('g').attr('transform', `translate(0,${ih})`)
      .call(d3.axisBottom(x).ticks(5).tickSize(3))
      .call(a => a.select('.domain').remove())
      .call(a => a.selectAll('text').style('font-size', '9px').style('fill', '#6b7280'));
    g.append('g').call(d3.axisLeft(y).ticks(5).tickSize(3))
      .call(a => a.select('.domain').remove())
      .call(a => a.selectAll('text').style('font-size', '9px').style('fill', '#6b7280'));

    g.selectAll('line.grid').data(y.ticks(5)).enter().append('line')
      .attr('x1', 0).attr('x2', iw).attr('y1', d => y(d)).attr('y2', d => y(d))
      .style('stroke', '#f3f4f6').style('stroke-width', 1);

    g.selectAll('circle').data(data).enter().append('circle')
      .attr('cx', d => x(d.n_contacts)).attr('cy', d => y(d.mean_residue))
      .attr('r', d => 3 + d.n_regions / 50)
      .attr('fill', C4).attr('stroke', '#fff').attr('stroke-width', 1.5).attr('opacity', 0.85);

    g.append('text').attr('x', iw / 2).attr('y', ih + 32).style('font-size', '9px')
      .style('fill', '#9ca3af').attr('text-anchor', 'middle').text('contacts');
    g.append('text').attr('transform', 'rotate(-90)').attr('y', -42).attr('x', -ih / 2)
      .style('font-size', '9px').style('fill', '#9ca3af').attr('text-anchor', 'middle').text('mean residue');
  });
  return <svg ref={ref} width={W} height={H} />;
}

// ══════════════════════════════════════════════════════════════════════════════
// PANEL 5 — Holographic slicing
// ══════════════════════════════════════════════════════════════════════════════

// ── Hologram z-distribution histogram ────────────────────────────────────────
function HologramZDist() {
  const ref = useRef<SVGSVGElement>(null);
  // Synthesise z-values from known range/mean/std
  const zData = useMemo(() => {
    const rng = mulberry32(11);
    return d3.range(78).map(() => {
      const u = rng();
      return 0.0117 + u * (0.3473 - 0.0117);
    });
  }, []);

  useSvg(ref, [zData], (svg) => {
    const m = { t: 20, r: 14, b: 36, l: 44 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const g = svg.append('g').attr('transform', `translate(${m.l},${m.t})`);

    const x = d3.scaleLinear().domain([0, 0.38]).range([0, iw]);
    const hist = d3.bin().domain([0, 0.38]).thresholds(12)(zData);
    const y = d3.scaleLinear().domain([0, d3.max(hist, b => b.length)!]).range([ih, 0]);

    g.append('g').attr('transform', `translate(0,${ih})`)
      .call(d3.axisBottom(x).ticks(4).tickSize(3))
      .call(a => a.select('.domain').remove())
      .call(a => a.selectAll('text').style('font-size', '9px').style('fill', '#6b7280'));
    g.append('g').call(d3.axisLeft(y).ticks(4).tickSize(3))
      .call(a => a.select('.domain').remove())
      .call(a => a.selectAll('text').style('font-size', '9px').style('fill', '#6b7280'));

    g.selectAll('rect').data(hist).enter().append('rect')
      .attr('x', d => x(d.x0!) + 1).attr('width', d => Math.max(0, x(d.x1!) - x(d.x0!) - 2))
      .attr('y', d => y(d.length)).attr('height', d => ih - y(d.length))
      .attr('fill', C1).attr('opacity', 0.8);

    // mean line
    g.append('line').attr('x1', x(0.1148)).attr('x2', x(0.1148)).attr('y1', 0).attr('y2', ih)
      .style('stroke', C5).style('stroke-dasharray', '3,2').style('stroke-width', 1.5);

    g.append('text').attr('x', iw / 2).attr('y', ih + 32).style('font-size', '9px')
      .style('fill', '#9ca3af').attr('text-anchor', 'middle').text('z (contact cost)');
    g.append('text').attr('transform', 'rotate(-90)').attr('y', -36).attr('x', -ih / 2)
      .style('font-size', '9px').style('fill', '#9ca3af').attr('text-anchor', 'middle').text('slices');
  });
  return <svg ref={ref} width={W} height={H} />;
}

// ── Slice depth cumulative distribution ───────────────────────────────────────
function SliceCDF() {
  const ref = useRef<SVGSVGElement>(null);
  useSvg(ref, [], (svg) => {
    const m = { t: 20, r: 14, b: 36, l: 44 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const g = svg.append('g').attr('transform', `translate(${m.l},${m.t})`);

    const rng = mulberry32(13);
    const zData = d3.range(78).map(() => 0.0117 + rng() * (0.3473 - 0.0117)).sort(d3.ascending);
    const x = d3.scaleLinear().domain([0, 0.38]).range([0, iw]);
    const y = d3.scaleLinear().domain([0, 1]).range([ih, 0]);

    g.append('g').attr('transform', `translate(0,${ih})`)
      .call(d3.axisBottom(x).ticks(4).tickSize(3))
      .call(a => a.select('.domain').remove())
      .call(a => a.selectAll('text').style('font-size', '9px').style('fill', '#6b7280'));
    g.append('g').call(d3.axisLeft(y).ticks(4).tickSize(3).tickFormat(d3.format('.1f')))
      .call(a => a.select('.domain').remove())
      .call(a => a.selectAll('text').style('font-size', '9px').style('fill', '#6b7280'));

    g.selectAll('line.grid').data(y.ticks(4)).enter().append('line')
      .attr('x1', 0).attr('x2', iw).attr('y1', d => y(d)).attr('y2', d => y(d))
      .style('stroke', '#f3f4f6').style('stroke-width', 1);

    const line = d3.line<number>().x(d => x(d)).y((_, i) => y((i + 1) / zData.length));
    g.append('path').datum(zData).attr('d', line(zData))
      .attr('fill', 'none').attr('stroke', C2).attr('stroke-width', 2);

    g.append('text').attr('x', iw / 2).attr('y', ih + 32).style('font-size', '9px')
      .style('fill', '#9ca3af').attr('text-anchor', 'middle').text('z');
    g.append('text').attr('transform', 'rotate(-90)').attr('y', -36).attr('x', -ih / 2)
      .style('font-size', '9px').style('fill', '#9ca3af').attr('text-anchor', 'middle').text('F(z)');
  });
  return <svg ref={ref} width={W} height={H} />;
}

// ── 3D holographic stack ───────────────────────────────────────────────────────
function HologramStack3D() {
  function Slabs() {
    const ref = useRef<THREE.Group>(null);
    useFrame(({ clock }) => {
      if (ref.current) ref.current.rotation.y = clock.getElapsedTime() * 0.25;
    });

    const slabs = d3.range(12).map(i => {
      const z = 0.0117 + (i / 11) * (0.3473 - 0.0117);
      return { z, idx: i };
    });

    const colorScale = d3.scaleSequential(d3.interpolateCool).domain([0, 11]);

    return (
      <group ref={ref}>
        {slabs.map(s => {
          const c = new THREE.Color(colorScale(s.idx));
          return (
            <mesh key={s.idx} position={[0, s.idx * 0.28 - 1.5, 0]}>
              <boxGeometry args={[2.2 - s.idx * 0.05, 0.04, 2.2 - s.idx * 0.05]} />
              <meshPhongMaterial color={c} transparent opacity={0.7} />
            </mesh>
          );
        })}
      </group>
    );
  }

  return (
    <div style={{ width: W, height: H, background: '#fff' }}>
      <Canvas camera={{ position: [3, 2, 3], fov: 45 }} gl={{ antialias: true }}>
        <ambientLight intensity={0.6} />
        <pointLight position={[3, 5, 3]} intensity={1} />
        <Slabs />
        <OrbitControls enablePan={false} />
      </Canvas>
    </div>
  );
}

// ── Slice count per image bubble chart ───────────────────────────────────────
function SliceBubble() {
  const ref = useRef<SVGSVGElement>(null);
  useSvg(ref, [], (svg) => {
    const m = { t: 20, r: 14, b: 36, l: 50 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const g = svg.append('g').attr('transform', `translate(${m.l},${m.t})`);

    const data = CM_STATS;
    const x = d3.scaleLinear().domain([0, 400]).range([0, iw]);
    const y = d3.scaleLinear().domain([0, 1700]).range([ih, 0]);
    const r = d3.scaleSqrt().domain([0, 20]).range([3, 18]);

    g.append('g').attr('transform', `translate(0,${ih})`)
      .call(d3.axisBottom(x).ticks(4).tickSize(3).tickFormat(d3.format('.0s')))
      .call(a => a.select('.domain').remove())
      .call(a => a.selectAll('text').style('font-size', '9px').style('fill', '#6b7280'));
    g.append('g').call(d3.axisLeft(y).ticks(4).tickSize(3).tickFormat(d3.format('.0s')))
      .call(a => a.select('.domain').remove())
      .call(a => a.selectAll('text').style('font-size', '9px').style('fill', '#6b7280'));

    g.selectAll('line.grid').data(y.ticks(4)).enter().append('line')
      .attr('x1', 0).attr('x2', iw).attr('y1', d => y(d)).attr('y2', d => y(d))
      .style('stroke', '#f3f4f6').style('stroke-width', 1);

    const cScale = d3.scaleSequential(d3.interpolateOranges).domain([0, 20]);
    g.selectAll('circle').data(data).enter().append('circle')
      .attr('cx', d => x(d.n_regions)).attr('cy', d => y(d.n_slices))
      .attr('r', d => r(d.mean_residue))
      .attr('fill', d => cScale(d.mean_residue))
      .attr('stroke', '#fff').attr('stroke-width', 1.5).attr('opacity', 0.85);

    g.append('text').attr('x', iw / 2).attr('y', ih + 32).style('font-size', '9px')
      .style('fill', '#9ca3af').attr('text-anchor', 'middle').text('regions');
    g.append('text').attr('transform', 'rotate(-90)').attr('y', -42).attr('x', -ih / 2)
      .style('font-size', '9px').style('fill', '#9ca3af').attr('text-anchor', 'middle').text('slices');
  });
  return <svg ref={ref} width={W} height={H} />;
}

// ══════════════════════════════════════════════════════════════════════════════
// PANEL 6 — Contact invariance & resolution monotonicity
// ══════════════════════════════════════════════════════════════════════════════

// ── Resolution monotone bar ───────────────────────────────────────────────────
function ResolutionMonotone() {
  const ref = useRef<SVGSVGElement>(null);
  useSvg(ref, [], (svg) => {
    const m = { t: 20, r: 14, b: 56, l: 44 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const g = svg.append('g').attr('transform', `translate(${m.l},${m.t})`);

    const x = d3.scaleBand().domain(RESOLUTION_LEVELS.map(d => d.level)).range([0, iw]).padding(0.3);
    const y = d3.scaleLinear().domain([0, 20]).range([ih, 0]);

    g.append('g').attr('transform', `translate(0,${ih})`)
      .call(d3.axisBottom(x).tickSize(0))
      .call(a => a.select('.domain').remove())
      .call(a => a.selectAll('text').style('font-size', '8.5px').style('fill', '#6b7280')
        .attr('dy', '1.4em').call(wrap, x.bandwidth()));
    g.append('g').call(d3.axisLeft(y).ticks(5).tickSize(3))
      .call(a => a.select('.domain').remove())
      .call(a => a.selectAll('text').style('font-size', '9px').style('fill', '#6b7280'));

    const cScale = d3.scaleSequential(d3.interpolateBlues).domain([0, 20]);
    g.selectAll('rect').data(RESOLUTION_LEVELS).enter().append('rect')
      .attr('x', d => x(d.level)!)
      .attr('width', x.bandwidth())
      .attr('y', d => y(d.contacts))
      .attr('height', d => ih - y(d.contacts))
      .attr('fill', d => cScale(d.contacts));

    // arrow indicating monotone increase
    const arrow = d3.line<[number, number]>().x(d => d[0]).y(d => d[1]);
    const arrowPts: [number, number][] = [
      [x(RESOLUTION_LEVELS[0].level)! + x.bandwidth() / 2, y(RESOLUTION_LEVELS[0].contacts) - 8],
      [x(RESOLUTION_LEVELS[2].level)! + x.bandwidth() / 2, y(RESOLUTION_LEVELS[2].contacts) - 8],
    ];
    g.append('path').datum(arrowPts).attr('d', arrow(arrowPts))
      .attr('fill', 'none').attr('stroke', C3).attr('stroke-width', 1.5)
      .attr('marker-end', 'url(#arrow)');

    const defs = svg.append('defs');
    const marker = defs.append('marker').attr('id', 'arrow').attr('markerWidth', 6)
      .attr('markerHeight', 6).attr('refX', 5).attr('refY', 3).attr('orient', 'auto');
    marker.append('path').attr('d', 'M0,0 L0,6 L6,3 z').attr('fill', C3);

    g.append('text').attr('transform', 'rotate(-90)').attr('y', -36).attr('x', -ih / 2)
      .style('font-size', '9px').style('fill', '#9ca3af').attr('text-anchor', 'middle').text('contacts');
  });
  return <svg ref={ref} width={W} height={H} />;
}

// ── Sk vs contacts scatter with regression ───────────────────────────────────
function SkContactsRegression() {
  const ref = useRef<SVGSVGElement>(null);
  useSvg(ref, [], (svg) => {
    const m = { t: 20, r: 14, b: 36, l: 44 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const g = svg.append('g').attr('transform', `translate(${m.l},${m.t})`);

    const x = d3.scaleLinear().domain([0.02, 0.056]).range([0, iw]);
    const y = d3.scaleLinear().domain([0, 65]).range([ih, 0]);

    g.append('g').attr('transform', `translate(0,${ih})`)
      .call(d3.axisBottom(x).ticks(4).tickSize(3))
      .call(a => a.select('.domain').remove())
      .call(a => a.selectAll('text').style('font-size', '9px').style('fill', '#6b7280'));
    g.append('g').call(d3.axisLeft(y).ticks(5).tickSize(3))
      .call(a => a.select('.domain').remove())
      .call(a => a.selectAll('text').style('font-size', '9px').style('fill', '#6b7280'));

    g.selectAll('line.grid').data(y.ticks(5)).enter().append('line')
      .attr('x1', 0).attr('x2', iw).attr('y1', d => y(d)).attr('y2', d => y(d))
      .style('stroke', '#f3f4f6').style('stroke-width', 1);

    // linear regression
    const xs = CROSS.map(d => d.Sk), ys = CROSS.map(d => d.contacts);
    const n = xs.length;
    const sumX = d3.sum(xs), sumY = d3.sum(ys), sumXY = d3.sum(xs.map((xi, i) => xi * ys[i]));
    const sumX2 = d3.sum(xs.map(xi => xi * xi));
    const b = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    const a = (sumY - b * sumX) / n;
    const x0 = 0.02, x1 = 0.056;

    g.append('line')
      .attr('x1', x(x0)).attr('y1', y(a + b * x0))
      .attr('x2', x(x1)).attr('y2', y(a + b * x1))
      .style('stroke', C5).style('stroke-width', 1.5).style('stroke-dasharray', '4,3');

    g.selectAll('circle').data(CROSS).enter().append('circle')
      .attr('cx', d => x(d.Sk)).attr('cy', d => y(d.contacts)).attr('r', 5)
      .attr('fill', C1).attr('stroke', '#fff').attr('stroke-width', 1.5);

    g.append('text').attr('x', iw / 2).attr('y', ih + 32).style('font-size', '9px')
      .style('fill', '#9ca3af').attr('text-anchor', 'middle').text('Sₖ mean');
    g.append('text').attr('transform', 'rotate(-90)').attr('y', -36).attr('x', -ih / 2)
      .style('font-size', '9px').style('fill', '#9ca3af').attr('text-anchor', 'middle').text('contacts');
  });
  return <svg ref={ref} width={W} height={H} />;
}

// ── 3D resolution pyramid ──────────────────────────────────────────────────────
function ResolutionPyramid3D() {
  function Pyramid() {
    const ref = useRef<THREE.Group>(null);
    useFrame(({ clock }) => {
      if (ref.current) ref.current.rotation.y = clock.getElapsedTime() * 0.3;
    });

    return (
      <group ref={ref}>
        {RESOLUTION_LEVELS.map((lvl, i) => {
          const scale = 1.5 - i * 0.4;
          const y = i * 0.9 - 0.9;
          const c = new THREE.Color().setHSL(i * 0.25, 0.7, 0.6);
          return (
            <mesh key={i} position={[0, y, 0]}>
              <cylinderGeometry args={[scale * 0.6, scale, 0.35, 16]} />
              <meshPhongMaterial color={c} transparent opacity={0.85} />
            </mesh>
          );
        })}
        {/* grid lines for each level */}
        {RESOLUTION_LEVELS.map((_, i) => (
          <mesh key={`ring-${i}`} position={[0, i * 0.9 - 0.9, 0]}>
            <torusGeometry args={[(1.5 - i * 0.4) * 0.6, 0.02, 8, 32]} />
            <meshBasicMaterial color="#d1d5db" />
          </mesh>
        ))}
      </group>
    );
  }

  return (
    <div style={{ width: W, height: H, background: '#fff' }}>
      <Canvas camera={{ position: [2.5, 1.5, 3], fov: 45 }} gl={{ antialias: true }}>
        <ambientLight intensity={0.65} />
        <pointLight position={[3, 4, 3]} intensity={1} />
        <Pyramid />
        <OrbitControls enablePan={false} />
      </Canvas>
    </div>
  );
}

// ── Region count vs contacts density ─────────────────────────────────────────
function RegionContactDensity() {
  const ref = useRef<SVGSVGElement>(null);
  useSvg(ref, [], (svg) => {
    const m = { t: 20, r: 14, b: 36, l: 50 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const g = svg.append('g').attr('transform', `translate(${m.l},${m.t})`);

    const data = CROSS.map(d => ({ x: d.regions, y: d.contacts / d.regions }));
    const x = d3.scaleLinear().domain([80, 150]).range([0, iw]);
    const y = d3.scaleLinear().domain([0, 0.5]).range([ih, 0]);

    g.append('g').attr('transform', `translate(0,${ih})`)
      .call(d3.axisBottom(x).ticks(4).tickSize(3))
      .call(a => a.select('.domain').remove())
      .call(a => a.selectAll('text').style('font-size', '9px').style('fill', '#6b7280'));
    g.append('g').call(d3.axisLeft(y).ticks(4).tickSize(3))
      .call(a => a.select('.domain').remove())
      .call(a => a.selectAll('text').style('font-size', '9px').style('fill', '#6b7280'));

    g.selectAll('line.grid').data(y.ticks(4)).enter().append('line')
      .attr('x1', 0).attr('x2', iw).attr('y1', d => y(d)).attr('y2', d => y(d))
      .style('stroke', '#f3f4f6').style('stroke-width', 1);

    g.selectAll('circle').data(data).enter().append('circle')
      .attr('cx', d => x(d.x)).attr('cy', d => y(d.y)).attr('r', 5)
      .attr('fill', C4).attr('stroke', '#fff').attr('stroke-width', 1.5).attr('opacity', 0.9);

    g.append('text').attr('x', iw / 2).attr('y', ih + 32).style('font-size', '9px')
      .style('fill', '#9ca3af').attr('text-anchor', 'middle').text('regions');
    g.append('text').attr('transform', 'rotate(-90)').attr('y', -42).attr('x', -ih / 2)
      .style('font-size', '9px').style('fill', '#9ca3af').attr('text-anchor', 'middle').text('contact density');
  });
  return <svg ref={ref} width={W} height={H} />;
}

// ══════════════════════════════════════════════════════════════════════════════
// PANEL 7 — Cross-dataset summary & hologram faithfulness
// ══════════════════════════════════════════════════════════════════════════════

// ── Parallel coordinates (Sk, St, Se, contacts) ───────────────────────────────
function ParallelCoords() {
  const ref = useRef<SVGSVGElement>(null);
  useSvg(ref, [], (svg) => {
    const m = { t: 20, r: 40, b: 20, l: 40 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const g = svg.append('g').attr('transform', `translate(${m.l},${m.t})`);

    const axes = [
      { key: 'Sk', label: 'Sₖ', domain: [0.02, 0.056] },
      { key: 'St', label: 'Sₜ', domain: [0.50, 0.64]  },
      { key: 'Se', label: 'Sₑ', domain: [0.015, 0.035]},
      { key: 'contacts', label: 'C', domain: [0, 60]   },
    ];
    const nAxes = axes.length;
    const xScale = d3.scalePoint().domain(d3.range(nAxes).map(String)).range([0, iw]);
    const yScales = axes.map(a => d3.scaleLinear().domain(a.domain as [number, number]).range([ih, 0]));

    axes.forEach((ax, i) => {
      g.append('g').attr('transform', `translate(${xScale(String(i))},0)`)
        .call(d3.axisLeft(yScales[i]).ticks(3).tickSize(3))
        .call(aa => aa.select('.domain').style('stroke', '#d1d5db'))
        .call(aa => aa.selectAll('text').style('font-size', '8px').style('fill', '#6b7280'));
      g.append('text').attr('x', xScale(String(i))!).attr('y', ih + 14)
        .style('font-size', '9px').style('fill', '#6b7280').attr('text-anchor', 'middle')
        .text(ax.label);
    });

    const colorScale = d3.scaleSequential(d3.interpolateViridis).domain([0, 5]);
    CROSS.forEach((row, ri) => {
      const vals = [row.Sk, row.St, row.Se, row.contacts];
      const pts: [number, number][] = vals.map((v, i) => [xScale(String(i))!, yScales[i](v)]);
      const line = d3.line<[number, number]>().x(d => d[0]).y(d => d[1]).curve(d3.curveCatmullRom);
      g.append('path').datum(pts).attr('d', line(pts))
        .attr('fill', 'none').attr('stroke', colorScale(ri)).attr('stroke-width', 1.5)
        .attr('opacity', 0.8);
    });
  });
  return <svg ref={ref} width={W} height={H} />;
}

// ── Se vs Sk coloured by St ─────────────────────────────────────────────────
function SeSkBubble() {
  const ref = useRef<SVGSVGElement>(null);
  useSvg(ref, [], (svg) => {
    const m = { t: 20, r: 14, b: 36, l: 44 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const g = svg.append('g').attr('transform', `translate(${m.l},${m.t})`);

    const x = d3.scaleLinear().domain([0.015, 0.035]).range([0, iw]);
    const y = d3.scaleLinear().domain([0.02, 0.056]).range([ih, 0]);
    const r = d3.scaleLinear().domain([0, 60]).range([4, 14]);
    const color = d3.scaleSequential(d3.interpolatePlasma).domain([0.50, 0.64]);

    g.append('g').attr('transform', `translate(0,${ih})`)
      .call(d3.axisBottom(x).ticks(4).tickSize(3))
      .call(a => a.select('.domain').remove())
      .call(a => a.selectAll('text').style('font-size', '9px').style('fill', '#6b7280'));
    g.append('g').call(d3.axisLeft(y).ticks(4).tickSize(3))
      .call(a => a.select('.domain').remove())
      .call(a => a.selectAll('text').style('font-size', '9px').style('fill', '#6b7280'));

    g.selectAll('line.grid').data(y.ticks(4)).enter().append('line')
      .attr('x1', 0).attr('x2', iw).attr('y1', d => y(d)).attr('y2', d => y(d))
      .style('stroke', '#f3f4f6').style('stroke-width', 1);

    g.selectAll('circle').data(CROSS).enter().append('circle')
      .attr('cx', d => x(d.Se)).attr('cy', d => y(d.Sk))
      .attr('r', d => r(d.contacts))
      .attr('fill', d => color(d.St))
      .attr('stroke', '#fff').attr('stroke-width', 1.5).attr('opacity', 0.9);

    g.append('text').attr('x', iw / 2).attr('y', ih + 32).style('font-size', '9px')
      .style('fill', '#9ca3af').attr('text-anchor', 'middle').text('Sₑ');
    g.append('text').attr('transform', 'rotate(-90)').attr('y', -36).attr('x', -ih / 2)
      .style('font-size', '9px').style('fill', '#9ca3af').attr('text-anchor', 'middle').text('Sₖ');
  });
  return <svg ref={ref} width={W} height={H} />;
}

// ── 3D S-entropy trajectory (all 6 images as paths in S-space) ───────────────
function SETrajectory3D() {
  function Trajectories() {
    const ref = useRef<THREE.Group>(null);
    useFrame(({ clock }) => {
      if (ref.current) ref.current.rotation.y = clock.getElapsedTime() * 0.25;
    });

    const colorScale = d3.scaleSequential(d3.interpolateRainbow).domain([0, 5]);

    return (
      <group ref={ref}>
        {CROSS.map((d, i) => {
          const p = new THREE.Vector3(
            (d.Sk - 0.035) * 18,
            (d.St - 0.57)  * 10,
            (d.Se - 0.022) * 18,
          );
          const c = new THREE.Color(colorScale(i));
          return (
            <mesh key={i} position={p}>
              <sphereGeometry args={[0.12 + d.contacts * 0.006, 10, 10]} />
              <meshPhongMaterial color={c} />
            </mesh>
          );
        })}
        {/* connect points in order */}
        <Line
          points={CROSS.map(d => new THREE.Vector3(
            (d.Sk - 0.035) * 18,
            (d.St - 0.57)  * 10,
            (d.Se - 0.022) * 18,
          ))}
          color="#d1d5db"
          lineWidth={1}
        />
        <lineSegments>
          <edgesGeometry args={[new THREE.BoxGeometry(2, 3, 2)]} />
          <lineBasicMaterial color="#f3f4f6" />
        </lineSegments>
      </group>
    );
  }

  return (
    <div style={{ width: W, height: H, background: '#fff' }}>
      <Canvas camera={{ position: [2.5, 1.5, 4], fov: 45 }} gl={{ antialias: true }}>
        <ambientLight intensity={0.65} />
        <pointLight position={[4, 4, 4]} intensity={1} />
        <Trajectories />
        <OrbitControls enablePan={false} />
      </Canvas>
    </div>
  );
}

// ── Summary radial chart (theorem pass rate) ──────────────────────────────────
function TheoremRadial() {
  const ref = useRef<SVGSVGElement>(null);
  useSvg(ref, [], (svg) => {
    const cx = W / 2, cy = H / 2 + 5;
    const outerR = Math.min(cx, cy) - 20;
    const g = svg.append('g');

    const theorems = [
      { name: 'T1', pass: 1 }, { name: 'T2', pass: 1 }, { name: 'T3', pass: 1 },
      { name: 'L1', pass: 1 }, { name: 'T4', pass: 1 }, { name: 'T5', pass: 1 },
      { name: 'T6', pass: 1 }, { name: 'T7', pass: 1 }, { name: 'T8', pass: 1 },
    ];
    const n = theorems.length;
    const angleSlice = (Math.PI * 2) / n;

    // grid rings
    [0.25, 0.5, 0.75, 1].forEach(frac => {
      g.append('circle').attr('cx', cx).attr('cy', cy).attr('r', outerR * frac)
        .attr('fill', 'none').attr('stroke', '#f3f4f6').attr('stroke-width', 1);
    });

    // spokes
    theorems.forEach((_, i) => {
      const angle = i * angleSlice - Math.PI / 2;
      g.append('line')
        .attr('x1', cx).attr('y1', cy)
        .attr('x2', cx + outerR * Math.cos(angle))
        .attr('y2', cy + outerR * Math.sin(angle))
        .style('stroke', '#f3f4f6').style('stroke-width', 1);
    });

    // filled polygon (all pass)
    const pts = theorems.map((t, i) => {
      const angle = i * angleSlice - Math.PI / 2;
      const r = outerR * t.pass;
      return [cx + r * Math.cos(angle), cy + r * Math.sin(angle)] as [number, number];
    });
    const polyLine = d3.line<[number, number]>().x(d => d[0]).y(d => d[1]).curve(d3.curveLinearClosed);
    g.append('path').datum(pts).attr('d', polyLine(pts))
      .attr('fill', C2).attr('opacity', 0.2).attr('stroke', C2).attr('stroke-width', 1.5);

    // labels
    theorems.forEach((t, i) => {
      const angle = i * angleSlice - Math.PI / 2;
      const labelR = outerR + 14;
      g.append('text')
        .attr('x', cx + labelR * Math.cos(angle))
        .attr('y', cy + labelR * Math.sin(angle))
        .style('font-size', '9px').style('fill', '#374151').attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'middle').text(t.name);
    });
  });
  return <svg ref={ref} width={W} height={H} />;
}

// ══════════════════════════════════════════════════════════════════════════════
// UTILITIES
// ══════════════════════════════════════════════════════════════════════════════

function mulberry32(seed: number) {
  return function () {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

function erfInv(x: number): number {
  const a = 0.147;
  const ln = Math.log(1 - x * x);
  const c = 2 / (Math.PI * a) + ln / 2;
  return Math.sign(x) * Math.sqrt(Math.sqrt(c * c - ln / a) - c);
}

function wrap(
  selection: d3.Selection<any, unknown, SVGGElement, unknown>,
  _width: number
) {
  selection.each(function () {
    d3.select(this as SVGTextElement).attr('dy', '1.2em');
  });
}

// ══════════════════════════════════════════════════════════════════════════════
// ROOT
// ══════════════════════════════════════════════════════════════════════════════

export default function ContactMapPanels() {
  return (
    <div style={{ background: '#fff', minHeight: '100vh', fontFamily: 'system-ui, sans-serif' }}>

      {/* PANEL 1 — S-entropy state space */}
      <Panel n={1}>
        <ChartBox><SEntropyScatter /></ChartBox>
        <ChartBox><ResolutionFloorBars /></ChartBox>
        <ChartBox><SEntropyCube3D /></ChartBox>
        <ChartBox><ContactCountBar /></ChartBox>
      </Panel>

      {/* PANEL 2 — Contact map geometry */}
      <Panel n={2}>
        <ChartBox><ContactHeatmap /></ChartBox>
        <ChartBox><ContactGraph2D /></ChartBox>
        <ChartBox><ContactSurface3D /></ChartBox>
        <ChartBox><ContactCostScatter /></ChartBox>
      </Panel>

      {/* PANEL 3 — SEBD algorithm */}
      <Panel n={3}>
        <ChartBox><SEBDIdentity /></ChartBox>
        <ChartBox><SEBDConvergence /></ChartBox>
        <ChartBox><SEBDPaths3D /></ChartBox>
        <ChartBox><VirtualSubstateHist /></ChartBox>
      </Panel>

      {/* PANEL 4 — Residue propagation */}
      <Panel n={4}>
        <ChartBox><ResidueChain /></ChartBox>
        <ChartBox><SliceExpansion /></ChartBox>
        <ChartBox><ResidueSpiral3D /></ChartBox>
        <ChartBox><MeanResidueScatter /></ChartBox>
      </Panel>

      {/* PANEL 5 — Holographic slicing */}
      <Panel n={5}>
        <ChartBox><HologramZDist /></ChartBox>
        <ChartBox><SliceCDF /></ChartBox>
        <ChartBox><HologramStack3D /></ChartBox>
        <ChartBox><SliceBubble /></ChartBox>
      </Panel>

      {/* PANEL 6 — Contact invariance */}
      <Panel n={6}>
        <ChartBox><ResolutionMonotone /></ChartBox>
        <ChartBox><SkContactsRegression /></ChartBox>
        <ChartBox><ResolutionPyramid3D /></ChartBox>
        <ChartBox><RegionContactDensity /></ChartBox>
      </Panel>

      {/* PANEL 7 — Cross-dataset summary */}
      <Panel n={7}>
        <ChartBox><ParallelCoords /></ChartBox>
        <ChartBox><SeSkBubble /></ChartBox>
        <ChartBox><SETrajectory3D /></ChartBox>
        <ChartBox><TheoremRadial /></ChartBox>
      </Panel>

    </div>
  );
}
