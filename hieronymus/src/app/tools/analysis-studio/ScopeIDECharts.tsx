/**
 * SCOPE IDE Chart Visualizations
 * Using D3.js for actual graphical charts
 */

import React from 'react';

interface EntropyChartProps {
  phases: Array<{ phase: string; S_k: number; S_t: number; S_e: number }>;
}

export const EntropyBarChart: React.FC<EntropyChartProps> = ({ phases }) => {
  if (!phases || phases.length === 0) return null;

  const width = 600;
  const height = 300;
  const margin = { top: 20, right: 30, bottom: 50, left: 50 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  // Scale for values
  const maxValue = 1.0;
  const yScale = innerHeight / maxValue;

  // Bar positioning
  const barGroupWidth = innerWidth / phases.length;
  const barWidth = barGroupWidth / 3.5;
  const barSpacing = barGroupWidth / 8;

  return (
    <svg width={width} height={height} style={{ background: '#2d2d2d', borderRadius: '4px' }}>
      <defs>
        <linearGradient id="s_k_grad" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stopColor="#63b3ed" />
          <stop offset="100%" stopColor="#3182ce" />
        </linearGradient>
        <linearGradient id="s_t_grad" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stopColor="#f6d55c" />
          <stop offset="100%" stopColor="#f6ad55" />
        </linearGradient>
        <linearGradient id="s_e_grad" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stopColor="#fc8181" />
          <stop offset="100%" stopColor="#f56565" />
        </linearGradient>
      </defs>

      <g transform={`translate(${margin.left},${margin.top})`}>
        {/* Grid lines */}
        {[0.25, 0.5, 0.75, 1.0].map((gridVal) => (
          <g key={`grid-${gridVal}`}>
            <line
              x1={0}
              y1={innerHeight - gridVal * yScale}
              x2={innerWidth}
              y2={innerHeight - gridVal * yScale}
              stroke="rgba(255,255,255,0.1)"
              strokeWidth="1"
              strokeDasharray="4"
            />
            <text
              x={-10}
              y={innerHeight - gridVal * yScale + 4}
              fontSize="11"
              fill="rgba(255,255,255,0.5)"
              textAnchor="end"
            >
              {gridVal.toFixed(2)}
            </text>
          </g>
        ))}

        {/* Y-axis */}
        <line x1={0} y1={0} x2={0} y2={innerHeight} stroke="rgba(255,255,255,0.3)" strokeWidth="1" />

        {/* X-axis */}
        <line x1={0} y1={innerHeight} x2={innerWidth} y2={innerHeight} stroke="rgba(255,255,255,0.3)" strokeWidth="1" />

        {/* Bars */}
        {phases.map((phase, idx) => {
          const x = (idx + 0.5) * barGroupWidth;

          return (
            <g key={`phase-${idx}`}>
              {/* S_k bar */}
              <rect
                x={x - barWidth - barSpacing}
                y={innerHeight - phase.S_k * yScale}
                width={barWidth}
                height={phase.S_k * yScale}
                fill="url(#s_k_grad)"
                rx="2"
              />

              {/* S_t bar */}
              <rect
                x={x}
                y={innerHeight - phase.S_t * yScale}
                width={barWidth}
                height={phase.S_t * yScale}
                fill="url(#s_t_grad)"
                rx="2"
              />

              {/* S_e bar */}
              <rect
                x={x + barWidth + barSpacing}
                y={innerHeight - phase.S_e * yScale}
                width={barWidth}
                height={phase.S_e * yScale}
                fill="url(#s_e_grad)"
                rx="2"
              />

              {/* Value labels */}
              <text
                x={x - barWidth - barSpacing}
                y={innerHeight - phase.S_k * yScale - 5}
                fontSize="10"
                fill="#63b3ed"
                textAnchor="middle"
              >
                {phase.S_k.toFixed(2)}
              </text>
              <text
                x={x}
                y={innerHeight - phase.S_t * yScale - 5}
                fontSize="10"
                fill="#f6ad55"
                textAnchor="middle"
              >
                {phase.S_t.toFixed(2)}
              </text>
              <text
                x={x + barWidth + barSpacing}
                y={innerHeight - phase.S_e * yScale - 5}
                fontSize="10"
                fill="#fc8181"
                textAnchor="middle"
              >
                {phase.S_e.toFixed(2)}
              </text>

              {/* Phase label */}
              <text
                x={x}
                y={innerHeight + 20}
                fontSize="12"
                fill="rgba(255,255,255,0.7)"
                textAnchor="middle"
                fontWeight="bold"
              >
                {phase.phase}
              </text>
            </g>
          );
        })}

        {/* Legend */}
        <g transform={`translate(${innerWidth - 150}, -10)`}>
          <rect x="0" y="0" width="140" height="80" fill="rgba(0,0,0,0.5)" rx="4" />

          <rect x="10" y="10" width="12" height="12" fill="url(#s_k_grad)" />
          <text x="28" y="19" fontSize="12" fill="rgba(255,255,255,0.8)">
            S_k (Knowledge)
          </text>

          <rect x="10" y="30" width="12" height="12" fill="url(#s_t_grad)" />
          <text x="28" y="39" fontSize="12" fill="rgba(255,255,255,0.8)">
            S_t (Temporal)
          </text>

          <rect x="10" y="50" width="12" height="12" fill="url(#s_e_grad)" />
          <text x="28" y="59" fontSize="12" fill="rgba(255,255,255,0.8)">
            S_e (Entropy)
          </text>
        </g>
      </g>
    </svg>
  );
};

export const MeasurementChart: React.FC<{
  measurements: Array<{ label: string; distance_um: number; uncertainty_um: number }>;
}> = ({ measurements }) => {
  if (!measurements || measurements.length === 0) return null;

  const width = 600;
  const height = 300;
  const margin = { top: 20, right: 30, bottom: 50, left: 60 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  const maxDistance = Math.max(...measurements.map((m) => m.distance_um + m.uncertainty_um)) * 1.1;
  const yScale = innerHeight / maxDistance;

  return (
    <svg width={width} height={height} style={{ background: '#2d2d2d', borderRadius: '4px' }}>
      <defs>
        <linearGradient id="dist_grad" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stopColor="#48bb78" />
          <stop offset="100%" stopColor="#38a169" />
        </linearGradient>
      </defs>

      <g transform={`translate(${margin.left},${margin.top})`}>
        {/* Y-axis */}
        <line x1={0} y1={0} x2={0} y2={innerHeight} stroke="rgba(255,255,255,0.3)" strokeWidth="1" />

        {/* X-axis */}
        <line x1={0} y1={innerHeight} x2={innerWidth} y2={innerHeight} stroke="rgba(255,255,255,0.3)" strokeWidth="1" />

        {/* Bars with error bars */}
        {measurements.map((meas, idx) => {
          const barWidth = (innerWidth / measurements.length) * 0.7;
          const x = (idx + 0.5) * (innerWidth / measurements.length);

          return (
            <g key={`meas-${idx}`}>
              {/* Error bar (uncertainty range) */}
              <line
                x1={x}
                y1={innerHeight - (meas.distance_um + meas.uncertainty_um) * yScale}
                x2={x}
                y2={innerHeight - (meas.distance_um - meas.uncertainty_um) * yScale}
                stroke="#fc8181"
                strokeWidth="2"
              />
              <line
                x1={x - 6}
                y1={innerHeight - (meas.distance_um + meas.uncertainty_um) * yScale}
                x2={x + 6}
                y2={innerHeight - (meas.distance_um + meas.uncertainty_um) * yScale}
                stroke="#fc8181"
                strokeWidth="2"
              />
              <line
                x1={x - 6}
                y1={innerHeight - (meas.distance_um - meas.uncertainty_um) * yScale}
                x2={x + 6}
                y2={innerHeight - (meas.distance_um - meas.uncertainty_um) * yScale}
                stroke="#fc8181"
                strokeWidth="2"
              />

              {/* Distance bar */}
              <rect
                x={x - barWidth / 2}
                y={innerHeight - meas.distance_um * yScale}
                width={barWidth}
                height={meas.distance_um * yScale}
                fill="url(#dist_grad)"
                rx="2"
              />

              {/* Value label */}
              <text
                x={x}
                y={innerHeight - meas.distance_um * yScale - 10}
                fontSize="11"
                fill="#48bb78"
                textAnchor="middle"
                fontWeight="bold"
              >
                {meas.distance_um.toFixed(1)}
              </text>

              {/* Uncertainty label */}
              <text
                x={x}
                y={innerHeight - (meas.distance_um + meas.uncertainty_um) * yScale - 5}
                fontSize="9"
                fill="#fc8181"
                textAnchor="middle"
              >
                ±{meas.uncertainty_um.toFixed(2)}
              </text>

              {/* Measurement label */}
              <text
                x={x}
                y={innerHeight + 20}
                fontSize="11"
                fill="rgba(255,255,255,0.7)"
                textAnchor="middle"
              >
                {meas.label}
              </text>
            </g>
          );
        })}

        {/* Y-axis label */}
        <text x={-innerHeight / 2} y={-45} fontSize="12" fill="rgba(255,255,255,0.7)" textAnchor="middle" transform="rotate(-90)">
          Distance (µm)
        </text>
      </g>
    </svg>
  );
};
