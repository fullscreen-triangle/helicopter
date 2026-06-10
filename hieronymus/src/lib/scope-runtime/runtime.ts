// SCOPE Runtime — orchestrates the four phases for one program + one image

import type { ScopeProgram } from '@/lib/scope-compiler/ast';
import { compilePhase } from './phases/compile';
import { measurePhase } from './phases/measure';
import { executePhase, ExecuteOutput } from './phases/execute';
import { emitPhase } from './phases/emit';
import type { ScopeResult, EntropyPhasePoint } from './result-types';

export type { ScopeResult } from './result-types';

export interface ImagePayload {
  data: Float32Array;
  width: number;
  height: number;
}

export async function runScope(
  program: ScopeProgram,
  imagePayload: ImagePayload,
): Promise<ScopeResult> {
  const { data: image, width, height } = imagePayload;

  const depth = program.coordinateSpace?.depth
    ?? program.morphisms[0]?.expr.observe.depth
    ?? 10;
  const fieldSizeUm = program.coordinateSpace?.fieldX ?? 100;

  // ── Phase 1: COMPILE ───────────────────────────────────────────────────
  const compileOut = compilePhase(image, program);
  let { sk, st, se } = compileOut;

  // ── Phase 2: MEASURE ───────────────────────────────────────────────────
  const measureOut = measurePhase(image, width, height, sk, st, se);
  // MEASURE is deterministic — entropy unchanged
  sk = measureOut.entropyPoint.sk;
  st = measureOut.entropyPoint.st;
  se = measureOut.entropyPoint.se;

  // ── Phase 3: EXECUTE ───────────────────────────────────────────────────
  // Run each morphism in declaration order; pass results forward for fuse()
  const morphismResults: Record<string, ExecuteOutput | null> = {};
  const allEntropyPoints: EntropyPhasePoint[] = [];
  let lastExecute: ExecuteOutput | null = null;

  // Determine which morphism to run:
  // If dispatch block exists and a cell was matched, run the dispatched morphism.
  // Otherwise run the last declared morphism (as per spec).
  let targetMorphismName: string | null = null;
  if (program.dispatch && compileOut.cellName) {
    const rule = program.dispatch.rules.find(r => r.cell === compileOut.cellName);
    if (rule?.action.kind === 'ExecuteAction') {
      targetMorphismName = rule.action.morphismRef;
    }
  }
  if (!targetMorphismName) {
    targetMorphismName = program.morphisms[program.morphisms.length - 1]?.name ?? null;
  }

  for (const morphism of program.morphisms) {
    // Only run morphisms needed: either the target or those it fuses
    const isFused = program.morphisms.some(m =>
      m.expr.steps.some(s => s.kind === 'FuseStep' && (s as any).morphismRef === morphism.name)
    );
    const isTarget = morphism.name === targetMorphismName;
    if (!isTarget && !isFused) continue;

    const execOut = executePhase(
      morphism,
      image, width, height,
      measureOut.scaleField.alpha,
      measureOut.scaleField.mean,
      depth, fieldSizeUm,
      sk, st, se,
      morphismResults,
    );
    morphismResults[morphism.name] = execOut;
    allEntropyPoints.push(...execOut.entropyPoints);

    if (isTarget) {
      lastExecute = execOut;
      sk = execOut.sk; st = execOut.st; se = execOut.se;
    }
  }

  if (!lastExecute) {
    throw new Error('No morphism was executed');
  }

  // ── Phase 4: EMIT ──────────────────────────────────────────────────────
  const result = emitPhase(
    program, image, width, height,
    measureOut, lastExecute,
    compileOut.sk, compileOut.st, compileOut.se,
    measureOut.snr, measureOut.crlbPixels, measureOut.channelCapacity,
    allEntropyPoints,
  );

  // Combine all log lines in phase order
  result.log = [
    ...compileOut.log,
    ...measureOut.log,
    ...lastExecute.log,
    ...result.log,
  ];

  return result;
}
