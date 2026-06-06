/**
 * SCOPE Type Checker — Invariant Verification
 */

import { ScopeProgram, ChannelDeclaration, MorphismChain, SyncDeclaration, CellDeclaration } from './parser';

export interface TypeCheckResult {
  errors: string[];
  warnings: string[];
}

export function typeCheck(ast: ScopeProgram): TypeCheckResult {
  const errors: string[] = [];
  const warnings: string[] = [];

  const { channels, coordinateSpace, morphisms, dispatch } = ast;

  // Extract depth and cell IDs
  const declaredDepth = coordinateSpace.depth;
  const cellIds = new Set<string>();
  const chainIds = new Set<string>();

  // Collect cell and chain names
  for (const decl of channels.declarations) {
    if (decl.type === 'CellDeclaration') {
      cellIds.add((decl as CellDeclaration).id);
    }
  }

  for (const chain of morphisms.chains) {
    chainIds.add(chain.id);
  }

  // Rule 1: Partition depth consistency
  for (const chain of morphisms.chains) {
    for (const step of chain.steps) {
      if (step.type === 'ObserveStep') {
        const observeStep = step as any;
        if (observeStep.n !== declaredDepth) {
          errors.push(
            `Morphism '${chain.id}': observe depth ${observeStep.n} ` +
            `does not match coordinate_space depth ${declaredDepth}`
          );
        }
      }
    }
  }

  // Rule 2: S-entropy balance heuristic
  for (const chain of morphisms.chains) {
    let catalyzeCount = 0;
    let accessCount = 0;
    for (const step of chain.steps) {
      if (step.type === 'CatalyzeStep') catalyzeCount++;
      if (step.type === 'AccessStep') accessCount++;
    }
    if (Math.abs(catalyzeCount - accessCount) > 3) {
      warnings.push(
        `Morphism '${chain.id}': catalyze ${catalyzeCount} vs access ${accessCount} ` +
        `steps seem imbalanced (S-entropy may not conserve)`
      );
    }
  }

  // Rule 3: Dispatch completeness
  const dispatchedCells = new Set<string>();
  for (const when of dispatch.whenStatements) {
    dispatchedCells.add(when.cellId);
  }

  for (const cellId of cellIds) {
    if (!dispatchedCells.has(cellId)) {
      errors.push(`Cell '${cellId}' has no dispatch rule`);
    }
  }

  // Rule 4: Chain existence
  for (const decl of channels.declarations) {
    if (decl.type === 'CellDeclaration') {
      const cellDecl = decl as CellDeclaration;
      if (!chainIds.has(cellDecl.action)) {
        errors.push(
          `Cell '${cellDecl.id}': action '${cellDecl.action}' is not defined in morphisms`
        );
      }
    }
  }

  for (const when of dispatch.whenStatements) {
    if (when.action.type === 'ExecuteAction') {
      const executeAction = when.action as any;
      if (!chainIds.has(executeAction.chainId)) {
        errors.push(
          `Dispatch rule: action '${executeAction.chainId}' is not defined in morphisms`
        );
      }
    }
  }

  return { errors, warnings };
}
