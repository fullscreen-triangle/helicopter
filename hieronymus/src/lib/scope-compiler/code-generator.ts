/**
 * SCOPE Code Generator — Emit ExecutionPlan
 */

import { ScopeProgram, MorphismChain } from './parser';

export interface ExecutionPlan {
  name: string;
  coordinate_space: {
    field_width_um: number;
    field_height_um: number;
    depth: number;
    lambda_s: number;
    lambda_t: number;
  };
  channels: {
    sync: { id: string; frequency: number } | null;
    cells: Array<{
      id: string;
      bounds_min: number;
      bounds_max: number;
      morphism_id: string;
    }>;
  };
  morphisms: Array<{
    id: string;
    steps: Array<{
      type: string;
      params: Record<string, any>;
    }>;
  }>;
}

export function generateExecutionPlan(ast: ScopeProgram): ExecutionPlan {
  const { name, channels, coordinateSpace, morphisms, dispatch } = ast;

  // Build morphism lookup
  const morphismMap = new Map<string, MorphismChain>();
  for (const chain of morphisms.chains) {
    morphismMap.set(chain.id, chain);
  }

  // Build dispatch lookup
  const dispatchMap = new Map<string, string>();
  for (const when of dispatch.whenStatements) {
    if (when.action.type === 'ExecuteAction') {
      dispatchMap.set(when.cellId, (when.action as any).chainId);
    }
  }

  // Generate cells
  const cells = channels.declarations
    .filter((d) => d.type === 'CellDeclaration')
    .map((d: any) => ({
      id: d.id,
      bounds_min: d.bounds[0],
      bounds_max: d.bounds[1],
      morphism_id: dispatchMap.get(d.id) || d.action,
    }));

  // Extract sync
  const syncDecl = channels.declarations.find((d) => d.type === 'SyncDeclaration') as any;
  const sync = syncDecl
    ? { id: syncDecl.id, frequency: syncDecl.frequency }
    : null;

  // Generate morphisms
  const morphismsList = morphisms.chains.map((chain) => ({
    id: chain.id,
    steps: chain.steps.map((step) => {
      switch (step.type) {
        case 'ObserveStep':
          return {
            type: 'observe',
            params: {
              frame: (step as any).frameRef,
              depth: (step as any).n,
            },
          };
        case 'CatalyzeStep':
          return {
            type: 'catalyze',
            params: {
              constraint: (step as any).catalyst,
            },
          };
        case 'FuseStep':
          return {
            type: 'fuse',
            params: {
              chain: (step as any).chainRef,
              rho: (step as any).rho,
            },
          };
        case 'MeasureDistanceStep':
          return {
            type: 'measure',
            params: {
              target_a: (step as any).target1,
              target_b: (step as any).target2,
            },
          };
        case 'AccessStep':
          return {
            type: 'access',
            params: {
              structure: (step as any).target,
            },
          };
        default:
          return { type: 'unknown', params: {} };
      }
    }),
  }));

  return {
    name,
    coordinate_space: {
      field_width_um: coordinateSpace.field[0],
      field_height_um: coordinateSpace.field[1],
      depth: coordinateSpace.depth,
      lambda_s: coordinateSpace.lambdaS,
      lambda_t: coordinateSpace.lambdaT,
    },
    channels: {
      sync,
      cells,
    },
    morphisms: morphismsList,
  };
}
