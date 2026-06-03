// SCOPE Compiler — compiles SCOPE source to executable IR

import { Lexer } from './lexer';
import { Parser, ScopeProgram } from './parser';

export interface CompiledProgram {
  name: string;
  version: string;
  ast: ScopeProgram;
  ir: ProgramIR;
  errors: CompileError[];
  warnings: CompileWarning[];
}

export interface ProgramIR {
  name: string;
  channels: ChannelIR[];
  coordinateSpace: CoordinateSpaceIR;
  morphisms: MorphismIR[];
  dispatchTable: DispatchTableEntry[];
}

export interface ChannelIR {
  id: string;
  type: 'sync' | 'cell';
  frequency?: number;
  bounds?: [number, number];
}

export interface CoordinateSpaceIR {
  field: [number, number, string];
  depth: number;
  lambdaS: number;
  lambdaT: number;
}

export interface MorphismIR {
  id: string;
  steps: MorphismStepIR[];
}

export interface MorphismStepIR {
  type: string;
  params: Record<string, any>;
}

export interface DispatchTableEntry {
  cellId: string;
  action: string;
  chainId?: string;
  label?: string;
}

export interface CompileError {
  line: number;
  column: number;
  message: string;
}

export interface CompileWarning {
  line: number;
  message: string;
}

export class SCOPECompiler {
  private errors: CompileError[] = [];
  private warnings: CompileWarning[] = [];

  compile(source: string): CompiledProgram {
    this.errors = [];
    this.warnings = [];

    try {
      // Lexical analysis
      const lexer = new Lexer(source);
      const tokens = lexer.tokenize();

      // Parsing
      const parser = new Parser(tokens);
      const ast = parser.parse();

      // Semantic analysis and code generation
      const ir = this.generateIR(ast);

      // Validation
      this.validate(ast, ir);

      return {
        name: ast.name,
        version: '1.0.0',
        ast,
        ir,
        errors: this.errors,
        warnings: this.warnings,
      };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.errors.push({
        line: 0,
        column: 0,
        message,
      });

      return {
        name: 'unknown',
        version: '1.0.0',
        ast: null as any,
        ir: null as any,
        errors: this.errors,
        warnings: this.warnings,
      };
    }
  }

  private generateIR(ast: ScopeProgram): ProgramIR {
    const channels: ChannelIR[] = ast.channels.declarations.map((decl) => {
      if (decl.type === 'SyncDeclaration') {
        return {
          id: decl.id,
          type: 'sync',
          frequency: decl.frequency,
        };
      } else {
        return {
          id: decl.id,
          type: 'cell',
          bounds: decl.bounds,
        };
      }
    });

    const morphisms: MorphismIR[] = ast.morphisms.chains.map((chain) => ({
      id: chain.id,
      steps: chain.steps.map((step) => ({
        type: step.type,
        params: this.extractStepParams(step),
      })),
    }));

    const dispatchTable: DispatchTableEntry[] = ast.dispatch.whenStatements.map((when) => {
      const action = when.action;
      if (action.type === 'ExecuteAction') {
        return {
          cellId: when.cellId,
          action: 'execute',
          chainId: action.chainId,
        };
      } else if (action.type === 'EmitAction') {
        return {
          cellId: when.cellId,
          action: 'emit',
          label: action.label,
        };
      } else {
        return {
          cellId: when.cellId,
          action: 'block',
        };
      }
    });

    return {
      name: ast.name,
      channels,
      coordinateSpace: {
        field: ast.coordinateSpace.field,
        depth: ast.coordinateSpace.depth,
        lambdaS: ast.coordinateSpace.lambdaS,
        lambdaT: ast.coordinateSpace.lambdaT,
      },
      morphisms,
      dispatchTable,
    };
  }

  private extractStepParams(step: any): Record<string, any> {
    switch (step.type) {
      case 'ObserveStep':
        return { frameRef: step.frameRef, n: step.n };
      case 'CatalyzeStep':
        return { catalyst: step.catalyst };
      case 'FuseStep':
        return { chainRef: step.chainRef, rho: step.rho };
      case 'MeasureDistanceStep':
        return { target1: step.target1, target2: step.target2 };
      case 'AccessStep':
        return { target: step.target };
      default:
        return {};
    }
  }

  private validate(ast: ScopeProgram, ir: ProgramIR): void {
    // Check all referenced chains exist
    const chainIds = new Set(ir.morphisms.map((m) => m.id));
    for (const entry of ir.dispatchTable) {
      if (entry.chainId && !chainIds.has(entry.chainId)) {
        this.errors.push({
          line: 0,
          column: 0,
          message: `Morphism chain "${entry.chainId}" referenced in dispatch but not defined`,
        });
      }
    }

    // Check all referenced cells exist
    const cellIds = new Set(ir.channels.filter((c) => c.type === 'cell').map((c) => c.id));
    for (const entry of ir.dispatchTable) {
      if (!cellIds.has(entry.cellId)) {
        this.errors.push({
          line: 0,
          column: 0,
          message: `Cell "${entry.cellId}" referenced in dispatch but not defined`,
        });
      }
    }

    // Warn about unused chains
    const usedChains = new Set(
      ir.dispatchTable.filter((e) => e.chainId).map((e) => e.chainId)
    );
    for (const chain of ir.morphisms) {
      if (!usedChains.has(chain.id)) {
        this.warnings.push({
          line: 0,
          message: `Morphism chain "${chain.id}" is defined but never used`,
        });
      }
    }
  }
}

// Export convenience function
export function compileScope(source: string): CompiledProgram {
  const compiler = new SCOPECompiler();
  return compiler.compile(source);
}
