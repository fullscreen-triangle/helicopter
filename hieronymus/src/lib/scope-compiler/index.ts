/**
 * SCOPE Compiler — Main Entry Point
 * Orchestrates lexer → parser → type-checker → code-generator
 */

import { Lexer } from './lexer';
import { Parser } from './parser';
import { typeCheck } from './type-checker';
import { generateExecutionPlan, ExecutionPlan } from './code-generator';

export interface CompileResult {
  success: boolean;
  name?: string;
  ir?: ExecutionPlan;
  errors: string[];
  warnings: string[];
}

export function compileScope(sourceCode: string): CompileResult {
  const errors: string[] = [];
  const warnings: string[] = [];

  try {
    // Stage 1: Lex
    const lexer = new Lexer(sourceCode);
    const tokens = lexer.tokenize();

    // Stage 2: Parse
    const parser = new Parser(tokens);
    const ast = parser.parse();

    // Stage 3: Type Check
    const typeCheckResult = typeCheck(ast);
    if (typeCheckResult.errors.length > 0) {
      return {
        success: false,
        errors: typeCheckResult.errors,
        warnings: typeCheckResult.warnings,
      };
    }
    warnings.push(...typeCheckResult.warnings);

    // Stage 4: Code Generate
    const ir = generateExecutionPlan(ast);

    return {
      success: true,
      name: ast.name,
      ir,
      errors: [],
      warnings,
    };
  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : String(error);
    errors.push(errorMsg);
    return {
      success: false,
      errors,
      warnings,
    };
  }
}

export type { ExecutionPlan };
export { Lexer, TokenType } from './lexer';
export { Parser } from './parser';
export { typeCheck } from './type-checker';
