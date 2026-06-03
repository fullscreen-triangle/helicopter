// SCOPE Parser — builds AST from tokens

import { Token, TokenType, Lexer } from './lexer';

export interface ASTNode {
  type: string;
}

export interface ScopeProgram extends ASTNode {
  type: 'ScopeProgram';
  name: string;
  channels: ChannelsBlock;
  coordinateSpace: CoordinateSpaceBlock;
  morphisms: MorphismsBlock;
  dispatch: DispatchBlock;
}

export interface ChannelsBlock extends ASTNode {
  type: 'ChannelsBlock';
  declarations: ChannelDeclaration[];
}

export interface ChannelDeclaration extends ASTNode {
  type: 'SyncDeclaration' | 'CellDeclaration';
}

export interface SyncDeclaration extends ChannelDeclaration {
  type: 'SyncDeclaration';
  id: string;
  frequency: number;
}

export interface CellDeclaration extends ChannelDeclaration {
  type: 'CellDeclaration';
  id: string;
  bounds: [number, number];
  action: string;
}

export interface CoordinateSpaceBlock extends ASTNode {
  type: 'CoordinateSpaceBlock';
  field: [number, number, string];
  depth: number;
  lambdaS: number;
  lambdaT: number;
}

export interface MorphismsBlock extends ASTNode {
  type: 'MorphismsBlock';
  chains: MorphismChain[];
}

export interface MorphismChain extends ASTNode {
  type: 'MorphismChain';
  id: string;
  steps: MorphismStep[];
}

export interface MorphismStep extends ASTNode {
  type:
    | 'ObserveStep'
    | 'CatalyzeStep'
    | 'FuseStep'
    | 'MeasureDistanceStep'
    | 'AccessStep';
}

export interface ObserveStep extends MorphismStep {
  type: 'ObserveStep';
  frameRef: string;
  n: number;
}

export interface CatalyzeStep extends MorphismStep {
  type: 'CatalyzeStep';
  catalyst: string;
}

export interface FuseStep extends MorphismStep {
  type: 'FuseStep';
  chainRef: string;
  rho: number;
}

export interface MeasureDistanceStep extends MorphismStep {
  type: 'MeasureDistanceStep';
  target1: string;
  target2: string;
}

export interface AccessStep extends MorphismStep {
  type: 'AccessStep';
  target: string;
}

export interface DispatchBlock extends ASTNode {
  type: 'DispatchBlock';
  whenStatements: WhenStatement[];
}

export interface WhenStatement extends ASTNode {
  type: 'WhenStatement';
  cellId: string;
  action: DispatchAction;
}

export interface DispatchAction extends ASTNode {
  type: 'ExecuteAction' | 'EmitAction' | 'BlockAction';
}

export interface ExecuteAction extends DispatchAction {
  type: 'ExecuteAction';
  chainId: string;
}

export interface EmitAction extends DispatchAction {
  type: 'EmitAction';
  label: string;
}

export interface BlockAction extends DispatchAction {
  type: 'BlockAction';
  actions: DispatchAction[];
}

export class Parser {
  private tokens: Token[];
  private position: number = 0;

  constructor(tokens: Token[]) {
    this.tokens = tokens;
  }

  static fromSource(source: string): Parser {
    const lexer = new Lexer(source);
    const tokens = lexer.tokenize();
    return new Parser(tokens);
  }

  parse(): ScopeProgram {
    this.expect(TokenType.SCOPE);
    const name = this.expectIdentifier();
    this.expect(TokenType.LBRACE);

    const channels = this.parseChannels();
    const coordinateSpace = this.parseCoordinateSpace();
    const morphisms = this.parseMorphisms();
    const dispatch = this.parseDispatch();

    this.expect(TokenType.RBRACE);
    this.expect(TokenType.EOF);

    return {
      type: 'ScopeProgram',
      name,
      channels,
      coordinateSpace,
      morphisms,
      dispatch,
    };
  }

  private parseChannels(): ChannelsBlock {
    this.expect(TokenType.CHANNELS);
    this.expect(TokenType.LBRACE);

    const declarations: ChannelDeclaration[] = [];
    while (this.current().type !== TokenType.RBRACE) {
      if (this.current().type === TokenType.SYNC) {
        declarations.push(this.parseSyncDeclaration());
      } else if (this.current().type === TokenType.CELL) {
        declarations.push(this.parseCellDeclaration());
      } else {
        throw new Error(`Unexpected token in channels: ${this.current().value}`);
      }
    }

    this.expect(TokenType.RBRACE);

    return { type: 'ChannelsBlock', declarations };
  }

  private parseSyncDeclaration(): SyncDeclaration {
    this.expect(TokenType.SYNC);
    const id = this.expectIdentifier();
    this.expect(TokenType.AT);
    const frequency = parseFloat(this.expectNumber());

    return {
      type: 'SyncDeclaration',
      id,
      frequency,
    };
  }

  private parseCellDeclaration(): CellDeclaration {
    this.expect(TokenType.CELL);
    const id = this.expectIdentifier();
    this.expect(TokenType.BOUNDS);
    this.expect(TokenType.LPAREN);
    const min = parseFloat(this.expectNumber());
    this.expect(TokenType.COMMA);
    const max = parseFloat(this.expectNumber());
    this.expect(TokenType.RPAREN);
    this.expect(TokenType.ACTION);
    const action = this.expectIdentifier();

    return {
      type: 'CellDeclaration',
      id,
      bounds: [min, max],
      action,
    };
  }

  private parseCoordinateSpace(): CoordinateSpaceBlock {
    this.expect(TokenType.COORDINATE_SPACE);
    this.expect(TokenType.LBRACE);

    this.expect(TokenType.FIELD);
    const fieldX = parseFloat(this.expectNumber());
    this.expect(TokenType.IDENTIFIER); // 'x'
    const fieldY = parseFloat(this.expectNumber());
    const unit = this.expectIdentifier(); // 'µm'

    this.expect(TokenType.DEPTH);
    const depth = parseInt(this.expectNumber(), 10);

    this.expect(TokenType.IDENTIFIER); // 'lambda_s'
    const lambdaS = parseFloat(this.expectNumber());

    this.expect(TokenType.IDENTIFIER); // 'lambda_t'
    const lambdaT = parseFloat(this.expectNumber());

    this.expect(TokenType.RBRACE);

    return {
      type: 'CoordinateSpaceBlock',
      field: [fieldX, fieldY, unit],
      depth,
      lambdaS,
      lambdaT,
    };
  }

  private parseMorphisms(): MorphismsBlock {
    this.expect(TokenType.MORPHISMS);
    this.expect(TokenType.LBRACE);

    const chains: MorphismChain[] = [];
    while (this.current().type !== TokenType.RBRACE) {
      const id = this.expectIdentifier();
      this.expect(TokenType.EQUALS);

      const steps: MorphismStep[] = [];
      steps.push(this.parseStep());

      while (this.current().type === TokenType.PIPE) {
        this.expect(TokenType.PIPE);
        steps.push(this.parseStep());
      }

      chains.push({ type: 'MorphismChain', id, steps });
    }

    this.expect(TokenType.RBRACE);

    return { type: 'MorphismsBlock', chains };
  }

  private parseStep(): MorphismStep {
    const token = this.current();

    if (token.type === TokenType.OBSERVE) {
      return this.parseObserveStep();
    } else if (token.type === TokenType.CATALYZE) {
      return this.parseCatalyzeStep();
    } else if (token.type === TokenType.FUSE) {
      return this.parseFuseStep();
    } else if (token.type === TokenType.MEASURE_DISTANCE) {
      return this.parseMeasureDistanceStep();
    } else if (token.type === TokenType.ACCESS) {
      return this.parseAccessStep();
    } else {
      throw new Error(`Unexpected step: ${token.value}`);
    }
  }

  private parseObserveStep(): ObserveStep {
    this.expect(TokenType.OBSERVE);
    this.expect(TokenType.LPAREN);
    const frameRef = this.expectIdentifier();
    this.expect(TokenType.COMMA);
    this.expectIdentifier(); // 'n'
    this.expect(TokenType.EQUALS);
    const n = parseInt(this.expectNumber(), 10);
    this.expect(TokenType.RPAREN);

    return { type: 'ObserveStep', frameRef, n };
  }

  private parseCatalyzeStep(): CatalyzeStep {
    this.expect(TokenType.CATALYZE);
    this.expect(TokenType.LPAREN);
    const catalyst = this.expectIdentifier();
    this.expect(TokenType.RPAREN);

    return { type: 'CatalyzeStep', catalyst };
  }

  private parseFuseStep(): FuseStep {
    this.expect(TokenType.FUSE);
    this.expect(TokenType.LPAREN);
    const chainRef = this.expectIdentifier();
    this.expect(TokenType.COMMA);
    this.expectIdentifier(); // 'rho'
    this.expect(TokenType.EQUALS);
    const rho = parseFloat(this.expectNumber());
    this.expect(TokenType.RPAREN);

    return { type: 'FuseStep', chainRef, rho };
  }

  private parseMeasureDistanceStep(): MeasureDistanceStep {
    this.expect(TokenType.MEASURE_DISTANCE);
    this.expect(TokenType.LPAREN);
    const target1 = this.expectIdentifier();
    this.expect(TokenType.COMMA);
    const target2 = this.expectIdentifier();
    this.expect(TokenType.RPAREN);

    return { type: 'MeasureDistanceStep', target1, target2 };
  }

  private parseAccessStep(): AccessStep {
    this.expect(TokenType.ACCESS);
    this.expect(TokenType.LPAREN);
    const target = this.expectIdentifier();
    this.expect(TokenType.RPAREN);

    return { type: 'AccessStep', target };
  }

  private parseDispatch(): DispatchBlock {
    this.expect(TokenType.DISPATCH);
    this.expect(TokenType.LBRACE);

    const whenStatements: WhenStatement[] = [];
    while (this.current().type !== TokenType.RBRACE) {
      this.expect(TokenType.WHEN);
      const cellId = this.expectIdentifier();
      this.expect(TokenType.DO);

      const action = this.parseDispatchAction();
      whenStatements.push({ type: 'WhenStatement', cellId, action });
    }

    this.expect(TokenType.RBRACE);

    return { type: 'DispatchBlock', whenStatements };
  }

  private parseDispatchAction(): DispatchAction {
    if (this.current().type === TokenType.EXECUTE) {
      this.expect(TokenType.EXECUTE);
      this.expect(TokenType.LPAREN);
      const chainId = this.expectIdentifier();
      this.expect(TokenType.RPAREN);
      return { type: 'ExecuteAction', chainId };
    } else if (this.current().type === TokenType.EMIT) {
      this.expect(TokenType.EMIT);
      const label = this.expectIdentifier();
      return { type: 'EmitAction', label };
    } else if (this.current().type === TokenType.LBRACE) {
      this.expect(TokenType.LBRACE);
      const actions: DispatchAction[] = [];
      while (this.current().type !== TokenType.RBRACE) {
        actions.push(this.parseDispatchAction());
        if (this.current().type === TokenType.COMMA) {
          this.expect(TokenType.COMMA);
        }
      }
      this.expect(TokenType.RBRACE);
      return { type: 'BlockAction', actions };
    } else {
      throw new Error(`Unexpected dispatch action: ${this.current().value}`);
    }
  }

  private current(): Token {
    return this.tokens[this.position] || { type: TokenType.EOF, value: '', line: 0, column: 0 };
  }

  private advance(): Token {
    const token = this.current();
    this.position++;
    return token;
  }

  private expect(type: TokenType): Token {
    const token = this.current();
    if (token.type !== type) {
      throw new Error(
        `Expected ${type} but got ${token.type} at ${token.line}:${token.column}`
      );
    }
    return this.advance();
  }

  private expectIdentifier(): string {
    const token = this.expect(TokenType.IDENTIFIER);
    return token.value;
  }

  private expectNumber(): string {
    const token = this.expect(TokenType.NUMBER);
    return token.value;
  }
}
