// SCOPE Lexer — tokenizes SCOPE source code

export enum TokenType {
  // Keywords
  SCOPE = 'SCOPE',
  CHANNELS = 'CHANNELS',
  SYNC = 'SYNC',
  CELL = 'CELL',
  COORDINATE_SPACE = 'COORDINATE_SPACE',
  FIELD = 'FIELD',
  DEPTH = 'DEPTH',
  MORPHISMS = 'MORPHISMS',
  OBSERVE = 'OBSERVE',
  CATALYZE = 'CATALYZE',
  FUSE = 'FUSE',
  MEASURE_DISTANCE = 'MEASURE_DISTANCE',
  ACCESS = 'ACCESS',
  DISPATCH = 'DISPATCH',
  WHEN = 'WHEN',
  DO = 'DO',
  EXECUTE = 'EXECUTE',
  EMIT = 'EMIT',
  AT = 'AT',
  BOUNDS = 'BOUNDS',
  ACTION = 'ACTION',

  // Literals
  IDENTIFIER = 'IDENTIFIER',
  NUMBER = 'NUMBER',
  STRING = 'STRING',

  // Symbols
  LBRACE = '{',
  RBRACE = '}',
  LPAREN = '(',
  RPAREN = ')',
  LBRACKET = '[',
  RBRACKET = ']',
  EQUALS = '=',
  PIPE = '|>',
  COMMA = ',',
  DOT = '.',
  COLON = ':',

  // Special
  EOF = 'EOF',
  NEWLINE = 'NEWLINE',
}

export interface Token {
  type: TokenType;
  value: string;
  line: number;
  column: number;
}

export class Lexer {
  private input: string;
  private position: number = 0;
  private line: number = 1;
  private column: number = 1;
  private tokens: Token[] = [];

  private keywords = new Map<string, TokenType>([
    ['scope', TokenType.SCOPE],
    ['channels', TokenType.CHANNELS],
    ['sync', TokenType.SYNC],
    ['cell', TokenType.CELL],
    ['coordinate_space', TokenType.COORDINATE_SPACE],
    ['field', TokenType.FIELD],
    ['depth', TokenType.DEPTH],
    ['morphisms', TokenType.MORPHISMS],
    ['observe', TokenType.OBSERVE],
    ['catalyze', TokenType.CATALYZE],
    ['fuse', TokenType.FUSE],
    ['measure_distance', TokenType.MEASURE_DISTANCE],
    ['access', TokenType.ACCESS],
    ['dispatch', TokenType.DISPATCH],
    ['when', TokenType.WHEN],
    ['do', TokenType.DO],
    ['execute', TokenType.EXECUTE],
    ['emit', TokenType.EMIT],
    ['at', TokenType.AT],
    ['bounds', TokenType.BOUNDS],
    ['action', TokenType.ACTION],
  ]);

  constructor(input: string) {
    this.input = input;
  }

  tokenize(): Token[] {
    while (this.position < this.input.length) {
      this.skipWhitespaceAndComments();
      if (this.position >= this.input.length) break;

      const char = this.current();

      if (char === '{') this.addToken(TokenType.LBRACE, '{');
      else if (char === '}') this.addToken(TokenType.RBRACE, '}');
      else if (char === '(') this.addToken(TokenType.LPAREN, '(');
      else if (char === ')') this.addToken(TokenType.RPAREN, ')');
      else if (char === '[') this.addToken(TokenType.LBRACKET, '[');
      else if (char === ']') this.addToken(TokenType.RBRACKET, ']');
      else if (char === '=') this.addToken(TokenType.EQUALS, '=');
      else if (char === ',') this.addToken(TokenType.COMMA, ',');
      else if (char === '.') {
        // Could be a dot or start of a number like .5
        if (/\d/.test(this.peek())) {
          this.readNumber();
        } else {
          this.addToken(TokenType.DOT, '.');
        }
      }
      else if (char === ':') this.addToken(TokenType.COLON, ':');
      else if (char === '|' && this.peek() === '>') {
        this.advance();
        this.addToken(TokenType.PIPE, '|>');
      } else if (char === '"' || char === "'") {
        this.readString();
      } else if (char === '-' && /[\d.]/.test(this.peek())) {
        // Negative number
        this.readNumber();
      } else if (/\d/.test(char)) {
        this.readNumber();
      } else if (/[a-zA-Z_µ]/.test(char) || /[À-ſ]/.test(char)) {
        // Handle Latin characters and Greek letters like µ
        this.readIdentifier();
      } else {
        throw new Error(`Unexpected character: ${char} (${char.charCodeAt(0)}) at ${this.line}:${this.column}`);
      }
    }

    this.addToken(TokenType.EOF, '');
    return this.tokens;
  }

  private current(): string {
    return this.input[this.position] || '';
  }

  private peek(offset = 1): string {
    return this.input[this.position + offset] || '';
  }

  private advance(): string {
    const char = this.current();
    this.position++;
    if (char === '\n') {
      this.line++;
      this.column = 1;
    } else {
      this.column++;
    }
    return char;
  }

  private skipWhitespaceAndComments(): void {
    while (this.position < this.input.length) {
      const char = this.current();
      if (/\s/.test(char)) {
        this.advance();
      } else if (char === '/' && this.peek() === '/') {
        // Skip line comment
        while (this.current() !== '\n' && this.position < this.input.length) {
          this.advance();
        }
      } else {
        break;
      }
    }
  }

  private readString(): void {
    const quote = this.current();
    const startLine = this.line;
    const startColumn = this.column;
    this.advance(); // skip opening quote

    let value = '';
    while (this.current() !== quote && this.position < this.input.length) {
      if (this.current() === '\\') {
        this.advance();
        const char = this.current();
        if (char === 'n') value += '\n';
        else if (char === 't') value += '\t';
        else if (char === '\\') value += '\\';
        else if (char === quote) value += quote;
        else value += char;
        this.advance();
      } else {
        value += this.current();
        this.advance();
      }
    }

    if (this.current() !== quote) {
      throw new Error(`Unterminated string at ${startLine}:${startColumn}`);
    }
    this.advance(); // skip closing quote

    this.tokens.push({
      type: TokenType.STRING,
      value,
      line: startLine,
      column: startColumn,
    });
  }

  private readNumber(): void {
    const startLine = this.line;
    const startColumn = this.column;
    let value = '';

    // Handle optional leading minus
    if (this.current() === '-') {
      value += this.current();
      this.advance();
    }

    // Read digits and decimal point
    while (/[\d.]/.test(this.current())) {
      value += this.current();
      this.advance();
    }

    // Handle scientific notation (e.g., 1.5e-6)
    if (/[eE]/.test(this.current())) {
      value += this.current();
      this.advance();

      if (/[+\-]/.test(this.current())) {
        value += this.current();
        this.advance();
      }

      while (/\d/.test(this.current())) {
        value += this.current();
        this.advance();
      }
    }

    this.tokens.push({
      type: TokenType.NUMBER,
      value,
      line: startLine,
      column: startColumn,
    });
  }

  private readIdentifier(): void {
    const startLine = this.line;
    const startColumn = this.column;
    let value = '';

    while (/[a-zA-Z0-9_µ]/.test(this.current()) || /[À-ſ]/.test(this.current())) {
      value += this.current();
      this.advance();
    }

    const type = this.keywords.get(value.toLowerCase()) || TokenType.IDENTIFIER;
    this.tokens.push({
      type,
      value,
      line: startLine,
      column: startColumn,
    });
  }

  private addToken(type: TokenType, value: string): void {
    this.tokens.push({
      type,
      value,
      line: this.line,
      column: this.column,
    });
    this.advance();
  }
}
