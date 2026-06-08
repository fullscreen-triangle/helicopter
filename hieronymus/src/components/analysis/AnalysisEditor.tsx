'use client';

import React, { useState, useEffect } from 'react';

interface AnalysisEditorProps {
  value: string;
  onChange: (value: string) => void;
}

const KEYWORDS = ['const', 'let', 'var', 'function', 'return', 'if', 'else', 'for', 'while', 'await', 'async', 'new', 'true', 'false', 'null'];
const CHART_METHODS = ['line', 'bar', 'scatter', 'area', 'pie', 'radar', 'treemap', 'composed', 'title', 'data', 'x', 'y', 'key', 'series', 'colors', 'options', 'build', 'update'];
const BUILT_INS = ['c', 'log', 'data', 'Math', 'Array', 'Object', 'String', 'Number', 'JSON', 'console'];

export default function AnalysisEditor({ value, onChange }: AnalysisEditorProps) {
  const [lineNumbers, setLineNumbers] = useState<number[]>([]);

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    onChange(e.target.value);
    updateLineNumbers(e.target.value);
  };

  const updateLineNumbers = (text: string) => {
    const lines = text.split('\n').length;
    setLineNumbers(Array.from({ length: lines }, (_, i) => i + 1));
  };

  useEffect(() => {
    updateLineNumbers(value);
  }, []);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Tab') {
      e.preventDefault();
      const target = e.currentTarget;
      const start = target.selectionStart;
      const end = target.selectionEnd;

      const newValue = value.substring(0, start) + '  ' + value.substring(end);
      onChange(newValue);

      setTimeout(() => {
        target.selectionStart = target.selectionEnd = start + 2;
      }, 0);
    }
  };

  return (
    <div className="h-full flex overflow-hidden bg-[#0a0e27] font-mono text-sm">
      {/* Line numbers */}
      <div className="w-12 bg-[#0f1420] border-r border-gray-800 flex flex-col items-center py-4 overflow-hidden">
        {lineNumbers.map((num) => (
          <div
            key={num}
            className="h-6 text-gray-600 text-right pr-3 text-xs leading-6 select-none"
          >
            {num}
          </div>
        ))}
      </div>

      {/* Editor */}
      <div className="flex-1 relative overflow-hidden">
        <textarea
          value={value}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          className="absolute inset-0 w-full h-full resize-none outline-none bg-transparent text-gray-100 p-4 overflow-auto whitespace-pre-wrap break-words"
          style={{
            fontFamily: "'Cascadia Code', 'Monaco', 'Menlo', monospace",
            lineHeight: '1.5',
            caretColor: '#06b6d4',
          }}
          spellCheck={false}
        />

        {/* Syntax highlighting overlay */}
        <div
          className="absolute inset-0 pointer-events-none text-gray-100 p-4 overflow-auto whitespace-pre-wrap break-words"
          style={{
            fontFamily: "'Cascadia Code', 'Monaco', 'Menlo', monospace",
            lineHeight: '1.5',
            color: 'transparent',
          }}
        >
          {value.split('\n').map((line, lineIdx) => (
            <div key={lineIdx}>
              {(() => {
                // Simple tokenizer
                const tokens: Array<{ text: string; type: string }> = [];
                let current = '';
                let i = 0;

                while (i < line.length) {
                  if (line[i] === '/' && line[i + 1] === '/') {
                    // Comment
                    if (current) tokens.push({ text: current, type: 'normal' });
                    tokens.push({ text: line.substring(i), type: 'comment' });
                    break;
                  } else if (line[i] === '"' || line[i] === "'" || line[i] === '`') {
                    // String
                    if (current) tokens.push({ text: current, type: 'normal' });
                    const quote = line[i];
                    let str = quote;
                    i++;
                    while (i < line.length && line[i] !== quote) {
                      str += line[i];
                      if (line[i] === '\\') {
                        str += line[++i];
                      }
                      i++;
                    }
                    if (i < line.length) str += line[i];
                    tokens.push({ text: str, type: 'string' });
                    current = '';
                    i++;
                  } else if (/\w/.test(line[i]) || line[i] === '$' || line[i] === '_') {
                    // Identifier
                    current += line[i];
                    i++;
                  } else {
                    // Other
                    if (current) {
                      const type = KEYWORDS.includes(current)
                        ? 'keyword'
                        : CHART_METHODS.includes(current)
                        ? 'method'
                        : BUILT_INS.includes(current)
                        ? 'builtin'
                        : 'identifier';
                      tokens.push({ text: current, type });
                      current = '';
                    }
                    tokens.push({ text: line[i], type: /[\(\[\{]|[\)\]\}]/.test(line[i]) ? 'bracket' : 'operator' });
                    i++;
                  }
                }

                if (current) {
                  const type = KEYWORDS.includes(current)
                    ? 'keyword'
                    : CHART_METHODS.includes(current)
                    ? 'method'
                    : BUILT_INS.includes(current)
                    ? 'builtin'
                    : 'identifier';
                  tokens.push({ text: current, type });
                }

                return tokens.map((token, i) => {
                  let color = 'text-gray-400';
                  if (token.type === 'keyword') color = 'text-cyan-400 font-semibold';
                  else if (token.type === 'method') color = 'text-yellow-400 font-semibold';
                  else if (token.type === 'string') color = 'text-green-400';
                  else if (token.type === 'comment') color = 'text-gray-600 italic';
                  else if (token.type === 'builtin') color = 'text-violet-400 font-semibold';
                  else if (token.type === 'bracket') color = 'text-pink-400';
                  else if (token.type === 'operator') color = 'text-orange-400';

                  return (
                    <span key={i} className={color}>
                      {token.text}
                    </span>
                  );
                });
              })()}
            </div>
          ))}
        </div>
      </div>

      {/* Help sidebar */}
      <div className="w-48 border-l border-gray-800/50 bg-[#0f1420] p-3 overflow-y-auto text-[8px] hidden lg:block">
        <div className="space-y-4">
          <div>
            <div className="text-gray-500 uppercase tracking-widest mb-2 font-semibold">Chart Types</div>
            <div className="space-y-1 text-gray-400">
              <div>c.<span className="text-yellow-400">line</span>(&apos;id&apos;)</div>
              <div>c.<span className="text-yellow-400">bar</span>(&apos;id&apos;)</div>
              <div>c.<span className="text-yellow-400">scatter</span>(&apos;id&apos;)</div>
              <div>c.<span className="text-yellow-400">area</span>(&apos;id&apos;)</div>
              <div>c.<span className="text-yellow-400">pie</span>(&apos;id&apos;)</div>
              <div>c.<span className="text-yellow-400">radar</span>(&apos;id&apos;)</div>
            </div>
          </div>

          <div>
            <div className="text-gray-500 uppercase tracking-widest mb-2 font-semibold">Methods</div>
            <div className="space-y-1 text-gray-400 font-mono">
              <div>.<span className="text-yellow-400">title</span>(str)</div>
              <div>.<span className="text-yellow-400">data</span>(arr)</div>
              <div>.<span className="text-yellow-400">x</span>(key)</div>
              <div>.<span className="text-yellow-400">y</span>(key)</div>
              <div>.<span className="text-yellow-400">series</span>(arr)</div>
              <div>.<span className="text-yellow-400">build</span>()</div>
            </div>
          </div>

          <div>
            <div className="text-gray-500 uppercase tracking-widest mb-2 font-semibold">Functions</div>
            <div className="space-y-1 text-gray-400">
              <div><span className="text-cyan-400">log</span>(msg)</div>
              <div>data<span className="text-orange-400">.</span>fourier</div>
              <div>Math<span className="text-orange-400">.</span>*</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
