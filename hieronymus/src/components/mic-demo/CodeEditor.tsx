'use client';

import React, { useState } from 'react';

interface CodeEditorProps {
  value: string;
  onChange: (value: string) => void;
}

export default function CodeEditor({ value, onChange }: CodeEditorProps) {
  const [lineNumbers, setLineNumbers] = useState<number[]>([]);

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    onChange(e.target.value);
    updateLineNumbers(e.target.value);
  };

  const updateLineNumbers = (text: string) => {
    const lines = text.split('\n').length;
    setLineNumbers(Array.from({ length: lines }, (_, i) => i + 1));
  };

  React.useEffect(() => {
    updateLineNumbers(value);
  }, []);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Tab') {
      e.preventDefault();
      const target = e.currentTarget;
      const start = target.selectionStart;
      const end = target.selectionEnd;

      const newValue = value.substring(0, start) + '\t' + value.substring(end);
      onChange(newValue);

      setTimeout(() => {
        target.selectionStart = target.selectionEnd = start + 1;
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
            className="h-6 text-gray-600 text-right pr-3 text-xs leading-6"
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
          }}
          spellCheck={false}
        />

        {/* Syntax highlighting background (overlay) */}
        <div
          className="absolute inset-0 pointer-events-none text-gray-100 p-4 overflow-auto whitespace-pre-wrap break-words"
          style={{
            fontFamily: "'Cascadia Code', 'Monaco', 'Menlo', monospace",
            lineHeight: '1.5',
            color: 'transparent',
          }}
        >
          {/* Keywords highlighting */}
          {value.split('\n').map((line, idx) => (
            <div key={idx}>
              {line
                .split(/(analyze|load|estimate|visualize|measure|channel|from|to|as:|{|})/g)
                .map((token, i) => {
                  if (['analyze', 'load', 'estimate', 'visualize', 'measure', 'channel', 'from', 'to', 'as:'].includes(token)) {
                    return (
                      <span key={i} className="text-cyan-400 font-semibold">
                        {token}
                      </span>
                    );
                  }
                  if (['{', '}'].includes(token)) {
                    return (
                      <span key={i} className="text-yellow-400">
                        {token}
                      </span>
                    );
                  }
                  return (
                    <span key={i} className="text-gray-400">
                      {token}
                    </span>
                  );
                })}
            </div>
          ))}
        </div>
      </div>

      {/* Side info panel */}
      <div className="w-40 border-l border-gray-800/50 bg-[#0f1420] p-3 overflow-y-auto text-[9px]">
        <div className="space-y-4">
          <div>
            <div className="text-gray-500 uppercase tracking-widest mb-2">Keywords</div>
            <div className="space-y-1 text-gray-400">
              <div><span className="text-cyan-400">analyze</span> - start analysis</div>
              <div><span className="text-cyan-400">load</span> - load channel</div>
              <div><span className="text-cyan-400">estimate</span> - compute field</div>
              <div><span className="text-cyan-400">visualize</span> - render mode</div>
              <div><span className="text-cyan-400">measure</span> - compute metric</div>
            </div>
          </div>

          <div>
            <div className="text-gray-500 uppercase tracking-widest mb-2">Channels</div>
            <div className="space-y-1 text-gray-400">
              <div>synthetic</div>
              <div>dapi</div>
              <div>gfp</div>
              <div>red</div>
            </div>
          </div>

          <div>
            <div className="text-gray-500 uppercase tracking-widest mb-2">Modes</div>
            <div className="space-y-1 text-gray-400">
              <div>heatmap</div>
              <div>segmentation</div>
              <div>distance</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
