'use client';

import React, { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import AnalysisEditor from '@/components/analysis/AnalysisEditor';
import { getScopeExample } from '@/lib/scope-examples';

interface FileTreeNode {
  name: string;
  type: 'folder' | 'file';
  path: string;
  children?: FileTreeNode[];
}

interface IDEProps {
  onCodeChange: (code: string) => void;
  onFileSelect: (path: string, code: string) => void;
  output: string[];
  isExecuting: boolean;
}

export default function ScopeIDELayout({ onCodeChange, onFileSelect, output, isExecuting }: IDEProps) {
  const [code, setCode] = useState<string>(() => {
    try {
      return getScopeExample('PROPHASE', 'synthetic');
    } catch {
      return '';
    }
  });
  const [selectedFile, setSelectedFile] = useState('examples/prophase.scope');
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set(['examples', 'datasets', '/']));
  const outputRef = useRef<HTMLDivElement>(null);

  // File tree structure
  const fileTree: FileTreeNode = {
    name: 'scope-sandbox',
    type: 'folder',
    path: '/',
    children: [
      {
        name: 'examples',
        type: 'folder',
        path: '/examples',
        children: [
          { name: 'prophase.scope', type: 'file', path: '/examples/prophase.scope' },
          { name: 'metaphase.scope', type: 'file', path: '/examples/metaphase.scope' },
          { name: 'anaphase.scope', type: 'file', path: '/examples/anaphase.scope' },
          { name: 'tutorial_01_hello.scope', type: 'file', path: '/examples/tutorial_01_hello.scope' },
          { name: 'tutorial_02_measurement.scope', type: 'file', path: '/examples/tutorial_02_measurement.scope' },
          { name: 'tutorial_03_data_sources.scope', type: 'file', path: '/examples/tutorial_03_data_sources.scope' },
        ],
      },
      {
        name: 'datasets',
        type: 'folder',
        path: '/datasets',
        children: [
          { name: 'BBBC007_v1_images', type: 'folder', path: '/datasets/BBBC007_v1_images', children: [] },
          { name: 'BBBC011_v1_images', type: 'folder', path: '/datasets/BBBC011_v1_images', children: [] },
          { name: 'AICS-24-part06', type: 'folder', path: '/datasets/AICS-24-part06', children: [] },
          { name: 'drosophila_kc167', type: 'folder', path: '/datasets/drosophila_kc167', children: [] },
          { name: 'c_elegans', type: 'folder', path: '/datasets/c_elegans', children: [] },
          { name: 'chinese_hamster_ovary', type: 'folder', path: '/datasets/chinese_hamster_ovary', children: [] },
          { name: 'human_HT29_colon_cancer', type: 'folder', path: '/datasets/human_HT29_colon_cancer', children: [] },
        ],
      },
      {
        name: 'my_scripts',
        type: 'folder',
        path: '/my_scripts',
        children: [
          { name: 'analysis_draft.scope', type: 'file', path: '/my_scripts/analysis_draft.scope' },
        ],
      },
    ],
  };

  const toggleFolder = (path: string) => {
    const newExpanded = new Set(expandedFolders);
    if (newExpanded.has(path)) {
      newExpanded.delete(path);
    } else {
      newExpanded.add(path);
    }
    setExpandedFolders(newExpanded);
  };

  const handleFileSelect = (path: string, name: string) => {
    if (name.endsWith('.scope')) {
      setSelectedFile(path);
      // Load example code
      const exampleCode = getScopeExample('PROPHASE', 'synthetic');
      setCode(exampleCode);
      onFileSelect(path, exampleCode);
    }
  };

  const renderFileTree = (node: FileTreeNode, level: number = 0) => {
    const isExpanded = expandedFolders.has(node.path);
    const isFolder = node.type === 'folder';
    const isSelected = selectedFile === node.path;

    return (
      <div key={node.path}>
        {level >= 0 && (
          <div
            onClick={() => isFolder && toggleFolder(node.path)}
            className={`flex items-center gap-1 px-2 py-1 text-[11px] cursor-pointer transition-colors ${
              isSelected ? 'bg-cyan-400/20 text-cyan-400' : 'text-gray-400 hover:text-gray-200'
            }`}
            style={{ paddingLeft: `${level * 12}px` }}
          >
            {isFolder && (
              <span className="text-[8px]">{isExpanded ? '▼' : '▶'}</span>
            )}
            {!isFolder && <span className="text-[8px]">•</span>}
            <span
              onClick={(e) => {
                if (!isFolder) {
                  e.stopPropagation();
                  handleFileSelect(node.path, node.name);
                }
              }}
            >
              {node.name}
            </span>
          </div>
        )}
        {isFolder && isExpanded && node.children && (
          <div>
            {node.children.map((child) => renderFileTree(child, level + 1))}
          </div>
        )}
      </div>
    );
  };

  useEffect(() => {
    if (outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight;
    }
  }, [output]);

  return (
    <div className="min-h-[calc(100vh-140px)] flex flex-col bg-[#050810]">
      {/* Header */}
      <div className="px-6 py-3 border-b border-gray-800/50 flex items-center justify-between">
        <div>
          <h1 className="text-sm font-semibold tracking-wider text-cyan-400">SCOPE SANDBOX</h1>
          <p className="text-[10px] text-gray-600 mt-1">
            File Browser | Code Editor | Execution Output
          </p>
        </div>
        <div className="text-[9px] text-gray-500">
          {selectedFile}
        </div>
      </div>

      {/* Three-column layout */}
      <div className="flex-1 grid grid-cols-3 overflow-hidden gap-0">
        {/* Left: File Browser */}
        <div className="border-r border-gray-800/50 flex flex-col overflow-hidden bg-[#0a0e18]">
          <div className="px-4 py-2 border-b border-gray-800/50">
            <div className="text-[9px] text-gray-600 uppercase tracking-widest">Files</div>
          </div>
          <div className="flex-1 overflow-y-auto font-mono text-[11px]">
            {fileTree.children && fileTree.children.map((node) => renderFileTree(node, 1))}
          </div>
        </div>

        {/* Middle: Code Editor */}
        <div className="border-r border-gray-800/50 flex flex-col overflow-hidden">
          <div className="px-4 py-2 border-b border-gray-800/50 flex items-center justify-between">
            <div className="text-[9px] text-gray-600 uppercase tracking-widest">Editor</div>
            <div className="text-[8px] text-gray-700">
              {isExecuting ? (
                <span className="text-yellow-400">Executing...</span>
              ) : (
                <span className="text-green-400">Ready</span>
              )}
            </div>
          </div>
          <div className="flex-1 overflow-hidden">
            <AnalysisEditor
              value={code}
              onChange={(newCode) => {
                setCode(newCode);
                onCodeChange(newCode);
              }}
            />
          </div>
        </div>

        {/* Right: Output */}
        <div className="border-l border-gray-800/50 flex flex-col overflow-hidden">
          <div className="px-4 py-2 border-b border-gray-800/50">
            <div className="text-[9px] text-gray-600 uppercase tracking-widest">Output</div>
          </div>
          <div
            ref={outputRef}
            className="flex-1 overflow-y-auto p-3 font-mono text-[8px] space-y-0.5"
          >
            {output.length === 0 ? (
              <div className="text-gray-700">Ready to execute...</div>
            ) : (
              output.map((line, i) => (
                <div
                  key={i}
                  className={
                    line.startsWith('✓')
                      ? 'text-green-400'
                      : line.startsWith('❌')
                        ? 'text-red-400'
                        : line.startsWith('⚠')
                          ? 'text-yellow-400'
                          : line.startsWith('Generated') || line.startsWith('Loaded') || line.startsWith('Fetching')
                            ? 'text-cyan-400'
                            : 'text-gray-400'
                  }
                >
                  {line}
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
