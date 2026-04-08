'use client';

import React, { useCallback, useRef, useState } from 'react';

interface TextDropZoneProps {
  onData: (text: string) => void;
  accept: string;
  fileTypes: string;
  placeholder?: string;
}

export default function TextDropZone({
  onData,
  accept,
  fileTypes,
  placeholder = 'Drop data file here',
}: TextDropZoneProps) {
  const [dragging, setDragging] = useState(false);
  const [fileName, setFileName] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const processFile = useCallback(
    (file: File) => {
      setFileName(file.name);
      const reader = new FileReader();
      reader.onload = (e) => {
        const text = e.target?.result as string;
        if (text) onData(text);
      };
      reader.readAsText(file);
    },
    [onData]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setDragging(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) processFile(file);
    },
    [processFile]
  );

  const handleClick = useCallback(() => {
    inputRef.current?.click();
  }, []);

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) processFile(file);
    },
    [processFile]
  );

  return (
    <div className="space-y-3">
      <div
        onClick={handleClick}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`border border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors duration-200
          ${
            dragging
              ? 'border-cyan-400 bg-cyan-400/5 text-cyan-400'
              : fileName
              ? 'border-green-500/50 text-green-400'
              : 'border-gray-700 text-gray-500 hover:border-gray-500 hover:text-gray-400'
          }`}
      >
        {fileName ? (
          <div className="flex flex-col items-center gap-2">
            <div className="text-xs font-mono bg-gray-900 px-3 py-1 rounded">
              {fileName}
            </div>
            <span className="text-xs text-gray-600">Click to replace</span>
          </div>
        ) : (
          <div className="space-y-1">
            <div className="text-lg">{placeholder}</div>
            <div className="text-xs">or click to browse</div>
            <div className="text-xs text-gray-700 mt-2">{fileTypes}</div>
          </div>
        )}
      </div>
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        onChange={handleFileChange}
        className="hidden"
      />
    </div>
  );
}
