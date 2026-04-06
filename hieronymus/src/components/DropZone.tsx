'use client';

import React, { useCallback, useRef, useState } from 'react';

interface DropZoneProps {
  onDrop: (imageData: ImageData) => void;
}

export default function DropZone({ onDrop }: DropZoneProps) {
  const [dragging, setDragging] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const processFile = useCallback(
    (file: File) => {
      if (!file.type.startsWith('image/')) return;

      setFileName(file.name);

      const reader = new FileReader();
      reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
          // Create canvas to get ImageData
          const canvas = document.createElement('canvas');
          canvas.width = img.naturalWidth || img.width;
          canvas.height = img.naturalHeight || img.height;
          const ctx = canvas.getContext('2d');
          if (!ctx) return;

          ctx.drawImage(img, 0, 0);
          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

          // Create preview thumbnail
          const thumbCanvas = document.createElement('canvas');
          const maxDim = 200;
          const scale = Math.min(maxDim / canvas.width, maxDim / canvas.height, 1);
          thumbCanvas.width = Math.round(canvas.width * scale);
          thumbCanvas.height = Math.round(canvas.height * scale);
          const thumbCtx = thumbCanvas.getContext('2d');
          if (thumbCtx) {
            thumbCtx.drawImage(img, 0, 0, thumbCanvas.width, thumbCanvas.height);
            setPreview(thumbCanvas.toDataURL());
          }

          onDrop(imageData);
        };
        img.src = e.target?.result as string;
      };
      reader.readAsDataURL(file);
    },
    [onDrop]
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
              : preview
              ? 'border-green-500/50 text-green-400'
              : 'border-gray-700 text-gray-500 hover:border-gray-500 hover:text-gray-400'
          }`}
      >
        {preview ? (
          <div className="flex flex-col items-center gap-2">
            <img
              src={preview}
              alt="Preview"
              className="max-w-full max-h-32 rounded"
              style={{ imageRendering: 'pixelated' }}
            />
            <span className="text-xs">{fileName}</span>
            <span className="text-xs text-gray-600">Click to replace</span>
          </div>
        ) : (
          <div className="space-y-1">
            <div className="text-lg">Drop image here</div>
            <div className="text-xs">or click to browse</div>
            <div className="text-xs text-gray-700 mt-2">PNG / JPG / TIFF</div>
          </div>
        )}
      </div>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        onChange={handleFileChange}
        className="hidden"
      />
    </div>
  );
}
