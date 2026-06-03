'use client';

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';

interface Dataset {
  db: string;
  dataset_id: string;
  name: string;
  description: string;
  resolution: number;
  channels: string[];
  url?: string;
  image_count?: number;
}

interface DatabaseBrowserProps {
  onImageSelected: (imageData: {
    db: string;
    dataset_id: string;
    image_id: string;
    filename: string;
    data: string; // base64
    shape: [number, number];
    dtype: string;
  }) => void;
  compact?: boolean;
}

const API_BASE = process.env.NEXT_PUBLIC_SCOPE_API || 'http://localhost:5000';

export default function DatabaseBrowser({
  onImageSelected,
  compact = false,
}: DatabaseBrowserProps) {
  const [datasets, setDatasets] = useState<Record<string, Dataset[]>>({});
  const [selectedDb, setSelectedDb] = useState<string>('');
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [images, setImages] = useState<string[]>([]);
  const [selectedImage, setSelectedImage] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');
  const [preview, setPreview] = useState<string>('');

  // Load databases on mount
  useEffect(() => {
    const loadDatabases = async () => {
      try {
        const response = await fetch(`${API_BASE}/databases`);
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const data = await response.json();
        setDatasets(data);

        // Select first database and dataset
        const firstDb = Object.keys(data)[0];
        if (firstDb && data[firstDb].length > 0) {
          setSelectedDb(firstDb);
          setSelectedDataset(data[firstDb][0].dataset_id);
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        if (message.includes('Failed to fetch')) {
          setError('Database backend not available. Start Flask: cd turbine/scope/server && python -m flask --app app run --port 5000');
        } else {
          setError(`Database error: ${message}`);
        }
      }
    };

    loadDatabases();
  }, []);

  // Load images when dataset changes
  useEffect(() => {
    const loadImages = async () => {
      if (!selectedDb || !selectedDataset) return;

      try {
        setImages([]);
        setSelectedImage('');
        const response = await fetch(
          `${API_BASE}/databases/${selectedDb}/${selectedDataset}/images`
        );
        const data = await response.json();
        setImages(data.images || []);
        if (data.images && data.images.length > 0) {
          setSelectedImage(data.images[0]);
        }
      } catch (err) {
        setError(`Failed to load images: ${err}`);
      }
    };

    loadImages();
  }, [selectedDb, selectedDataset]);

  // Load image preview
  useEffect(() => {
    const loadPreview = async () => {
      if (!selectedDb || !selectedDataset || !selectedImage) return;

      try {
        setLoading(true);
        const response = await fetch(
          `${API_BASE}/databases/${selectedDb}/${selectedDataset}/${selectedImage}?channel=DAPI`
        );
        const data = await response.json();

        if (data.success) {
          // Decode image and create preview
          const binary = atob(data.data);
          const bytes = new Uint8Array(binary.length);
          for (let i = 0; i < binary.length; i++) {
            bytes[i] = binary.charCodeAt(i);
          }
          const float32 = new Float32Array(bytes.buffer);
          const arr = new Array(float32);

          // Create simple preview (normalize to 0-255)
          const min = Math.min(...float32);
          const max = Math.max(...float32);
          const normalized = new Uint8ClampedArray(float32.length);
          for (let i = 0; i < float32.length; i++) {
            normalized[i] = Math.round(((float32[i] - min) / (max - min)) * 255);
          }

          // Create canvas preview (simplified - just show stats)
          const stats = {
            shape: data.shape,
            min: min.toFixed(2),
            max: max.toFixed(2),
            mean: (float32.reduce((a, b) => a + b) / float32.length).toFixed(2),
          };

          setPreview(JSON.stringify(stats, null, 2));

          // Call parent callback
          onImageSelected({
            db: selectedDb,
            dataset_id: selectedDataset,
            image_id: selectedImage,
            filename: data.filename,
            data: data.data,
            shape: data.shape,
            dtype: data.dtype,
          });
        }
      } catch (err) {
        setError(`Failed to load preview: ${err}`);
      } finally {
        setLoading(false);
      }
    };

    loadPreview();
  }, [selectedImage, selectedDb, selectedDataset, onImageSelected]);

  const currentDataset = datasets[selectedDb]?.find(
    (d) => d.dataset_id === selectedDataset
  );

  if (compact) {
    // Compact mode: just dropdowns
    return (
      <div className="space-y-3">
        <div>
          <label className="text-[9px] text-gray-600 uppercase tracking-widest block mb-1">
            Database
          </label>
          <select
            value={selectedDb}
            onChange={(e) => setSelectedDb(e.target.value)}
            className="w-full px-2 py-1 bg-[#0f1420] border border-gray-800 rounded text-xs text-gray-300 focus:border-cyan-400 outline-none"
          >
            {Object.entries(datasets).map(([db, dsList]) => (
              <optgroup key={db} label={db}>
                {dsList.map((ds) => (
                  <option key={ds.dataset_id} value={db}>
                    {ds.name}
                  </option>
                ))}
              </optgroup>
            ))}
          </select>
        </div>

        <div>
          <label className="text-[9px] text-gray-600 uppercase tracking-widest block mb-1">
            Image
          </label>
          <select
            value={selectedImage}
            onChange={(e) => setSelectedImage(e.target.value)}
            className="w-full px-2 py-1 bg-[#0f1420] border border-gray-800 rounded text-xs text-gray-300 focus:border-cyan-400 outline-none"
          >
            {images.map((img) => (
              <option key={img} value={img}>
                {img}
              </option>
            ))}
          </select>
        </div>

        {error && (
          <div className="text-[8px] text-red-400 bg-red-900/20 p-2 rounded">
            {error}
          </div>
        )}

        {loading && (
          <div className="text-[8px] text-cyan-400">Loading image...</div>
        )}
      </div>
    );
  }

  // Full mode: with preview and details
  return (
    <div className="space-y-4">
      {/* Database selector */}
      <div>
        <label className="text-[10px] text-gray-600 uppercase tracking-widest block mb-2">
          Database
        </label>
        <div className="space-y-2">
          {Object.entries(datasets).map(([db, dsList]) => (
            <div key={db}>
              <div className="text-[8px] text-gray-600 mb-1">{db}</div>
              {dsList.map((ds) => (
                <button
                  key={ds.dataset_id}
                  onClick={() => {
                    setSelectedDb(db);
                    setSelectedDataset(ds.dataset_id);
                  }}
                  className={`w-full text-left px-3 py-2 rounded text-[9px] transition-colors ${
                    selectedDataset === ds.dataset_id
                      ? 'bg-cyan-400/20 text-cyan-400 border border-cyan-400'
                      : 'bg-gray-800/30 text-gray-400 border border-gray-800 hover:border-gray-700'
                  }`}
                >
                  <div className="font-semibold">{ds.name}</div>
                  <div className="text-[8px] text-gray-500 mt-1">
                    {ds.description}
                  </div>
                  <div className="text-[8px] text-gray-600 mt-1">
                    Resolution: {ds.resolution} µm/px • Channels:{' '}
                    {ds.channels.join(', ')}
                  </div>
                </button>
              ))}
            </div>
          ))}
        </div>
      </div>

      {/* Image selector */}
      {images.length > 0 && (
        <div>
          <label className="text-[10px] text-gray-600 uppercase tracking-widest block mb-2">
            Image ({images.length} available)
          </label>
          <div className="grid grid-cols-2 gap-2">
            {images.map((img) => (
              <button
                key={img}
                onClick={() => setSelectedImage(img)}
                className={`px-2 py-2 rounded text-[8px] transition-colors ${
                  selectedImage === img
                    ? 'bg-cyan-400/20 text-cyan-400 border border-cyan-400'
                    : 'bg-gray-800/30 text-gray-400 border border-gray-800 hover:border-gray-700'
                }`}
              >
                {img}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Preview / Details */}
      {currentDataset && (
        <div className="pt-3 border-t border-gray-800">
          <div className="text-[9px] text-gray-600 uppercase tracking-widest mb-2">
            Image Details
          </div>
          {loading ? (
            <div className="text-[8px] text-cyan-400">Loading preview...</div>
          ) : preview ? (
            <div className="bg-gray-900/50 p-2 rounded text-[8px] text-gray-400 font-mono whitespace-pre overflow-auto max-h-40">
              {preview}
            </div>
          ) : null}
        </div>
      )}

      {/* Error display */}
      {error && (
        <div className="bg-red-900/20 border border-red-800 rounded p-3">
          <div className="text-[9px] text-red-400 font-semibold">Error</div>
          <div className="text-[8px] text-red-300 mt-1">{error}</div>
        </div>
      )}
    </div>
  );
}
