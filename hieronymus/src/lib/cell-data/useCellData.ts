'use client';
import { useState, useEffect, useCallback } from 'react';
import { createCellDataManager } from './cellDataService';

export const useCellData = (huggingfaceApiKey: string) => {
  const [manager] = useState(() => createCellDataManager(huggingfaceApiKey));
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  return { manager, loading, error, setLoading, setError };
};

export const useAllenCell = (cellId: string | null, huggingfaceApiKey: string) => {
  const [cellData, setCellData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const { manager } = useCellData(huggingfaceApiKey);

  useEffect(() => {
    if (!cellId) return;
    setLoading(true);
    setError(null);
    manager.getAllenCell().getCellById(cellId)
      .then(setCellData)
      .catch((err: Error) => setError(err))
      .finally(() => setLoading(false));
  }, [cellId, manager]);

  return { cellData, loading, error };
};

export const useHuggingFaceModel = (huggingfaceApiKey: string) => {
  const { manager } = useCellData(huggingfaceApiKey);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const segmentCell = useCallback(async (imageBlob: Blob) => {
    setLoading(true); setError(null);
    try {
      const r = await manager.getHuggingFace().segmentCell(imageBlob);
      setResult(r); return r;
    } catch (err) { setError(err as Error); throw err; }
    finally { setLoading(false); }
  }, [manager]);

  const classifyCell = useCallback(async (imageBlob: Blob) => {
    setLoading(true); setError(null);
    try {
      const r = await manager.getHuggingFace().classifyCell(imageBlob);
      setResult(r); return r;
    } catch (err) { setError(err as Error); throw err; }
    finally { setLoading(false); }
  }, [manager]);

  const detectCells = useCallback(async (imageBlob: Blob) => {
    setLoading(true); setError(null);
    try {
      const r = await manager.getHuggingFace().detectCells(imageBlob);
      setResult(r); return r;
    } catch (err) { setError(err as Error); throw err; }
    finally { setLoading(false); }
  }, [manager]);

  return { segmentCell, classifyCell, detectCells, result, loading, error };
};

export const useMeshData = (huggingfaceApiKey: string) => {
  const { manager } = useCellData(huggingfaceApiKey);
  const [meshData, setMeshData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const loadSTL = useCallback(async (file: File) => {
    setLoading(true); setError(null);
    try {
      const buffer = await file.arrayBuffer();
      const mesh = manager.getMeshProcessor().parseSTL(buffer);
      setMeshData(mesh); return mesh;
    } catch (err) { setError(err as Error); throw err; }
    finally { setLoading(false); }
  }, [manager]);

  const loadOBJ = useCallback(async (file: File) => {
    setLoading(true); setError(null);
    try {
      const text = await file.text();
      const mesh = manager.getMeshProcessor().parseOBJ(text);
      setMeshData(mesh); return mesh;
    } catch (err) { setError(err as Error); throw err; }
    finally { setLoading(false); }
  }, [manager]);

  return { meshData, loadSTL, loadOBJ, loading, error };
};

export const useIDRData = (huggingfaceApiKey: string) => {
  const { manager } = useCellData(huggingfaceApiKey);
  const [projects, setProjects] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const loadProjects = useCallback(async () => {
    setLoading(true); setError(null);
    try {
      const data = await manager.getIDR().getProjects();
      setProjects(data); return data;
    } catch (err) { setError(err as Error); throw err; }
    finally { setLoading(false); }
  }, [manager]);

  return { projects, loadProjects, loading, error };
};
