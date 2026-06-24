// hooks/useCellData.ts
import { useState, useEffect, useCallback } from 'react';
import { createCellDataManager } from './cellDataService';

export const useCellData = (huggingfaceApiKey: string) => {
  const [manager] = useState(() => createCellDataManager(huggingfaceApiKey));
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  return { manager, loading, error, setLoading, setError };
};

/**
 * Hook for loading Allen Cell data
 */
export const useAllenCell = (cellId: string | null, huggingfaceApiKey: string) => {
  const [cellData, setCellData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const { manager } = useCellData(huggingfaceApiKey);

  useEffect(() => {
    if (!cellId) return;

    const loadCell = async () => {
      setLoading(true);
      setError(null);
      try {
        const data = await manager.getAllenCell().getCellById(cellId);
        setCellData(data);
      } catch (err) {
        setError(err as Error);
      } finally {
        setLoading(false);
      }
    };

    loadCell();
  }, [cellId, manager]);

  return { cellData, loading, error };
};

/**
 * Hook for HuggingFace model inference
 */
export const useHuggingFaceModel = (huggingfaceApiKey: string) => {
  const { manager } = useCellData(huggingfaceApiKey);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const segmentCell = useCallback(async (imageBlob: Blob) => {
    setLoading(true);
    setError(null);
    try {
      const segmentation = await manager.getHuggingFace().segmentCell(imageBlob);
      setResult(segmentation);
      return segmentation;
    } catch (err) {
      setError(err as Error);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [manager]);

  const classifyCell = useCallback(async (imageBlob: Blob) => {
    setLoading(true);
    setError(null);
    try {
      const classification = await manager.getHuggingFace().classifyCell(imageBlob);
      setResult(classification);
      return classification;
    } catch (err) {
      setError(err as Error);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [manager]);

  const detectCells = useCallback(async (imageBlob: Blob) => {
    setLoading(true);
    setError(null);
    try {
      const detection = await manager.getHuggingFace().detectCells(imageBlob);
      setResult(detection);
      return detection;
    } catch (err) {
      setError(err as Error);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [manager]);

  return { segmentCell, classifyCell, detectCells, result, loading, error };
};

/**
 * Hook for mesh data loading
 */
export const useMeshData = (huggingfaceApiKey: string) => {
  const { manager } = useCellData(huggingfaceApiKey);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [meshData, setMeshData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const loadSTL = useCallback(async (file: File) => {
    setLoading(true);
    setError(null);
    try {
      const buffer = await file.arrayBuffer();
      const mesh = manager.getMeshProcessor().parseSTL(buffer);
      setMeshData(mesh);
      return mesh;
    } catch (err) {
      setError(err as Error);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [manager]);

  const loadOBJ = useCallback(async (file: File) => {
    setLoading(true);
    setError(null);
    try {
      const text = await file.text();
      const mesh = manager.getMeshProcessor().parseOBJ(text);
      setMeshData(mesh);
      return mesh;
    } catch (err) {
      setError(err as Error);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [manager]);

  const loadFromURL = useCallback(async (url: string, format: 'stl' | 'obj') => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(url);
      if (format === 'stl') {
        const buffer = await response.arrayBuffer();
        const mesh = manager.getMeshProcessor().parseSTL(buffer);
        setMeshData(mesh);
        return mesh;
      } else {
        const text = await response.text();
        const mesh = manager.getMeshProcessor().parseOBJ(text);
        setMeshData(mesh);
        return mesh;
      }
    } catch (err) {
      setError(err as Error);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [manager]);

  return { meshData, loadSTL, loadOBJ, loadFromURL, loading, error };
};

/**
 * Hook for IDR data
 */
export const useIDRData = (huggingfaceApiKey: string) => {
  const { manager } = useCellData(huggingfaceApiKey);
  const [projects, setProjects] = useState([]);
  const [images, setImages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const loadProjects = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await manager.getIDR().getProjects();
      setProjects(data);
      return data;
    } catch (err) {
      setError(err as Error);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [manager]);

  const loadImages = useCallback(async (projectId: string) => {
    setLoading(true);
    setError(null);
    try {
      const data = await manager.getIDR().getImages(projectId);
      setImages(data);
      return data;
    } catch (err) {
      setError(err as Error);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [manager]);

  return { projects, images, loadProjects, loadImages, loading, error };
};

/**
 * Hook for cell image search
 */
export const useCellImageSearch = (huggingfaceApiKey: string) => {
  const { manager } = useCellData(huggingfaceApiKey);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const search = useCallback(async (query: string, limit = 10) => {
    setLoading(true);
    setError(null);
    try {
      const data = await manager.getCellImageLibrary().searchImages(query, limit);
      setResults(data);
      return data;
    } catch (err) {
      setError(err as Error);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [manager]);

  return { results, search, loading, error };
};