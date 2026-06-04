// Dataset fetching utility for BBBC and other microscopy databases

export interface DatasetMetadata {
  id: string;
  name: string;
  source: string;
  description: string;
  downloadUrl: string;
  cachedPath?: string;
  fetchedAt?: number;
}

// BBBC datasets with known public URLs
export const BBBC_DATASETS: DatasetMetadata[] = [
  {
    id: 'BBBC039',
    name: 'HeLa Cells (Fluorescence)',
    source: 'BBBC',
    description: 'Human cervical cancer cells with DAPI and Actin staining',
    downloadUrl: 'https://data.broadinstitute.org/bbbc/BBBC039/',
    cachedPath: '/datasets/BBBC039_HeLa/',
  },
  {
    id: 'BBBC006',
    name: 'CHO Cells (Tubulin)',
    source: 'BBBC',
    description: 'Chinese hamster ovary cells with tubulin staining',
    downloadUrl: 'https://data.broadinstitute.org/bbbc/BBBC006/',
    cachedPath: '/datasets/BBBC006_CHO/',
  },
  {
    id: 'BBBC008',
    name: 'Drosophila Cells',
    source: 'BBBC',
    description: 'Fruit fly embryonic cells with multiple stains',
    downloadUrl: 'https://data.broadinstitute.org/bbbc/BBBC008/',
    cachedPath: '/datasets/BBBC008_Drosophila/',
  },
];

// AllenCell datasets
export const ALLEN_CELL_DATASETS: DatasetMetadata[] = [
  {
    id: 'AllenCell_3D',
    name: '3D Cell Structure',
    source: 'AllenCell',
    description: 'High-resolution 3D cell structure from volumetric imaging',
    downloadUrl: 'https://www.allencell.org/',
    cachedPath: '/datasets/AllenCell_3D/',
  },
];

// Combined dataset registry
export const ALL_DATASETS = [...BBBC_DATASETS, ...ALLEN_CELL_DATASETS];

/**
 * Check if a dataset is available locally
 */
export async function isDatasetAvailable(datasetId: string): Promise<boolean> {
  try {
    const response = await fetch(`/datasets/${datasetId}/manifest.json`, {
      method: 'HEAD',
    });
    return response.ok;
  } catch {
    return false;
  }
}

/**
 * Get available image files from a cached dataset
 */
export async function getDatasetImages(datasetId: string): Promise<string[]> {
  try {
    const response = await fetch(`/datasets/${datasetId}/manifest.json`);
    if (!response.ok) return [];

    const manifest = await response.json();
    return manifest.images || [];
  } catch (error) {
    console.error(`Failed to load dataset ${datasetId}:`, error);
    return [];
  }
}

/**
 * Fetch a specific image from a cached dataset
 */
export async function fetchDatasetImage(
  datasetId: string,
  imageName: string
): Promise<ArrayBuffer | null> {
  try {
    const response = await fetch(`/datasets/${datasetId}/${imageName}`);
    if (!response.ok) return null;

    return await response.arrayBuffer();
  } catch (error) {
    console.error(`Failed to fetch image ${imageName} from ${datasetId}:`, error);
    return null;
  }
}

/**
 * Get dataset metadata
 */
export function getDatasetMetadata(datasetId: string): DatasetMetadata | undefined {
  return ALL_DATASETS.find((d) => d.id === datasetId);
}

/**
 * Get all available datasets
 */
export function listAllDatasets(): DatasetMetadata[] {
  return ALL_DATASETS;
}
