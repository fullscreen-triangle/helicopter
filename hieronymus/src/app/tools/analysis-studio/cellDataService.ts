// services/cellDataService.ts

/**
 * Allen Cell Explorer API Service
 */
export class AllenCellService {
  private baseUrl = 'https://api.allencell.org/v1';
  
  async getCellById(cellId: string) {
    try {
      const response = await fetch(`${this.baseUrl}/cells/${cellId}`);
      if (!response.ok) throw new Error('Failed to fetch cell data');
      return await response.json();
    } catch (error) {
      console.error('Allen Cell API Error:', error);
      throw error;
    }
  }
  
  async getCellCollection(params: {
    limit?: number;
    offset?: number;
    cellLine?: string;
  }) {
    const queryParams = new URLSearchParams(params as any);
    const response = await fetch(`${this.baseUrl}/cells?${queryParams}`);
    return await response.json();
  }
  
  async getSegmentationData(cellId: string) {
    const response = await fetch(`${this.baseUrl}/cells/${cellId}/segmentation`);
    return await response.json();
  }
  
  async downloadMeshData(meshUrl: string): Promise<ArrayBuffer> {
    const response = await fetch(meshUrl);
    return await response.arrayBuffer();
  }
}

/**
 * Cell Image Library Service
 */
export class CellImageLibraryService {
  private baseUrl = 'http://www.cellimagelibrary.org/api';
  
  async searchImages(query: string, limit = 10) {
    try {
      const response = await fetch(
        `${this.baseUrl}/search?q=${encodeURIComponent(query)}&limit=${limit}`
      );
      return await response.json();
    } catch (error) {
      console.error('Cell Image Library Error:', error);
      throw error;
    }
  }
  
  async getImageMetadata(imageId: string) {
    const response = await fetch(`${this.baseUrl}/images/${imageId}`);
    return await response.json();
  }
}

/**
 * IDR (Image Data Resource) Service
 */
export class IDRService {
  private baseUrl = 'https://idr.openmicroscopy.org/api/v0';
  
  async getProjects() {
    const response = await fetch(`${this.baseUrl}/m/projects/`);
    return await response.json();
  }
  
  async getImages(projectId: string) {
    const response = await fetch(`${this.baseUrl}/m/projects/${projectId}/images/`);
    return await response.json();
  }
  
  async getImageData(imageId: string) {
    const response = await fetch(`${this.baseUrl}/m/images/${imageId}/`);
    return await response.json();
  }
  
  async getThumbnail(imageId: string, size = 256) {
    return `${this.baseUrl}/m/images/${imageId}/thumbnail/${size}/`;
  }
}

/**
 * OME-Zarr Data Loader
 */
export class ZarrDataService {
  async loadZarrData(url: string) {
    try {
      // Using zarr.js for loading OME-Zarr data
      const { openArray } = await import('zarr');
      const store = await fetch(url).then(r => r.arrayBuffer());
      return await openArray({ store });
    } catch (error) {
      console.error('Zarr loading error:', error);
      throw error;
    }
  }
  
  async loadMultiscaleData(zarrUrl: string, level = 0) {
    // Load specific resolution level from multiscale OME-Zarr
    const response = await fetch(`${zarrUrl}/${level}`);
    return await response.arrayBuffer();
  }
}

/**
 * HuggingFace Model Service
 */
export class HuggingFaceService {
  private apiKey: string;
  private baseUrl = 'https://api-inference.huggingface.co/models';
  
  constructor(apiKey: string) {
    this.apiKey = apiKey;
  }
  
  private async query(modelId: string, data: any) {
    const response = await fetch(`${this.baseUrl}/${modelId}`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });
    
    if (!response.ok) {
      throw new Error(`HuggingFace API Error: ${response.statusText}`);
    }
    
    return await response.json();
  }
  
  /**
   * Cell Segmentation using HuggingFace models
   */
  async segmentCell(imageData: Blob, modelId = 'facebook/mask2former-swin-large-coco-instance') {
    const formData = new FormData();
    formData.append('file', imageData);
    
    const response = await fetch(`${this.baseUrl}/${modelId}`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
      },
      body: formData,
    });
    
    return await response.json();
  }
  
  /**
   * Cell Classification
   */
  async classifyCell(imageData: Blob, modelId = 'microsoft/resnet-50') {
    const formData = new FormData();
    formData.append('file', imageData);
    
    const response = await fetch(`${this.baseUrl}/${modelId}`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
      },
      body: formData,
    });
    
    return await response.json();
  }
  
  /**
   * Feature Extraction for cell analysis
   */
  async extractFeatures(imageData: Blob, modelId = 'facebook/dinov2-base') {
    const formData = new FormData();
    formData.append('file', imageData);
    
    const response = await fetch(`${this.baseUrl}/${modelId}`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
      },
      body: formData,
    });
    
    return await response.json();
  }
  
  /**
   * Cell Detection using object detection models
   */
  async detectCells(imageData: Blob, modelId = 'facebook/detr-resnet-50') {
    const formData = new FormData();
    formData.append('file', imageData);
    
    const response = await fetch(`${this.baseUrl}/${modelId}`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
      },
      body: formData,
    });
    
    return await response.json();
  }
}

/**
 * Mesh Processing Service
 */
export class MeshProcessingService {
  /**
   * Convert point cloud to mesh data
   */
  convertPointCloudToMesh(points: Float32Array, indices?: Uint32Array) {
    return {
      positions: points,
      indices: indices || this.generateIndices(points.length / 3),
    };
  }
  
  private generateIndices(vertexCount: number): Uint32Array {
    const indices = new Uint32Array((vertexCount - 2) * 3);
    for (let i = 0; i < vertexCount - 2; i++) {
      indices[i * 3] = 0;
      indices[i * 3 + 1] = i + 1;
      indices[i * 3 + 2] = i + 2;
    }
    return indices;
  }
  
  /**
   * Parse STL file format
   */
  parseSTL(buffer: ArrayBuffer) {
    const view = new DataView(buffer);
    const isASCII = this.isASCIISTL(buffer);
    
    if (isASCII) {
      return this.parseASCIISTL(buffer);
    } else {
      return this.parseBinarySTL(view);
    }
  }
  
  private isASCIISTL(buffer: ArrayBuffer): boolean {
    const text = new TextDecoder().decode(buffer.slice(0, 5));
    return text === 'solid';
  }
  
  private parseASCIISTL(buffer: ArrayBuffer) {
    const text = new TextDecoder().decode(buffer);
    const vertices: number[] = [];
    const normals: number[] = [];
    
    const vertexPattern = /vertex\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)/g;
    const normalPattern = /normal\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)/g;
    
    let match;
    while ((match = vertexPattern.exec(text)) !== null) {
      vertices.push(parseFloat(match[1]), parseFloat(match[2]), parseFloat(match[3]));
    }
    
    while ((match = normalPattern.exec(text)) !== null) {
      normals.push(parseFloat(match[1]), parseFloat(match[2]), parseFloat(match[3]));
    }
    
    return {
      positions: new Float32Array(vertices),
      normals: new Float32Array(normals),
    };
  }
  
  private parseBinarySTL(view: DataView) {
    const triangles = view.getUint32(80, true);
    const vertices: number[] = [];
    const normals: number[] = [];
    
    for (let i = 0; i < triangles; i++) {
      const offset = 84 + i * 50;
      
      // Normal
      const nx = view.getFloat32(offset, true);
      const ny = view.getFloat32(offset + 4, true);
      const nz = view.getFloat32(offset + 8, true);
      
      // Vertices
      for (let j = 0; j < 3; j++) {
        const vOffset = offset + 12 + j * 12;
        vertices.push(
          view.getFloat32(vOffset, true),
          view.getFloat32(vOffset + 4, true),
          view.getFloat32(vOffset + 8, true)
        );
        normals.push(nx, ny, nz);
      }
    }
    
    return {
      positions: new Float32Array(vertices),
      normals: new Float32Array(normals),
    };
  }
  
  /**
   * Parse OBJ file format
   */
  parseOBJ(text: string) {
    const vertices: number[] = [];
    const normals: number[] = [];
    const indices: number[] = [];
    
    const lines = text.split('\n');
    
    for (const line of lines) {
      const parts = line.trim().split(/\s+/);
      
      if (parts[0] === 'v') {
        vertices.push(parseFloat(parts[1]), parseFloat(parts[2]), parseFloat(parts[3]));
      } else if (parts[0] === 'vn') {
        normals.push(parseFloat(parts[1]), parseFloat(parts[2]), parseFloat(parts[3]));
      } else if (parts[0] === 'f') {
        for (let i = 1; i < parts.length; i++) {
          const vertexData = parts[i].split('/');
          indices.push(parseInt(vertexData[0]) - 1);
        }
      }
    }
    
    return {
      positions: new Float32Array(vertices),
      normals: new Float32Array(normals),
      indices: new Uint32Array(indices),
    };
  }
}

/**
 * Unified Cell Data Manager
 */
export class CellDataManager {
  private allenCell: AllenCellService;
  private cellImageLib: CellImageLibraryService;
  private idr: IDRService;
  private zarr: ZarrDataService;
  private huggingface: HuggingFaceService;
  private meshProcessor: MeshProcessingService;
  
  constructor(huggingfaceApiKey: string) {
    this.allenCell = new AllenCellService();
    this.cellImageLib = new CellImageLibraryService();
    this.idr = new IDRService();
    this.zarr = new ZarrDataService();
    this.huggingface = new HuggingFaceService(huggingfaceApiKey);
    this.meshProcessor = new MeshProcessingService();
  }
  
  getAllenCell() { return this.allenCell; }
  getCellImageLibrary() { return this.cellImageLib; }
  getIDR() { return this.idr; }
  getZarr() { return this.zarr; }
  getHuggingFace() { return this.huggingface; }
  getMeshProcessor() { return this.meshProcessor; }
}

// Export singleton instance factory
export const createCellDataManager = (huggingfaceApiKey: string) => {
  return new CellDataManager(huggingfaceApiKey);
};