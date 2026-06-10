// services/cellDataService.ts

export class AllenCellService {
  private baseUrl = 'https://api.allencell.org/v1';

  async getCellById(cellId: string) {
    const response = await fetch(`${this.baseUrl}/cells/${cellId}`);
    if (!response.ok) throw new Error('Failed to fetch cell data');
    return await response.json();
  }

  async getCellCollection(params: { limit?: number; offset?: number; cellLine?: string }) {
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

export class CellImageLibraryService {
  private baseUrl = 'http://www.cellimagelibrary.org/api';

  async searchImages(query: string, limit = 10) {
    const response = await fetch(
      `${this.baseUrl}/search?q=${encodeURIComponent(query)}&limit=${limit}`
    );
    return await response.json();
  }

  async getImageMetadata(imageId: string) {
    const response = await fetch(`${this.baseUrl}/images/${imageId}`);
    return await response.json();
  }
}

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

  getThumbnailUrl(imageId: string, size = 256) {
    return `${this.baseUrl}/m/images/${imageId}/thumbnail/${size}/`;
  }
}

export class HuggingFaceService {
  private apiKey: string;
  private baseUrl = 'https://api-inference.huggingface.co/models';

  constructor(apiKey: string) {
    this.apiKey = apiKey;
  }

  async segmentCell(imageData: Blob, modelId = 'facebook/mask2former-swin-large-coco-instance') {
    const formData = new FormData();
    formData.append('file', imageData);
    const response = await fetch(`${this.baseUrl}/${modelId}`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${this.apiKey}` },
      body: formData,
    });
    if (!response.ok) throw new Error(`HuggingFace API Error: ${response.statusText}`);
    return await response.json();
  }

  async classifyCell(imageData: Blob, modelId = 'microsoft/resnet-50') {
    const formData = new FormData();
    formData.append('file', imageData);
    const response = await fetch(`${this.baseUrl}/${modelId}`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${this.apiKey}` },
      body: formData,
    });
    if (!response.ok) throw new Error(`HuggingFace API Error: ${response.statusText}`);
    return await response.json();
  }

  async extractFeatures(imageData: Blob, modelId = 'facebook/dinov2-base') {
    const formData = new FormData();
    formData.append('file', imageData);
    const response = await fetch(`${this.baseUrl}/${modelId}`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${this.apiKey}` },
      body: formData,
    });
    if (!response.ok) throw new Error(`HuggingFace API Error: ${response.statusText}`);
    return await response.json();
  }

  async detectCells(imageData: Blob, modelId = 'facebook/detr-resnet-50') {
    const formData = new FormData();
    formData.append('file', imageData);
    const response = await fetch(`${this.baseUrl}/${modelId}`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${this.apiKey}` },
      body: formData,
    });
    if (!response.ok) throw new Error(`HuggingFace API Error: ${response.statusText}`);
    return await response.json();
  }
}

export class MeshProcessingService {
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

  parseSTL(buffer: ArrayBuffer) {
    const view = new DataView(buffer);
    const text5 = new TextDecoder().decode(buffer.slice(0, 5));
    if (text5 === 'solid') return this.parseASCIISTL(buffer);
    return this.parseBinarySTL(view);
  }

  private parseASCIISTL(buffer: ArrayBuffer) {
    const text = new TextDecoder().decode(buffer);
    const vertices: number[] = [];
    const normals: number[] = [];
    const vertexPattern = /vertex\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)/g;
    const normalPattern = /normal\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)/g;
    let match;
    while ((match = vertexPattern.exec(text)) !== null)
      vertices.push(parseFloat(match[1]), parseFloat(match[2]), parseFloat(match[3]));
    while ((match = normalPattern.exec(text)) !== null)
      normals.push(parseFloat(match[1]), parseFloat(match[2]), parseFloat(match[3]));
    return { positions: new Float32Array(vertices), normals: new Float32Array(normals) };
  }

  private parseBinarySTL(view: DataView) {
    const triangles = view.getUint32(80, true);
    const vertices: number[] = [];
    const normals: number[] = [];
    for (let i = 0; i < triangles; i++) {
      const offset = 84 + i * 50;
      const nx = view.getFloat32(offset, true);
      const ny = view.getFloat32(offset + 4, true);
      const nz = view.getFloat32(offset + 8, true);
      for (let j = 0; j < 3; j++) {
        const vOffset = offset + 12 + j * 12;
        vertices.push(view.getFloat32(vOffset, true), view.getFloat32(vOffset + 4, true), view.getFloat32(vOffset + 8, true));
        normals.push(nx, ny, nz);
      }
    }
    return { positions: new Float32Array(vertices), normals: new Float32Array(normals) };
  }

  parseOBJ(text: string) {
    const vertices: number[] = [];
    const normals: number[] = [];
    const indices: number[] = [];
    for (const line of text.split('\n')) {
      const parts = line.trim().split(/\s+/);
      if (parts[0] === 'v') vertices.push(parseFloat(parts[1]), parseFloat(parts[2]), parseFloat(parts[3]));
      else if (parts[0] === 'vn') normals.push(parseFloat(parts[1]), parseFloat(parts[2]), parseFloat(parts[3]));
      else if (parts[0] === 'f') {
        for (let i = 1; i < parts.length; i++)
          indices.push(parseInt(parts[i].split('/')[0]) - 1);
      }
    }
    return { positions: new Float32Array(vertices), normals: new Float32Array(normals), indices: new Uint32Array(indices) };
  }
}

export class CellDataManager {
  private allenCell: AllenCellService;
  private cellImageLib: CellImageLibraryService;
  private idr: IDRService;
  private huggingface: HuggingFaceService;
  private meshProcessor: MeshProcessingService;

  constructor(huggingfaceApiKey: string) {
    this.allenCell = new AllenCellService();
    this.cellImageLib = new CellImageLibraryService();
    this.idr = new IDRService();
    this.huggingface = new HuggingFaceService(huggingfaceApiKey);
    this.meshProcessor = new MeshProcessingService();
  }

  getAllenCell() { return this.allenCell; }
  getCellImageLibrary() { return this.cellImageLib; }
  getIDR() { return this.idr; }
  getHuggingFace() { return this.huggingface; }
  getMeshProcessor() { return this.meshProcessor; }
}

export const createCellDataManager = (apiKey: string) => new CellDataManager(apiKey);
