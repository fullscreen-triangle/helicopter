export interface EncoderInput {
  type: 'image' | 'frequencies' | 'sequence' | 'timeseries' | 'numeric';
  data: any;
}

export interface EncodedData {
  imageData: ImageData;
  metadata: {
    domain: string;
    originalSize: number;
    encodingTime: number;
  };
}
