# Helicopter API Reference

This document provides comprehensive documentation for all Helicopter APIs including the Python API, REST API, WebSocket API, and GraphQL API.

## Python API

### Core Classes

#### `HelicopterPipeline`

Main entry point for the Helicopter framework.

```python
from helicopter import HelicopterPipeline

class HelicopterPipeline:
    """Main pipeline for visual knowledge extraction."""
    
    def __init__(
        self,
        diffusion_model: str = "stabilityai/stable-diffusion-2-1",
        llm_model: str = "microsoft/DialoGPT-medium",
        vision_encoder: str = "openai/clip-vit-large-patch14",
        device: str = "auto",
        precision: str = "fp16"
    ) -> None:
        """Initialize the Helicopter pipeline."""
```

**Methods**:

##### `extract_visual_tokens()`
```python
def extract_visual_tokens(
    self,
    image: Union[str, PIL.Image.Image, np.ndarray, torch.Tensor],
    regions: Optional[List[Dict[str, Any]]] = None,
    return_attention: bool = False,
    text_condition: Optional[str] = None
) -> VisualTokens:
    """
    Extract visual tokens from an image.
    
    Args:
        image: Input image in various formats
        regions: List of region definitions for regional analysis
        return_attention: Whether to return attention maps
        text_condition: Optional text condition for guided extraction
        
    Returns:
        VisualTokens object containing extracted tokens and metadata
        
    Example:
        >>> pipeline = HelicopterPipeline()
        >>> tokens = pipeline.extract_visual_tokens("medical_scan.jpg")
        >>> print(f"Extracted {len(tokens)} visual tokens")
    """
```

##### `analyze_image()`
```python
def analyze_image(
    self,
    image: Union[str, PIL.Image.Image, np.ndarray],
    goal: str,
    domain: Optional[str] = None,
    quality_threshold: float = 0.8
) -> AnalysisResult:
    """
    Perform comprehensive image analysis with metacognitive orchestration.
    
    Args:
        image: Input image
        goal: High-level analysis goal
        domain: Domain context (medical, biomechanics, etc.)
        quality_threshold: Minimum quality score for acceptance
        
    Returns:
        AnalysisResult with extracted knowledge and metadata
        
    Example:
        >>> result = pipeline.analyze_image(
        ...     "chest_xray.jpg",
        ...     goal="Extract diagnostic features",
        ...     domain="medical"
        ... )
        >>> print(result.summary)
    """
```

##### `train_domain_model()`
```python
def train_domain_model(
    self,
    dataset_path: str,
    domain: str,
    base_model: str = "gpt2",
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    output_dir: str = "./models"
) -> TrainingResult:
    """
    Train a domain-specific LLM from visual knowledge.
    
    Args:
        dataset_path: Path to image dataset
        domain: Target domain for specialization
        base_model: Base language model to fine-tune
        epochs: Training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        output_dir: Directory to save trained model
        
    Returns:
        TrainingResult with metrics and model path
        
    Example:
        >>> result = pipeline.train_domain_model(
        ...     "data/medical_images/",
        ...     domain="radiology",
        ...     epochs=20
        ... )
        >>> print(f"Model saved to: {result.model_path}")
    """
```

#### `VisualTokenizer`

Core visual tokenization engine.

```python
from helicopter.core import VisualTokenizer

class VisualTokenizer:
    """Convert images to visual tokens."""
    
    def __init__(
        self,
        vision_encoder: str = "openai/clip-vit-large-patch14",
        diffusion_model: str = "stabilityai/stable-diffusion-2-1",
        patch_size: int = 16,
        max_tokens: int = 2048
    ) -> None:
        """Initialize visual tokenizer."""
```

**Methods**:

##### `tokenize_image()`
```python
def tokenize_image(
    self,
    image: Union[PIL.Image.Image, np.ndarray, torch.Tensor],
    regions: Optional[List[Dict]] = None,
    return_attention: bool = False
) -> VisualTokens:
    """
    Tokenize a single image.
    
    Returns:
        VisualTokens with the following attributes:
        - tokens: torch.Tensor of shape (num_tokens, token_dim)
        - metadata: Dict with tokenization metadata
        - attention_maps: Optional attention visualizations
        - regions: Regional token mappings
    """
```

##### `tokenize_batch()`
```python
def tokenize_batch(
    self,
    images: List[Union[PIL.Image.Image, np.ndarray]],
    batch_size: int = 8
) -> List[VisualTokens]:
    """Tokenize multiple images efficiently."""
```

#### `RegionalAnalyzer`

Regional analysis capabilities.

```python
from helicopter.utils import RegionalAnalyzer

class RegionalAnalyzer:
    """Analyze specific regions of images."""
    
    def extract_regional_tokens(
        self,
        image: Union[str, PIL.Image.Image],
        regions: List[Dict[str, Any]],
        specialized_models: Optional[Dict[str, str]] = None
    ) -> Dict[str, VisualTokens]:
        """
        Extract tokens from specific image regions.
        
        Args:
            image: Input image
            regions: List of region definitions
            specialized_models: Optional domain-specific models
            
        Returns:
            Dictionary mapping region names to visual tokens
            
        Example:
            >>> analyzer = RegionalAnalyzer(tokenizer)
            >>> regions = [
            ...     {"name": "lesion", "bbox": [100, 100, 200, 200]},
            ...     {"name": "normal", "bbox": [300, 300, 400, 400]}
            ... ]
            >>> tokens = analyzer.extract_regional_tokens(image, regions)
        """
```

#### `DomainTrainer`

Training domain-specific models.

```python
from helicopter.training import DomainTrainer

class DomainTrainer:
    """Train domain-specific visual LLMs."""
    
    def __init__(
        self,
        base_model: str,
        domain: str,
        device: str = "auto"
    ) -> None:
        """Initialize domain trainer."""
    
    def prepare_dataset(
        self,
        image_dir: str,
        annotations: Optional[str] = None,
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    ) -> VisualDataset:
        """Prepare visual dataset for training."""
    
    def train(
        self,
        dataset: VisualDataset,
        **training_args
    ) -> TrainedModel:
        """Execute training pipeline."""
```

### Data Classes

#### `VisualTokens`

```python
from dataclasses import dataclass
from typing import Dict, Any, Optional
import torch

@dataclass
class VisualTokens:
    """Container for visual tokens and metadata."""
    
    tokens: torch.Tensor  # Shape: (num_tokens, token_dim)
    metadata: Dict[str, Any]
    attention_maps: Optional[torch.Tensor] = None
    regional_tokens: Optional[Dict[str, torch.Tensor]] = None
    confidence_scores: Optional[torch.Tensor] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        
    def save(self, filepath: str) -> None:
        """Save visual tokens to file."""
        
    @classmethod
    def load(cls, filepath: str) -> 'VisualTokens':
        """Load visual tokens from file."""
```

#### `AnalysisResult`

```python
@dataclass
class AnalysisResult:
    """Result of comprehensive image analysis."""
    
    image_path: str
    goal: str
    domain: str
    visual_tokens: VisualTokens
    extracted_knowledge: Dict[str, Any]
    quality_score: float
    processing_time: float
    metadata: Dict[str, Any]
    
    @property
    def summary(self) -> str:
        """Human-readable summary of analysis."""
        
    def to_json(self) -> str:
        """Convert to JSON for storage/transmission."""
        
    def visualize(self, save_path: Optional[str] = None) -> None:
        """Create visualization of analysis results."""
```

### Exception Classes

```python
class HelicopterError(Exception):
    """Base exception for Helicopter framework."""
    pass

class TokenizationError(HelicopterError):
    """Error during visual tokenization."""
    pass

class ModelLoadError(HelicopterError):
    """Error loading models."""
    pass

class TrainingError(HelicopterError):
    """Error during model training."""
    pass

class QualityError(HelicopterError):
    """Analysis quality below threshold."""
    pass
```

## REST API

The Helicopter REST API provides web-based access to all framework functionality.

### Base URL
```
http://localhost:8000/api/v1
```

### Authentication

API key authentication (optional):
```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     https://api.helicopter.ai/v1/tokenize
```

### Endpoints

#### `POST /tokenize`

Extract visual tokens from an uploaded image.

**Request**:
```bash
curl -X POST "http://localhost:8000/api/v1/tokenize" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@image.jpg" \
     -F "regions=[{\"name\":\"roi\",\"bbox\":[0,0,100,100]}]" \
     -F "return_attention=true"
```

**Request Parameters**:
- `file` (required): Image file (multipart/form-data)
- `regions` (optional): JSON string of region definitions
- `return_attention` (optional): Boolean for attention maps
- `text_condition` (optional): Text condition for guided extraction

**Response**:
```json
{
    "status": "success",
    "token_count": 1024,
    "tokens": {
        "shape": [1024, 768],
        "dtype": "float32",
        "data_url": "http://localhost:8000/api/v1/tokens/abc123"
    },
    "metadata": {
        "image_size": [512, 512],
        "patch_size": 16,
        "model_used": "openai/clip-vit-large-patch14",
        "processing_time": 2.34
    },
    "attention_maps": {
        "data_url": "http://localhost:8000/api/v1/attention/abc123"
    },
    "regional_tokens": {
        "roi": {
            "token_count": 64,
            "data_url": "http://localhost:8000/api/v1/tokens/abc123/roi"
        }
    }
}
```

#### `POST /analyze`

Perform comprehensive image analysis.

**Request**:
```bash
curl -X POST "http://localhost:8000/api/v1/analyze" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@medical_scan.jpg" \
     -F "goal=Extract diagnostic features" \
     -F "domain=medical" \
     -F "quality_threshold=0.8"
```

**Request Parameters**:
- `file` (required): Image file
- `goal` (required): Analysis goal description
- `domain` (optional): Domain context
- `quality_threshold` (optional): Minimum quality score (default: 0.8)

**Response**:
```json
{
    "status": "success",
    "analysis_id": "analysis_abc123",
    "goal": "Extract diagnostic features",
    "domain": "medical",
    "quality_score": 0.92,
    "processing_time": 5.67,
    "extracted_knowledge": {
        "findings": [
            "Potential opacity in right lower lobe",
            "Normal cardiac silhouette",
            "Clear lung fields otherwise"
        ],
        "confidence_scores": [0.85, 0.95, 0.88],
        "regions_analyzed": ["lungs", "heart", "chest_wall"]
    },
    "visual_tokens": {
        "data_url": "http://localhost:8000/api/v1/tokens/analysis_abc123"
    },
    "visualization": {
        "image_url": "http://localhost:8000/api/v1/visualizations/analysis_abc123.png"
    }
}
```

#### `POST /train`

Start training a domain-specific model.

**Request**:
```bash
curl -X POST "http://localhost:8000/api/v1/train" \
     -H "Content-Type: application/json" \
     -d '{
       "dataset_path": "/path/to/dataset",
       "domain": "radiology",
       "base_model": "gpt2",
       "training_config": {
         "epochs": 10,
         "batch_size": 8,
         "learning_rate": 1e-4
       }
     }'
```

**Response**:
```json
{
    "status": "training_started",
    "job_id": "train_job_xyz789",
    "estimated_duration": "2.5 hours",
    "progress_url": "http://localhost:8000/api/v1/training/train_job_xyz789/progress"
}
```

#### `GET /training/{job_id}/progress`

Get training progress.

**Response**:
```json
{
    "job_id": "train_job_xyz789",
    "status": "training", // queued, training, completed, failed
    "progress": 0.45,
    "current_epoch": 4,
    "total_epochs": 10,
    "current_loss": 0.234,
    "estimated_time_remaining": "1.2 hours",
    "metrics": {
        "train_loss": 0.234,
        "val_loss": 0.267,
        "train_accuracy": 0.876,
        "val_accuracy": 0.843
    }
}
```

#### `GET /models`

List available models.

**Response**:
```json
{
    "models": {
        "base_models": [
            "gpt2", "gpt2-medium", "bert-base-uncased", "t5-small"
        ],
        "vision_encoders": [
            "openai/clip-vit-large-patch14",
            "facebook/dinov2-large",
            "google/siglip-large-patch16-384"
        ],
        "diffusion_models": [
            "stabilityai/stable-diffusion-2-1",
            "runwayml/stable-diffusion-v1-5"
        ],
        "trained_models": [
            {
                "name": "medical_radiology_v1",
                "domain": "medical",
                "created": "2024-01-15T10:30:00Z",
                "accuracy": 0.94,
                "model_url": "/api/v1/models/medical_radiology_v1"
            }
        ]
    }
}
```

#### `POST /chat`

Chat with a trained domain model.

**Request**:
```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@patient_scan.jpg" \
     -F "message=What abnormalities do you see?" \
     -F "model=medical_radiology_v1"
```

**Response**:
```json
{
    "response": "I can identify a potential mass in the upper right quadrant of the chest X-ray. The opacity appears to be approximately 2.3cm in diameter with irregular borders, which may warrant further investigation. The cardiac silhouette appears normal, and the remaining lung fields are clear.",
    "confidence": 0.87,
    "regions_highlighted": [
        {
            "name": "potential_mass",
            "bbox": [245, 123, 298, 176],
            "confidence": 0.85
        }
    ],
    "metadata": {
        "model_used": "medical_radiology_v1",
        "processing_time": 1.23,
        "visual_tokens_analyzed": 1024
    }
}
```

### Error Responses

All endpoints return standardized error responses:

```json
{
    "status": "error",
    "error_code": "TOKENIZATION_FAILED",
    "message": "Failed to extract visual tokens from image",
    "details": {
        "reason": "Unsupported image format",
        "supported_formats": ["JPEG", "PNG", "TIFF", "BMP"]
    },
    "request_id": "req_abc123"
}
```

**Common Error Codes**:
- `INVALID_IMAGE`: Image format not supported or corrupted
- `TOKENIZATION_FAILED`: Error during visual tokenization
- `MODEL_NOT_FOUND`: Requested model not available
- `QUALITY_TOO_LOW`: Analysis quality below threshold
- `TRAINING_FAILED`: Model training encountered error
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `INSUFFICIENT_RESOURCES`: Not enough compute resources

## WebSocket API

Real-time processing via WebSocket connections.

### Connection URL
```
ws://localhost:8000/ws/v1/process
```

### Connection Example

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/v1/process');

ws.onopen = function(event) {
    console.log('Connected to Helicopter WebSocket');
    
    // Send configuration
    ws.send(JSON.stringify({
        type: 'configure',
        config: {
            model: 'openai/clip-vit-large-patch14',
            batch_size: 4,
            return_attention: true
        }
    }));
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    switch(data.type) {
        case 'tokenization_result':
            console.log('Received tokens:', data.tokens);
            break;
        case 'progress':
            console.log('Progress:', data.progress);
            break;
        case 'error':
            console.error('Error:', data.message);
            break;
    }
};
```

### Message Types

#### Configuration Message
```json
{
    "type": "configure",
    "config": {
        "model": "openai/clip-vit-large-patch14",
        "batch_size": 4,
        "return_attention": true,
        "quality_threshold": 0.8
    }
}
```

#### Image Processing Message
```json
{
    "type": "process_image",
    "image_data": "base64_encoded_image_data",
    "metadata": {
        "filename": "image.jpg",
        "timestamp": "2024-01-15T10:30:00Z"
    },
    "regions": [
        {"name": "roi", "bbox": [0, 0, 100, 100]}
    ]
}
```

#### Batch Processing Message
```json
{
    "type": "process_batch",
    "images": [
        {
            "id": "img1",
            "data": "base64_encoded_image_data_1",
            "metadata": {"filename": "img1.jpg"}
        },
        {
            "id": "img2", 
            "data": "base64_encoded_image_data_2",
            "metadata": {"filename": "img2.jpg"}
        }
    ]
}
```

### Response Messages

#### Tokenization Result
```json
{
    "type": "tokenization_result",
    "image_id": "img1",
    "tokens": {
        "count": 1024,
        "data": "base64_encoded_token_data"
    },
    "metadata": {
        "processing_time": 1.23,
        "model_used": "openai/clip-vit-large-patch14"
    },
    "attention_maps": "base64_encoded_attention_data"
}
```

#### Progress Update
```json
{
    "type": "progress",
    "current": 5,
    "total": 10,
    "progress": 0.5,
    "eta_seconds": 12.5
}
```

#### Error Message
```json
{
    "type": "error",
    "error_code": "PROCESSING_FAILED",
    "message": "Failed to process image",
    "image_id": "img1"
}
```

## GraphQL API

Flexible query interface for complex data retrieval.

### Endpoint
```
http://localhost:8000/graphql
```

### Schema Overview

```graphql
type Query {
    models: [Model!]!
    analysis(id: ID!): Analysis
    analyses(domain: String, limit: Int): [Analysis!]!
    visualTokens(id: ID!): VisualTokens
    trainingJobs: [TrainingJob!]!
}

type Mutation {
    tokenizeImage(input: TokenizeImageInput!): TokenizationResult!
    analyzeImage(input: AnalyzeImageInput!): Analysis!
    startTraining(input: TrainingInput!): TrainingJob!
    cancelTraining(jobId: ID!): Boolean!
}

type Subscription {
    trainingProgress(jobId: ID!): TrainingProgress!
    realTimeAnalysis: Analysis!
}
```

### Example Queries

#### Get Available Models
```graphql
query GetModels {
    models {
        name
        type
        domain
        accuracy
        createdAt
    }
}
```

#### Tokenize Image
```graphql
mutation TokenizeImage($input: TokenizeImageInput!) {
    tokenizeImage(input: $input) {
        tokenCount
        tokens {
            data
            shape
        }
        metadata {
            processingTime
            modelUsed
        }
        attentionMaps {
            data
        }
    }
}
```

Variables:
```json
{
    "input": {
        "imageData": "base64_encoded_image",
        "regions": [
            {"name": "roi", "bbox": [0, 0, 100, 100]}
        ],
        "returnAttention": true
    }
}
```

#### Get Analysis History
```graphql
query GetAnalyses($domain: String, $limit: Int) {
    analyses(domain: $domain, limit: $limit) {
        id
        goal
        domain
        qualityScore
        createdAt
        extractedKnowledge {
            summary
            confidence
        }
        visualTokens {
            tokenCount
        }
    }
}
```

#### Subscribe to Training Progress
```graphql
subscription TrainingProgress($jobId: ID!) {
    trainingProgress(jobId: $jobId) {
        jobId
        status
        progress
        currentEpoch
        totalEpochs
        metrics {
            trainLoss
            valLoss
            trainAccuracy
            valAccuracy
        }
        estimatedTimeRemaining
    }
}
```

## SDK Examples

### Python SDK

```python
from helicopter import HelicopterClient

# Initialize client
client = HelicopterClient(
    base_url="http://localhost:8000",
    api_key="your_api_key"
)

# Tokenize image
tokens = client.tokenize_image(
    "path/to/image.jpg",
    regions=[{"name": "roi", "bbox": [0, 0, 100, 100]}]
)

# Analyze image
result = client.analyze_image(
    "medical_scan.jpg",
    goal="Extract diagnostic features",
    domain="medical"
)

# Start training
training_job = client.start_training(
    dataset_path="data/medical_images/",
    domain="radiology",
    base_model="gpt2"
)

# Monitor training
for progress in client.watch_training(training_job.id):
    print(f"Progress: {progress.progress:.1%}")
    if progress.status == "completed":
        break
```

### JavaScript SDK

```javascript
import { HelicopterClient } from '@helicopter/js-sdk';

const client = new HelicopterClient({
    baseUrl: 'http://localhost:8000',
    apiKey: 'your_api_key'
});

// Tokenize image
const tokens = await client.tokenizeImage(imageFile, {
    regions: [{ name: 'roi', bbox: [0, 0, 100, 100] }],
    returnAttention: true
});

// Analyze image
const result = await client.analyzeImage(imageFile, {
    goal: 'Extract diagnostic features',
    domain: 'medical'
});

// Real-time processing
const ws = client.createWebSocket();
ws.on('tokenization_result', (data) => {
    console.log('Received tokens:', data.tokens);
});
ws.sendImage(imageFile);
```

## Rate Limiting

API endpoints are rate limited to ensure fair usage:

- **Free tier**: 100 requests/hour
- **Basic tier**: 1,000 requests/hour  
- **Premium tier**: 10,000 requests/hour
- **Enterprise**: Unlimited

Rate limit headers:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 995
X-RateLimit-Reset: 1640995200
```

## Monitoring and Analytics

### Health Check Endpoint

```bash
GET /health
```

Response:
```json
{
    "status": "healthy",
    "version": "0.1.0",
    "uptime": 3600,
    "dependencies": {
        "gpu": "available",
        "models": "loaded",
        "storage": "healthy"
    }
}
```

### Metrics Endpoint

```bash
GET /metrics
```

Returns Prometheus-formatted metrics for monitoring integration.
