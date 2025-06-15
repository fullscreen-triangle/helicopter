# Helicopter Module Reference

This document provides detailed technical documentation for all modules in the Helicopter framework, including their architecture, implementation details, and usage patterns.

## Project Architecture

```
helicopter/
├── core/                      # Core visual tokenization engine
│   ├── tokenizer.py          # Main visual tokenization logic
│   ├── diffusion_inverse.py  # Reverse diffusion processing
│   ├── vision_encoder.py     # Vision encoding modules
│   ├── patch_extractor.py    # Image patch extraction
│   └── projection.py         # Visual-to-LLM projection layer
├── models/                    # Model interfaces and wrappers
│   ├── diffusion_models.py   # Diffusion model interfaces
│   ├── llm_models.py         # Language model interfaces
│   ├── vision_models.py      # Vision model interfaces
│   └── specialized/          # Domain-specific models
├── orchestration/            # Metacognitive orchestration system
│   ├── context_manager.py   # Context and state management
│   ├── planner.py           # Goal-to-task planning
│   ├── reasoner.py          # Conflict resolution and reasoning
│   ├── quality_checker.py   # Output quality validation
│   └── solver.py            # Classical optimization solver
├── training/                 # Model training and fine-tuning
│   ├── trainer.py           # Main training orchestrator
│   ├── domain_trainer.py    # Domain-specific training
│   ├── curriculum.py        # Curriculum learning
│   └── losses.py            # Custom loss functions
├── utils/                    # Utilities and helpers
│   ├── image_processing.py  # Image manipulation utilities
│   ├── regional_analysis.py # Regional processing tools
│   ├── visualization.py     # Visualization utilities
│   ├── metrics.py           # Evaluation metrics
│   └── data_handling.py     # Data loading and preprocessing
├── api/                      # Web API interfaces
│   ├── server.py            # FastAPI server
│   ├── routes/              # API route definitions
│   ├── websocket.py         # Real-time processing
│   └── middleware.py        # API middleware
├── cli/                      # Command-line interfaces
│   ├── main.py              # Main CLI application
│   ├── train.py             # Training CLI
│   └── server.py            # Server management CLI
└── experimental/             # Experimental features
    ├── nerf_integration.py  # NeRF processing
    ├── causal_reasoning.py  # Causal inference
    ├── graph_networks.py   # Graph neural networks
    └── continual_learning.py # Continual learning
```

## Core Modules

### 1. Visual Tokenizer (`helicopter.core.tokenizer`)

**Purpose**: Main entry point for converting images to visual tokens.

**Key Classes**:

#### `VisualTokenizer`
```python
class VisualTokenizer:
    """Main visual tokenization engine."""
    
    def __init__(
        self,
        vision_encoder: str = "openai/clip-vit-large-patch14",
        diffusion_model: str = "stabilityai/stable-diffusion-2-1",
        patch_size: int = 16,
        max_tokens: int = 2048
    ):
        self.vision_encoder = VisionEncoder(vision_encoder)
        self.diffusion_inverse = DiffusionInverse(diffusion_model)
        self.patch_extractor = PatchExtractor(patch_size)
        self.projection_layer = ProjectionLayer()
    
    def tokenize_image(
        self, 
        image: Union[PIL.Image, np.ndarray, torch.Tensor],
        regions: Optional[List[Dict]] = None,
        return_attention: bool = False
    ) -> VisualTokens:
        """Convert image to visual tokens."""
        
    def tokenize_batch(
        self,
        images: List[Union[PIL.Image, np.ndarray]],
        batch_size: int = 8
    ) -> List[VisualTokens]:
        """Batch tokenization for efficiency."""
```

**Usage Example**:
```python
from helicopter.core import VisualTokenizer

tokenizer = VisualTokenizer(
    vision_encoder="openai/clip-vit-large-patch14",
    diffusion_model="stabilityai/stable-diffusion-2-1"
)

# Single image tokenization
visual_tokens = tokenizer.tokenize_image("path/to/image.jpg")

# Batch processing
images = ["img1.jpg", "img2.jpg", "img3.jpg"]
batch_tokens = tokenizer.tokenize_batch(images)
```

### 2. Diffusion Inverse (`helicopter.core.diffusion_inverse`)

**Purpose**: Implements reverse diffusion processing to extract semantic information.

**Key Classes**:

#### `DiffusionInverse`
```python
class DiffusionInverse:
    """Reverse diffusion processing for semantic extraction."""
    
    def __init__(self, model_name: str, num_timesteps: int = 50):
        self.model = DiffusionModel.from_pretrained(model_name)
        self.scheduler = DDIMScheduler.from_pretrained(model_name)
        self.num_timesteps = num_timesteps
    
    def extract_denoising_trajectory(
        self,
        image: torch.Tensor,
        text_condition: Optional[str] = None
    ) -> DenoisingTrajectory:
        """Extract complete denoising trajectory from image."""
        
    def extract_noise_predictions(
        self,
        trajectory: DenoisingTrajectory
    ) -> List[torch.Tensor]:
        """Extract noise predictions at each timestep."""
        
    def extract_attention_maps(
        self,
        trajectory: DenoisingTrajectory
    ) -> List[torch.Tensor]:
        """Extract cross-attention maps for spatial understanding."""
```

**Mathematical Implementation**:
```python
def forward_diffusion_inverse(self, x0: torch.Tensor) -> List[torch.Tensor]:
    """
    Inverse diffusion process: x_0 → x_1 → ... → x_T
    
    Args:
        x0: Original image tensor
        
    Returns:
        List of noisy latents at each timestep
    """
    latents = [x0]
    
    for t in range(1, self.num_timesteps + 1):
        # Add noise according to schedule
        noise = torch.randn_like(x0)
        alpha_t = self.scheduler.alphas_cumprod[t]
        
        x_t = torch.sqrt(alpha_t) * x0 + torch.sqrt(1 - alpha_t) * noise
        latents.append(x_t)
        
        # Extract noise prediction
        with torch.no_grad():
            epsilon_pred = self.model(x_t, t)
            
    return latents, epsilon_predictions
```

### 3. Vision Encoder (`helicopter.core.vision_encoder`)

**Purpose**: Encodes image patches into latent representations.

**Key Classes**:

#### `VisionEncoder`
```python
class VisionEncoder:
    """Vision encoding for patch-based representation."""
    
    def __init__(self, model_name: str):
        self.model = self._load_model(model_name)
        self.processor = self._load_processor(model_name)
    
    def encode_patches(
        self,
        patches: torch.Tensor
    ) -> torch.Tensor:
        """Encode image patches to latent vectors."""
        
    def encode_regions(
        self,
        image: torch.Tensor,
        regions: List[Dict]
    ) -> Dict[str, torch.Tensor]:
        """Encode specific regions with metadata."""
```

**Supported Models**:
- **CLIP**: `openai/clip-vit-large-patch14`, `openai/clip-vit-base-patch32`
- **DINOv2**: `facebook/dinov2-large`, `facebook/dinov2-base`
- **EVA**: `facebook/eva-clip-large`, `facebook/eva-clip-base`
- **SigLIP**: `google/siglip-large-patch16-384`, `google/siglip-base-patch16-224`

### 4. Regional Analysis (`helicopter.utils.regional_analysis`)

**Purpose**: Specialized processing for image regions.

**Key Classes**:

#### `RegionalAnalyzer`
```python
class RegionalAnalyzer:
    """Analyze specific regions of images."""
    
    def __init__(self, tokenizer: VisualTokenizer):
        self.tokenizer = tokenizer
        self.segmentation_model = load_segmentation_model()
    
    def extract_regional_tokens(
        self,
        image: Union[str, PIL.Image],
        regions: List[Dict],
        specialized_models: Optional[Dict] = None
    ) -> Dict[str, VisualTokens]:
        """Extract tokens from specific regions."""
        
    def auto_segment_regions(
        self,
        image: Union[str, PIL.Image],
        segmentation_method: str = "sam"
    ) -> List[Dict]:
        """Automatically detect semantic regions."""
```

**Region Definition Formats**:
```python
# Bounding box format
bbox_region = {
    "name": "region_name",
    "type": "bbox",
    "coordinates": [x1, y1, x2, y2],
    "model": "specialized_model_name"
}

# Polygon format
polygon_region = {
    "name": "region_name",
    "type": "polygon", 
    "coordinates": [(x1, y1), (x2, y2), (x3, y3), ...],
    "model": "specialized_model_name"
}

# Mask format
mask_region = {
    "name": "region_name",
    "type": "mask",
    "mask": numpy_array_or_pil_image,
    "model": "specialized_model_name"
}
```

## Orchestration Modules

### 1. Context Manager (`helicopter.orchestration.context_manager`)

**Purpose**: Manages persistent state and context across analysis sessions.

#### `ContextManager`
```python
class ContextManager:
    """Manages analysis context and state persistence."""
    
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid4())
        self.state = AnalysisState()
        self.history = AnalysisHistory()
        self.relationships = RelationshipGraph()
    
    def update_context(
        self,
        analysis_result: AnalysisResult,
        metadata: Dict[str, Any]
    ) -> None:
        """Update context with new analysis results."""
        
    def get_related_analyses(
        self,
        query: str,
        similarity_threshold: float = 0.8
    ) -> List[AnalysisResult]:
        """Retrieve related previous analyses."""
        
    def save_session(self, filepath: str) -> None:
        """Persist session state to disk."""
        
    def load_session(self, filepath: str) -> None:
        """Load session state from disk."""
```

### 2. Planner (`helicopter.orchestration.planner`)

**Purpose**: Converts high-level goals into executable analysis plans.

#### `AnalysisPlanner`
```python
class AnalysisPlanner:
    """Creates and optimizes analysis plans from high-level goals."""
    
    def __init__(self, model_registry: ModelRegistry):
        self.model_registry = model_registry
        self.task_templates = load_task_templates()
        self.optimization_engine = PlanOptimizer()
    
    def create_plan(
        self,
        goal: str,
        image_metadata: Dict[str, Any],
        constraints: Optional[Dict] = None
    ) -> AnalysisPlan:
        """Create optimized analysis plan from goal."""
        
    def optimize_plan(
        self,
        plan: AnalysisPlan,
        performance_history: List[PlanExecution]
    ) -> AnalysisPlan:
        """Optimize plan based on execution history."""
```

**Plan Structure**:
```python
@dataclass
class AnalysisPlan:
    """Represents an executable analysis plan."""
    
    id: str
    goal: str
    steps: List[AnalysisStep]
    dependencies: Dict[str, List[str]]
    estimated_duration: float
    resource_requirements: ResourceRequirements
    
@dataclass 
class AnalysisStep:
    """Individual step in analysis plan."""
    
    id: str
    task_type: str
    model_name: str
    parameters: Dict[str, Any]
    input_dependencies: List[str]
    output_schema: Dict[str, Any]
```

### 3. Quality Checker (`helicopter.orchestration.quality_checker`)

**Purpose**: Validates analysis outputs and provides quality scores.

#### `QualityChecker`
```python
class QualityChecker:
    """Validates and scores analysis quality."""
    
    def __init__(self):
        self.validators = {
            "consistency": ConsistencyValidator(),
            "completeness": CompletenessValidator(),
            "accuracy": AccuracyValidator(),
            "alignment": AlignmentValidator()
        }
    
    def evaluate_analysis(
        self,
        result: AnalysisResult,
        ground_truth: Optional[Dict] = None
    ) -> QualityScore:
        """Comprehensive quality evaluation."""
        
    def validate_visual_tokens(
        self,
        tokens: VisualTokens,
        original_image: torch.Tensor
    ) -> ValidationResult:
        """Validate visual token quality."""
```

## Model Interface Modules

### 1. LLM Models (`helicopter.models.llm_models`)

**Purpose**: Unified interface for different language models.

#### `LLMInterface`
```python
class LLMInterface:
    """Base interface for language models."""
    
    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.model = self._load_model()
        self.tokenizer = self._load_tokenizer()
    
    def generate(
        self,
        input_tokens: Union[torch.Tensor, List[int]],
        max_length: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> GenerationResult:
        """Generate text from input tokens."""
        
    def embed_tokens(
        self,
        tokens: Union[torch.Tensor, List[int]]
    ) -> torch.Tensor:
        """Get token embeddings."""
        
    def add_visual_vocabulary(
        self,
        visual_tokens: Dict[str, torch.Tensor]
    ) -> None:
        """Extend model vocabulary with visual tokens."""
```

**Supported Models**:
```python
SUPPORTED_LLMS = {
    "gpt": ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
    "gpt-neo": ["EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-2.7B"],
    "gpt-j": ["EleutherAI/gpt-j-6B"],
    "llama": ["meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf"],
    "bert": ["bert-base-uncased", "bert-large-uncased"],
    "roberta": ["roberta-base", "roberta-large"],
    "t5": ["t5-small", "t5-base", "t5-large"],
    "dialog": ["microsoft/DialoGPT-medium", "microsoft/DialoGPT-large"]
}
```

### 2. Specialized Models (`helicopter.models.specialized/`)

**Purpose**: Domain-specific model implementations.

#### Medical Models (`medical_models.py`)
```python
class MedicalImageAnalyzer:
    """Specialized analyzer for medical images."""
    
    def __init__(self):
        self.models = {
            "chest_xray": load_model("chest_xray_classifier"),
            "mri_brain": load_model("brain_mri_segmenter"),
            "ct_scan": load_model("ct_scan_analyzer"),
            "pathology": load_model("pathology_detector")
        }
    
    def analyze_medical_image(
        self,
        image: torch.Tensor,
        modality: str,
        anatomical_region: str
    ) -> MedicalAnalysisResult:
        """Analyze medical image with appropriate specialized model."""
```

#### Biomechanics Models (`biomechanics_models.py`)
```python
class BiomechanicsAnalyzer:
    """Specialized analyzer for biomechanical data."""
    
    def analyze_movement_pattern(
        self,
        sequence: torch.Tensor,
        movement_type: str
    ) -> BiomechanicsResult:
        """Analyze movement patterns from visual sequences."""
```

## Training Modules

### 1. Domain Trainer (`helicopter.training.domain_trainer`)

**Purpose**: Specialized training for domain-specific models.

#### `DomainTrainer`
```python
class DomainTrainer:
    """Training pipeline for domain-specific visual LLMs."""
    
    def __init__(
        self,
        base_model: str,
        domain: str,
        device: str = "auto"
    ):
        self.base_model = self._load_base_model(base_model)
        self.domain_config = load_domain_config(domain)
        self.training_config = TrainingConfig()
    
    def prepare_visual_dataset(
        self,
        image_dir: str,
        annotations: Optional[str] = None,
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    ) -> VisualDataset:
        """Prepare dataset with visual tokens and corresponding text."""
        
    def train(
        self,
        dataset: VisualDataset,
        epochs: int = 10,
        batch_size: int = 8,
        learning_rate: float = 1e-4,
        **kwargs
    ) -> TrainedModel:
        """Execute training pipeline."""
```

### 2. Curriculum Learning (`helicopter.training.curriculum`)

**Purpose**: Implements progressive learning strategies.

#### `CurriculumLearner`
```python
class CurriculumLearner:
    """Implements curriculum learning for visual understanding."""
    
    def __init__(self, difficulty_scorer: Callable):
        self.difficulty_scorer = difficulty_scorer
        self.curriculum_schedule = None
    
    def create_curriculum(
        self,
        dataset: VisualDataset,
        strategy: str = "difficulty_based"
    ) -> CurriculumSchedule:
        """Create learning curriculum from dataset."""
        
    def get_next_batch(
        self,
        epoch: int,
        global_step: int
    ) -> Batch:
        """Get next batch according to curriculum."""
```

## Utility Modules

### 1. Image Processing (`helicopter.utils.image_processing`)

**Purpose**: Image manipulation and preprocessing utilities.

```python
def load_image(
    path: Union[str, Path],
    target_size: Optional[Tuple[int, int]] = None,
    normalize: bool = True
) -> torch.Tensor:
    """Load and preprocess image."""

def extract_patches(
    image: torch.Tensor,
    patch_size: int = 16,
    stride: Optional[int] = None,
    padding: bool = True
) -> torch.Tensor:
    """Extract patches from image."""

def apply_regional_mask(
    image: torch.Tensor,
    regions: List[Dict],
    background_value: float = 0.0
) -> torch.Tensor:
    """Apply regional masks to image."""
```

### 2. Visualization (`helicopter.utils.visualization`)

**Purpose**: Visualization tools for analysis results.

```python
def visualize_visual_tokens(
    tokens: VisualTokens,
    original_image: torch.Tensor,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Visualize visual tokens overlaid on original image."""

def plot_attention_maps(
    attention_maps: List[torch.Tensor],
    original_image: torch.Tensor,
    timesteps: List[int]
) -> plt.Figure:
    """Plot attention maps across diffusion timesteps."""

def create_analysis_dashboard(
    analysis_results: List[AnalysisResult]
) -> plotly.graph_objects.Figure:
    """Create interactive dashboard for analysis results."""
```

## API Modules

### 1. FastAPI Server (`helicopter.api.server`)

**Purpose**: REST API server for web integration.

```python
from fastapi import FastAPI, UploadFile, BackgroundTasks
from helicopter import HelicopterPipeline

app = FastAPI(title="Helicopter API", version="0.1.0")
helicopter = HelicopterPipeline()

@app.post("/tokenize")
async def tokenize_image(
    file: UploadFile,
    regions: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """Tokenize uploaded image."""

@app.post("/analyze")
async def analyze_image(
    file: UploadFile,
    goal: str,
    domain: Optional[str] = None
) -> Dict[str, Any]:
    """Perform full image analysis."""

@app.websocket("/ws/process")
async def websocket_process(websocket: WebSocket):
    """Real-time processing via WebSocket."""
```

### 2. WebSocket Processing (`helicopter.api.websocket`)

**Purpose**: Real-time processing capabilities.

```python
class ProcessingWebSocket:
    """WebSocket handler for real-time processing."""
    
    async def handle_connection(self, websocket: WebSocket):
        """Handle WebSocket connection for real-time processing."""
        
    async def process_stream(
        self,
        image_stream: AsyncIterator[bytes]
    ) -> AsyncIterator[Dict]:
        """Process streaming images in real-time."""
```

## Experimental Modules

### 1. NeRF Integration (`helicopter.experimental.nerf_integration`)

**Purpose**: Neural Radiance Fields for 3D understanding.

```python
class NeRFTokenizer:
    """Extract 3D spatial tokens using NeRF."""
    
    def extract_3d_tokens(
        self,
        image_sequence: List[torch.Tensor],
        camera_poses: List[np.ndarray]
    ) -> SpatialTokens:
        """Extract 3D spatial understanding from 2D sequence."""
```

### 2. Causal Reasoning (`helicopter.experimental.causal_reasoning`)

**Purpose**: Causal inference in visual sequences.

```python
class CausalAnalyzer:
    """Identify causal relationships in visual data."""
    
    def discover_causal_structure(
        self,
        sequence: List[torch.Tensor],
        interventions: Optional[List[Dict]] = None
    ) -> CausalGraph:
        """Discover causal relationships in visual sequence."""
```

## Configuration and Extension

### Model Registry
```python
class ModelRegistry:
    """Central registry for all available models."""
    
    def register_model(
        self,
        name: str,
        model_class: Type,
        config: Dict[str, Any]
    ) -> None:
        """Register new model type."""
        
    def get_model(self, name: str) -> Any:
        """Retrieve registered model."""
        
    def list_models(self, category: Optional[str] = None) -> List[str]:
        """List available models."""
```

### Custom Model Integration
```python
class CustomModelAdapter:
    """Adapter for integrating custom models."""
    
    def adapt_model(
        self,
        model: Any,
        input_processor: Callable,
        output_processor: Callable
    ) -> ModelInterface:
        """Adapt custom model to Helicopter interface."""
```

This modular architecture ensures extensibility, maintainability, and clear separation of concerns while providing comprehensive functionality for visual knowledge extraction and domain-specific LLM creation.
