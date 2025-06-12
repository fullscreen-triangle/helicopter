---
layout: page
title: API Reference
permalink: /api-reference/
---

# API Reference

Complete API documentation for Helicopter's autonomous reconstruction framework. This reference covers all classes, methods, and functions for implementing the genius insight: "reconstruction ability demonstrates understanding."

## 🧠 Core Classes

### AutonomousReconstructionEngine

The heart of Helicopter's revolutionary approach - tests understanding through reconstruction ability.

```python
class AutonomousReconstructionEngine:
    """
    Autonomous reconstruction engine that demonstrates understanding through
    the ability to reconstruct images from partial information.
    
    The genius insight: If a system can perfectly reconstruct an image,
    it has truly understood it.
    """
```

#### Constructor

```python
def __init__(self, patch_size=32, context_size=96, device="auto"):
    """
    Initialize the autonomous reconstruction engine.
    
    Args:
        patch_size (int): Size of patches to reconstruct (16, 32, 64)
        context_size (int): Context window size for prediction (64, 96, 128)
        device (str): Device to use ("cuda", "cpu", "auto")
    
    Example:
        >>> engine = AutonomousReconstructionEngine(
        ...     patch_size=32,
        ...     context_size=96,
        ...     device="cuda"
        ... )
    """
```

#### Primary Methods

##### autonomous_analyze()

```python
def autonomous_analyze(self, image, max_iterations=50, target_quality=0.90):
    """
    Perform autonomous reconstruction analysis - the ultimate test.
    
    This method embodies the core insight: can the system reconstruct
    what it sees? If yes, it has truly understood the image.
    
    Args:
        image (np.ndarray): Input image (H, W, 3) in BGR format
        max_iterations (int): Maximum reconstruction attempts
        target_quality (float): Stop when this quality achieved (0.0-1.0)
    
    Returns:
        dict: Comprehensive analysis results containing:
            - autonomous_reconstruction: Reconstruction metrics
            - understanding_insights: What reconstruction demonstrates
            - reconstruction_history: Learning progression
            - reconstruction_image: Final reconstructed image
    
    Example:
        >>> results = engine.autonomous_analyze(
        ...     image=image,
        ...     max_iterations=50,
        ...     target_quality=0.90
        ... )
        >>> quality = results['autonomous_reconstruction']['final_quality']
        >>> understanding = results['understanding_insights']['understanding_level']
    """
```

#### Return Structure

```python
{
    'autonomous_reconstruction': {
        'final_quality': float,              # 0.0-1.0, reconstruction fidelity
        'completion_percentage': float,       # Percentage of image reconstructed
        'patches_reconstructed': int,         # Number of patches completed
        'total_patches': int,                # Total patches in image
        'reconstruction_iterations': int,     # Iterations to convergence
        'average_confidence': float,         # Average prediction confidence
        'convergence_achieved': bool         # Whether target quality reached
    },
    'understanding_insights': {
        'understanding_level': str,          # 'excellent', 'good', 'moderate', 'limited'
        'reconstruction_demonstrates': list, # What reconstruction proves
        'key_insights': list,               # Understanding insights
        'confidence_assessment': str        # Confidence in understanding
    },
    'reconstruction_history': [
        {
            'iteration': int,               # Iteration number
            'quality': float,              # Quality at this iteration
            'confidence': float,           # Prediction confidence
            'patch_location': tuple,       # (x, y) of reconstructed patch
            'strategy_used': str          # Reconstruction strategy
        }
    ],
    'reconstruction_image': np.ndarray      # Final reconstructed image
}
```

### ComprehensiveAnalysisEngine

Integrates autonomous reconstruction with traditional computer vision methods for complete analysis.

```python
class ComprehensiveAnalysisEngine:
    """
    Comprehensive analysis engine that uses autonomous reconstruction
    as the primary method, validated by traditional CV approaches.
    """
```

#### Constructor

```python
def __init__(self, reconstruction_config=None, learning_config=None):
    """
    Initialize comprehensive analysis engine.
    
    Args:
        reconstruction_config (dict): Configuration for reconstruction engine
        learning_config (dict): Configuration for learning system
    
    Example:
        >>> engine = ComprehensiveAnalysisEngine(
        ...     reconstruction_config={
        ...         'patch_size': 32,
        ...         'context_size': 96,
        ...         'max_iterations': 50
        ...     },
        ...     learning_config={
        ...         'target_confidence': 0.85,
        ...         'max_iterations': 10
        ...     }
        ... )
    """
```

#### Primary Methods

##### comprehensive_analysis()

```python
def comprehensive_analysis(self, image, metadata=None, 
                         enable_autonomous_reconstruction=True,
                         enable_iterative_learning=True):
    """
    Perform comprehensive analysis with autonomous reconstruction as primary method.
    
    Args:
        image (np.ndarray): Input image
        metadata (dict): Image metadata and context
        enable_autonomous_reconstruction (bool): Use reconstruction as primary
        enable_iterative_learning (bool): Enable learning and improvement
    
    Returns:
        dict: Complete analysis results with cross-validation
    
    Example:
        >>> results = engine.comprehensive_analysis(
        ...     image=image,
        ...     metadata={'domain': 'medical', 'type': 'xray'},
        ...     enable_autonomous_reconstruction=True,
        ...     enable_iterative_learning=True
        ... )
    """
```

## 🔄 Learning and Optimization Classes

### ContinuousLearningEngine

Implements continuous learning through Bayesian inference and fuzzy logic.

```python
class ContinuousLearningEngine:
    """
    Continuous learning system that improves understanding through
    iterative analysis and metacognitive orchestration.
    """
```

#### Key Methods

##### learn_from_analysis()

```python
def learn_from_analysis(self, image, analysis_results, ground_truth=None):
    """
    Learn from analysis results to improve future performance.
    
    Args:
        image (np.ndarray): Analyzed image
        analysis_results (dict): Results from analysis
        ground_truth (dict, optional): Ground truth for validation
    
    Returns:
        dict: Learning results and updated knowledge
    """
```

##### iterate_until_convergence()

```python
def iterate_until_convergence(self, images, initial_analysis_results, 
                            ground_truth=None):
    """
    Iteratively improve analysis until convergence achieved.
    
    Args:
        images (list): List of images to analyze
        initial_analysis_results (list): Initial analysis results
        ground_truth (list, optional): Ground truth data
    
    Returns:
        dict: Convergence results and final analysis
    """
```

### BayesianObjectiveEngine

Implements Bayesian belief networks for probabilistic visual reasoning.

```python
class BayesianObjectiveEngine:
    """
    Bayesian objective engine for probabilistic reasoning about visual data.
    Handles the non-binary, continuous nature of pixel information.
    """
```

#### Key Methods

##### update_beliefs()

```python
def update_beliefs(self, visual_evidence, prior_beliefs=None):
    """
    Update Bayesian beliefs based on visual evidence.
    
    Args:
        visual_evidence (dict): Evidence from visual analysis
        prior_beliefs (dict, optional): Prior belief state
    
    Returns:
        dict: Updated belief state
    """
```

### MetacognitiveOrchestrator

Orchestrates the learning process and optimizes analysis strategies.

```python
class MetacognitiveOrchestrator:
    """
    Metacognitive orchestrator that learns about the learning process
    and optimizes analysis strategies.
    """
```

#### Key Methods

##### optimize_learning_process()

```python
def optimize_learning_process(self, analysis_history):
    """
    Optimize learning process based on historical performance.
    
    Args:
        analysis_history (list): History of analysis attempts
    
    Returns:
        dict: Optimization strategy and recommendations
    """
```

## 🎯 Utility Classes and Functions

### ReconstructionState

Tracks the state of autonomous reconstruction process.

```python
class ReconstructionState:
    """
    Tracks the current state of autonomous reconstruction.
    
    Attributes:
        known_patches (list): List of known image patches
        unknown_patches (list): List of patches to reconstruct
        current_reconstruction (np.ndarray): Current reconstruction
        iteration (int): Current iteration number
        quality_history (list): History of quality scores
    """
```

### QualityAssessor

Assesses reconstruction quality against original image.

```python
class QualityAssessor:
    """
    Assesses the quality of reconstruction against original image.
    """
    
    def assess_quality(self, reconstruction, original, mask=None):
        """
        Assess reconstruction quality.
        
        Args:
            reconstruction (np.ndarray): Reconstructed image
            original (np.ndarray): Original image
            mask (np.ndarray, optional): Mask for assessment region
        
        Returns:
            float: Quality score (0.0-1.0)
        """
```

### ConfidenceEstimator

Estimates confidence in reconstruction predictions.

```python
class ConfidenceEstimator:
    """
    Estimates confidence in reconstruction predictions.
    """
    
    def estimate_confidence(self, context, prediction):
        """
        Estimate confidence in a reconstruction prediction.
        
        Args:
            context (np.ndarray): Context used for prediction
            prediction (np.ndarray): Predicted patch
        
        Returns:
            float: Confidence score (0.0-1.0)
        """
```

## 📊 Analysis Result Classes

### UnderstandingInsights

Provides insights into what reconstruction demonstrates about understanding.

```python
class UnderstandingInsights:
    """
    Analyzes reconstruction results to provide insights into understanding.
    """
    
    def analyze_understanding(self, reconstruction_results):
        """
        Analyze what reconstruction results demonstrate about understanding.
        
        Args:
            reconstruction_results (dict): Results from reconstruction
        
        Returns:
            dict: Understanding insights and assessment
        """
```

### CrossValidationEngine

Cross-validates reconstruction insights with traditional methods.

```python
class CrossValidationEngine:
    """
    Cross-validates reconstruction insights with supporting methods.
    """
    
    def validate_reconstruction_insights(self, reconstruction_results, 
                                       supporting_results):
        """
        Validate reconstruction insights against supporting methods.
        
        Args:
            reconstruction_results (dict): Reconstruction analysis results
            supporting_results (dict): Results from supporting methods
        
        Returns:
            dict: Cross-validation results
        """
```

## 🔧 Configuration Classes

### ReconstructionConfig

Configuration for autonomous reconstruction engine.

```python
@dataclass
class ReconstructionConfig:
    """Configuration for autonomous reconstruction engine."""
    
    patch_size: int = 32
    context_size: int = 96
    max_iterations: int = 50
    target_quality: float = 0.90
    device: str = "auto"
    
    # Strategy selection
    strategy_weights: Dict[str, float] = field(default_factory=lambda: {
        'edge_guided': 0.3,
        'content_aware': 0.3,
        'uncertainty_guided': 0.2,
        'progressive_refinement': 0.2
    })
```

### LearningConfig

Configuration for continuous learning system.

```python
@dataclass
class LearningConfig:
    """Configuration for continuous learning system."""
    
    target_confidence: float = 0.85
    max_iterations: int = 10
    convergence_threshold: float = 0.01
    learning_rate: float = 0.001
    
    # Bayesian parameters
    prior_strength: float = 0.1
    evidence_weight: float = 0.9
    
    # Fuzzy logic parameters
    fuzzy_membership_functions: List[str] = field(default_factory=lambda: [
        'gaussian', 'triangular', 'trapezoidal'
    ])
```

## 🎨 Visualization Functions

### visualize_reconstruction_process()

```python
def visualize_reconstruction_process(results, original_image, save_path=None):
    """
    Visualize the autonomous reconstruction process.
    
    Args:
        results (dict): Results from autonomous_analyze()
        original_image (np.ndarray): Original image
        save_path (str, optional): Path to save visualization
    
    Returns:
        matplotlib.figure.Figure: Visualization figure
    
    Example:
        >>> fig = visualize_reconstruction_process(results, original_image)
        >>> plt.show()
    """
```

### plot_learning_progression()

```python
def plot_learning_progression(reconstruction_history, save_path=None):
    """
    Plot the learning progression during reconstruction.
    
    Args:
        reconstruction_history (list): History from reconstruction results
        save_path (str, optional): Path to save plot
    
    Returns:
        matplotlib.figure.Figure: Learning progression plot
    """
```

### create_understanding_dashboard()

```python
def create_understanding_dashboard(comprehensive_results, save_path=None):
    """
    Create comprehensive dashboard showing understanding analysis.
    
    Args:
        comprehensive_results (dict): Results from comprehensive_analysis()
        save_path (str, optional): Path to save dashboard
    
    Returns:
        matplotlib.figure.Figure: Understanding dashboard
    """
```

## 🔍 Debugging and Monitoring

### ReconstructionMonitor

Monitor reconstruction process in real-time.

```python
class ReconstructionMonitor:
    """
    Monitor autonomous reconstruction process in real-time.
    """
    
    def __init__(self, enable_logging=True, log_level="INFO"):
        """
        Initialize reconstruction monitor.
        
        Args:
            enable_logging (bool): Enable detailed logging
            log_level (str): Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
        """
    
    def monitor_iteration(self, iteration_data):
        """
        Monitor a single reconstruction iteration.
        
        Args:
            iteration_data (dict): Data from current iteration
        """
    
    def get_performance_metrics(self):
        """
        Get performance metrics from monitoring.
        
        Returns:
            dict: Performance metrics and statistics
        """
```

### AnalysisProfiler

Profile analysis performance and resource usage.

```python
class AnalysisProfiler:
    """
    Profile analysis performance and resource usage.
    """
    
    def profile_analysis(self, analysis_function, *args, **kwargs):
        """
        Profile an analysis function.
        
        Args:
            analysis_function: Function to profile
            *args: Arguments for function
            **kwargs: Keyword arguments for function
        
        Returns:
            tuple: (results, profiling_data)
        """
```

## 🚀 Batch Processing

### BatchAnalysisEngine

Process multiple images with autonomous reconstruction.

```python
class BatchAnalysisEngine:
    """
    Batch processing engine for autonomous reconstruction analysis.
    """
    
    def __init__(self, reconstruction_config=None, parallel_workers=4):
        """
        Initialize batch analysis engine.
        
        Args:
            reconstruction_config (ReconstructionConfig): Configuration
            parallel_workers (int): Number of parallel workers
        """
    
    def analyze_batch(self, image_paths, output_dir, progress_callback=None):
        """
        Analyze a batch of images.
        
        Args:
            image_paths (list): List of image file paths
            output_dir (str): Output directory for results
            progress_callback (callable): Progress callback function
        
        Returns:
            dict: Batch analysis results and summary
        """
```

## 📈 Performance Optimization

### optimize_for_speed()

```python
def optimize_for_speed(engine_config):
    """
    Optimize configuration for speed.
    
    Args:
        engine_config (ReconstructionConfig): Current configuration
    
    Returns:
        ReconstructionConfig: Speed-optimized configuration
    """
```

### optimize_for_quality()

```python
def optimize_for_quality(engine_config):
    """
    Optimize configuration for quality.
    
    Args:
        engine_config (ReconstructionConfig): Current configuration
    
    Returns:
        ReconstructionConfig: Quality-optimized configuration
    """
```

### optimize_for_memory()

```python
def optimize_for_memory(engine_config):
    """
    Optimize configuration for memory usage.
    
    Args:
        engine_config (ReconstructionConfig): Current configuration
    
    Returns:
        ReconstructionConfig: Memory-optimized configuration
    """
```

## 🔗 Integration Functions

### integrate_with_vibrio()

```python
def integrate_with_vibrio(helicopter_results, vibrio_results):
    """
    Integrate Helicopter results with Vibrio motion analysis.
    
    Args:
        helicopter_results (dict): Helicopter analysis results
        vibrio_results (dict): Vibrio motion analysis results
    
    Returns:
        dict: Integrated analysis results
    """
```

### integrate_with_moriarty()

```python
def integrate_with_moriarty(helicopter_results, moriarty_results):
    """
    Integrate Helicopter results with Moriarty pose detection.
    
    Args:
        helicopter_results (dict): Helicopter analysis results
        moriarty_results (dict): Moriarty pose detection results
    
    Returns:
        dict: Integrated analysis results
    """
```

## 📝 Example Usage Patterns

### Basic Reconstruction Analysis

```python
# Initialize engine
engine = AutonomousReconstructionEngine(
    patch_size=32,
    context_size=96,
    device="cuda"
)

# Load image
image = cv2.imread("image.jpg")

# Perform analysis
results = engine.autonomous_analyze(
    image=image,
    max_iterations=50,
    target_quality=0.90
)

# Check understanding
quality = results['autonomous_reconstruction']['final_quality']
understanding = results['understanding_insights']['understanding_level']

print(f"Reconstruction Quality: {quality:.1%}")
print(f"Understanding Level: {understanding}")
```

### Comprehensive Analysis with Learning

```python
# Initialize comprehensive engine
analysis_engine = ComprehensiveAnalysisEngine(
    reconstruction_config={
        'patch_size': 32,
        'max_iterations': 50,
        'target_quality': 0.90
    },
    learning_config={
        'target_confidence': 0.85,
        'max_iterations': 10
    }
)

# Perform comprehensive analysis
results = analysis_engine.comprehensive_analysis(
    image=image,
    metadata={'domain': 'medical', 'type': 'xray'},
    enable_autonomous_reconstruction=True,
    enable_iterative_learning=True
)

# Get final assessment
assessment = results['final_assessment']
print(f"Understanding Demonstrated: {assessment['understanding_demonstrated']}")
print(f"Confidence Score: {assessment['confidence_score']:.1%}")
```

### Batch Processing

```python
# Initialize batch engine
batch_engine = BatchAnalysisEngine(
    reconstruction_config=ReconstructionConfig(
        patch_size=32,
        max_iterations=30,
        target_quality=0.85
    ),
    parallel_workers=4
)

# Process batch
results = batch_engine.analyze_batch(
    image_paths=["img1.jpg", "img2.jpg", "img3.jpg"],
    output_dir="results/",
    progress_callback=lambda i, total, path: print(f"Processing {i+1}/{total}: {path}")
)

print(f"Batch completed: {results['summary']['success_rate']:.1%} success rate")
```

---

<div style="background: #f8f9fa; padding: 1.5rem; border-radius: 8px; margin: 2rem 0;">
<h3>🎯 API Design Philosophy</h3>
<p>The Helicopter API is designed around the core insight that <strong>reconstruction ability demonstrates understanding</strong>. Every method and class supports this principle by either:</p>
<ul>
<li><strong>Testing reconstruction ability</strong> - Primary analysis methods</li>
<li><strong>Supporting reconstruction insights</strong> - Validation and cross-checking</li>
<li><strong>Learning from reconstruction</strong> - Continuous improvement</li>
<li><strong>Optimizing reconstruction</strong> - Performance and quality tuning</li>
</ul>
<p>This unified approach ensures that all functionality serves the ultimate goal of measuring visual understanding through reconstruction fidelity.</p>
</div>

## 🔗 Related Documentation

- **[Getting Started](getting-started.html)** - Basic usage and installation
- **[Autonomous Reconstruction](autonomous-reconstruction.html)** - Core reconstruction engine
- **[Comprehensive Analysis](comprehensive-analysis.html)** - Full analysis pipeline
- **[Examples](examples.html)** - Practical applications and use cases
