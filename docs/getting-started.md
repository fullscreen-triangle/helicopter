---
layout: page
title: Getting Started
permalink: /getting-started/
---

# Getting Started with Helicopter

Welcome to Helicopter - the revolutionary computer vision framework that proves understanding through reconstruction ability. This guide will get you up and running with autonomous visual analysis in minutes.

## 🧠 The Core Insight

Before diving into installation, understand the genius behind Helicopter:

> **The best way to know if an AI has truly analyzed an image is if it can perfectly reconstruct it.**

Traditional computer vision asks: *"What do you see?"*  
Helicopter asks: *"Can you draw what you see?"*

If the answer is "yes" with high fidelity, then true understanding has been demonstrated.

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended, but CPU works too)
- 4GB+ RAM (8GB+ recommended)

### Quick Installation

```bash
# Create a virtual environment
python -m venv helicopter-env
source helicopter-env/bin/activate  # On Windows: helicopter-env\Scripts\activate

# Install Helicopter
pip install helicopter-cv

# Verify installation
python -c "from helicopter.core import AutonomousReconstructionEngine; print('✅ Helicopter installed successfully!')"
```

### Development Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/helicopter.git
cd helicopter

# Create virtual environment
python -m venv env
source env/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Run tests to verify
pytest tests/
```

## 🚀 Your First Reconstruction Analysis

Let's start with the simplest possible example - analyzing an image through autonomous reconstruction:

### Step 1: Basic Reconstruction

```python
import cv2
from helicopter.core import AutonomousReconstructionEngine

# Load an image (any image will work!)
image = cv2.imread("path/to/your/image.jpg")

# Initialize the autonomous reconstruction engine
engine = AutonomousReconstructionEngine(
    patch_size=32,      # Size of patches to reconstruct
    context_size=96,    # Context window for prediction
    device="auto"       # Automatically choose GPU/CPU
)

# Perform the ultimate test: Can it reconstruct what it sees?
results = engine.autonomous_analyze(
    image=image,
    max_iterations=30,    # Maximum reconstruction attempts
    target_quality=0.85   # Stop when 85% quality achieved
)

# Check the results
understanding_level = results['understanding_insights']['understanding_level']
reconstruction_quality = results['autonomous_reconstruction']['final_quality']

print(f"🧠 Understanding Level: {understanding_level}")
print(f"📊 Reconstruction Quality: {reconstruction_quality:.1%}")

if reconstruction_quality > 0.95:
    print("🎉 Perfect reconstruction achieved - complete understanding!")
elif reconstruction_quality > 0.8:
    print("✅ High-quality reconstruction - strong understanding demonstrated")
else:
    print("⚠️ Limited reconstruction quality - understanding incomplete")
```

### Step 2: Understanding the Results

The results contain rich information about what the system learned:

```python
# Detailed reconstruction metrics
recon_metrics = results['autonomous_reconstruction']
print(f"Patches reconstructed: {recon_metrics['patches_reconstructed']}/{recon_metrics['total_patches']}")
print(f"Completion: {recon_metrics['completion_percentage']:.1f}%")
print(f"Average confidence: {recon_metrics['average_confidence']:.3f}")
print(f"Iterations needed: {recon_metrics['reconstruction_iterations']}")

# Understanding insights
insights = results['understanding_insights']
print(f"\nWhat reconstruction demonstrates:")
for demo in insights['reconstruction_demonstrates']:
    print(f"  • {demo}")

print(f"\nKey insights:")
for insight in insights['key_insights']:
    print(f"  • {insight}")
```

### Step 3: Monitoring Learning Progress

```python
# View the learning progression
history = results['reconstruction_history']
print(f"\nLearning progression (last 5 iterations):")
for h in history[-5:]:
    print(f"  Iteration {h['iteration']}: Quality={h['quality']:.3f}, "
          f"Confidence={h['confidence']:.3f}")

# Plot the learning curve
import matplotlib.pyplot as plt

qualities = [h['quality'] for h in history]
confidences = [h['confidence'] for h in history]

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(qualities, label='Reconstruction Quality', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Quality')
plt.title('Learning Progress')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(confidences, label='Prediction Confidence', color='orange', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Confidence')
plt.title('Confidence Evolution')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## 🔬 Comprehensive Analysis

For more thorough analysis that includes cross-validation with traditional methods:

```python
from helicopter.core import ComprehensiveAnalysisEngine

# Initialize comprehensive analysis
analysis_engine = ComprehensiveAnalysisEngine()

# Perform full analysis with autonomous reconstruction as primary method
results = analysis_engine.comprehensive_analysis(
    image=image,
    metadata={'source': 'user_upload', 'timestamp': '2024-01-01'},
    enable_autonomous_reconstruction=True,  # The ultimate test
    enable_iterative_learning=True          # Learn and improve
)

# Get final assessment
assessment = results['final_assessment']
print(f"\n🎯 FINAL ASSESSMENT")
print(f"Primary method: {assessment['primary_method']}")
print(f"Understanding demonstrated: {assessment['understanding_demonstrated']}")
print(f"Confidence score: {assessment['confidence_score']:.1%}")

print(f"\n📋 KEY FINDINGS:")
for finding in assessment['key_findings']:
    print(f"  • {finding}")

print(f"\n💡 RECOMMENDATIONS:")
for rec in assessment['recommendations']:
    print(f"  • {rec}")
```

## 🎛️ Configuration Options

### Reconstruction Engine Parameters

```python
engine = AutonomousReconstructionEngine(
    patch_size=32,          # Smaller = more detail, slower
    context_size=96,        # Larger = more context, more memory
    device="cuda",          # "cuda", "cpu", or "auto"
)

# Analysis parameters
results = engine.autonomous_analyze(
    image=image,
    max_iterations=50,      # More iterations = better quality
    target_quality=0.90,    # Higher target = more thorough analysis
)
```

### Comprehensive Analysis Options

```python
results = analysis_engine.comprehensive_analysis(
    image=image,
    metadata=metadata,
    enable_autonomous_reconstruction=True,   # Primary method
    enable_iterative_learning=True,         # Learn and improve
)
```

## 🔍 Understanding the Output

### Reconstruction Quality Levels

| Quality Range | Understanding Level | Meaning |
|---------------|-------------------|---------|
| 95-100% | Excellent | Perfect pixel-level understanding |
| 80-94% | Good | Strong structural understanding |
| 60-79% | Moderate | Basic pattern recognition |
| 0-59% | Limited | Minimal understanding demonstrated |

### Key Metrics Explained

- **Reconstruction Quality**: How well the system can redraw the image (0-1)
- **Understanding Level**: Categorical assessment of comprehension
- **Prediction Confidence**: How confident the system is in its predictions
- **Completion Percentage**: How much of the image was successfully reconstructed
- **Learning Progress**: How much the system improved during analysis

## 🚨 Common Issues and Solutions

### Issue: Low Reconstruction Quality

```python
# Solution 1: Increase iterations
results = engine.autonomous_analyze(
    image=image,
    max_iterations=100,  # More attempts
    target_quality=0.80  # Lower target initially
)

# Solution 2: Adjust patch size
engine = AutonomousReconstructionEngine(
    patch_size=16,  # Smaller patches for more detail
    context_size=64
)
```

### Issue: Out of Memory

```python
# Solution: Reduce context size or use CPU
engine = AutonomousReconstructionEngine(
    patch_size=32,
    context_size=64,    # Smaller context
    device="cpu"        # Use CPU instead of GPU
)
```

### Issue: Slow Performance

```python
# Solution: Optimize parameters
engine = AutonomousReconstructionEngine(
    patch_size=64,      # Larger patches = fewer iterations
    context_size=96,
    device="cuda"       # Use GPU if available
)

results = engine.autonomous_analyze(
    image=image,
    max_iterations=20,  # Fewer iterations for speed
    target_quality=0.75 # Lower quality target
)
```

## 🎯 Next Steps

Now that you've got Helicopter running, explore these advanced topics:

1. **[Autonomous Reconstruction](autonomous-reconstruction.html)** - Deep dive into the core engine
2. **[Comprehensive Analysis](comprehensive-analysis.html)** - Full analysis pipeline
3. **[Examples](examples.html)** - Real-world use cases and applications
4. **[API Reference](api-reference.html)** - Complete API documentation

## 💡 Quick Tips

- **Start simple**: Use default parameters for your first analyses
- **Monitor quality**: Watch reconstruction quality as your primary metric
- **Iterate gradually**: Increase complexity as you understand the system
- **Visualize progress**: Plot learning curves to understand system behavior
- **Cross-validate**: Use comprehensive analysis for important applications

---

<div style="background: #f8f9fa; padding: 1.5rem; border-radius: 8px; margin: 2rem 0;">
<h3>🎉 Congratulations!</h3>
<p>You've just performed your first autonomous reconstruction analysis. The system has demonstrated its understanding by showing whether it can "draw what it sees."</p>
<p><strong>Remember</strong>: Perfect reconstruction = Perfect understanding. This is the genius insight that makes Helicopter revolutionary.</p>
</div>

## 🆘 Need Help?

- **Documentation**: Continue reading the detailed guides
- **Examples**: Check out practical examples in the [Examples](examples.html) section
- **Issues**: Report bugs on [GitHub Issues](https://github.com/yourusername/helicopter/issues)
- **Discussions**: Join conversations on [GitHub Discussions](https://github.com/yourusername/helicopter/discussions) 