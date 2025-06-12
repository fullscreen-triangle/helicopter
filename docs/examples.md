---
layout: page
title: Examples
permalink: /examples/
---

# Practical Examples

This page provides comprehensive examples demonstrating the power of autonomous reconstruction across various domains. Each example shows how the genius insight - "reconstruction ability demonstrates understanding" - applies to real-world scenarios.

## 🏥 Medical Imaging Analysis

### Example 1: Chest X-Ray Understanding Validation

```python
import cv2
import numpy as np
from helicopter.core import AutonomousReconstructionEngine, ComprehensiveAnalysisEngine

def validate_chest_xray_understanding(xray_path):
    """
    Validate AI understanding of chest X-ray through reconstruction ability
    
    The ultimate test: Can the system reconstruct what it sees?
    If yes, it truly understands the anatomical structures.
    """
    
    # Load chest X-ray
    xray_image = cv2.imread(xray_path, cv2.IMREAD_GRAYSCALE)
    xray_image = cv2.cvtColor(xray_image, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel
    
    print("🏥 CHEST X-RAY UNDERSTANDING VALIDATION")
    print("Testing: Can the AI reconstruct what it sees?")
    
    # Initialize reconstruction engine for medical imaging
    engine = AutonomousReconstructionEngine(
        patch_size=16,      # Smaller patches for medical detail
        context_size=64,    # Sufficient context for anatomy
        device="cuda"       # Use GPU for medical precision
    )
    
    # Perform the ultimate test
    results = engine.autonomous_analyze(
        image=xray_image,
        max_iterations=75,   # More iterations for medical precision
        target_quality=0.95  # High quality required for medical validation
    )
    
    # Extract results
    understanding_level = results['understanding_insights']['understanding_level']
    reconstruction_quality = results['autonomous_reconstruction']['final_quality']
    completion_percentage = results['autonomous_reconstruction']['completion_percentage']
    
    print(f"\n📊 RECONSTRUCTION RESULTS:")
    print(f"Understanding Level: {understanding_level}")
    print(f"Reconstruction Quality: {reconstruction_quality:.1%}")
    print(f"Completion: {completion_percentage:.1f}%")
    
    # Medical validation assessment
    if reconstruction_quality > 0.95:
        medical_assessment = "EXCELLENT - System demonstrates complete understanding of anatomical structures"
        clinical_confidence = "HIGH - Suitable for diagnostic assistance"
    elif reconstruction_quality > 0.85:
        medical_assessment = "GOOD - System shows strong understanding with minor gaps"
        clinical_confidence = "MODERATE - Requires expert review"
    elif reconstruction_quality > 0.70:
        medical_assessment = "MODERATE - Basic anatomical understanding demonstrated"
        clinical_confidence = "LOW - Additional validation needed"
    else:
        medical_assessment = "LIMITED - Insufficient understanding for medical applications"
        clinical_confidence = "VERY LOW - Not suitable for clinical use"
    
    print(f"\n🏥 MEDICAL ASSESSMENT:")
    print(f"Understanding: {medical_assessment}")
    print(f"Clinical Confidence: {clinical_confidence}")
    
    # Analyze reconstruction patterns for medical insights
    history = results['reconstruction_history']
    if history:
        print(f"\n📈 LEARNING PROGRESSION:")
        print(f"Initial Quality: {history[0]['quality']:.3f}")
        print(f"Final Quality: {history[-1]['quality']:.3f}")
        print(f"Quality Improvement: {history[-1]['quality'] - history[0]['quality']:.3f}")
        print(f"Iterations to Convergence: {len(history)}")
    
    return {
        'reconstruction_quality': reconstruction_quality,
        'understanding_level': understanding_level,
        'medical_assessment': medical_assessment,
        'clinical_confidence': clinical_confidence,
        'reconstruction_results': results
    }

# Example usage
xray_results = validate_chest_xray_understanding("examples/chest_xray.jpg")
```

### Example 2: MRI Scan Comprehension Test

```python
def test_mri_comprehension(mri_path, scan_type="brain"):
    """
    Test MRI scan comprehension through autonomous reconstruction
    
    Different scan types require different levels of understanding
    """
    
    mri_image = cv2.imread(mri_path)
    
    print(f"🧠 MRI {scan_type.upper()} SCAN COMPREHENSION TEST")
    
    # Configure for MRI analysis
    engine = AutonomousReconstructionEngine(
        patch_size=24,      # Medium patches for MRI detail
        context_size=72,    # Good context for tissue boundaries
        device="cuda"
    )
    
    # Comprehensive analysis with medical validation
    analysis_engine = ComprehensiveAnalysisEngine()
    
    results = analysis_engine.comprehensive_analysis(
        image=mri_image,
        metadata={
            'domain': 'medical',
            'modality': 'MRI',
            'scan_type': scan_type,
            'requires_high_precision': True
        },
        enable_autonomous_reconstruction=True,
        enable_iterative_learning=True
    )
    
    # Extract comprehensive results
    assessment = results['final_assessment']
    reconstruction_data = results['autonomous_reconstruction']
    
    print(f"\n🎯 COMPREHENSIVE ASSESSMENT:")
    print(f"Understanding Demonstrated: {assessment['understanding_demonstrated']}")
    print(f"Confidence Score: {assessment['confidence_score']:.1%}")
    
    # Cross-validation with supporting methods
    if 'cross_validation' in results:
        cross_val = results['cross_validation']
        print(f"\n🔍 CROSS-VALIDATION:")
        print(f"Validation Status: {cross_val['understanding_validation']['status']}")
        print(f"Support Ratio: {cross_val['understanding_validation']['support_ratio']:.1%}")
    
    # MRI-specific insights
    quality = reconstruction_data['autonomous_reconstruction']['final_quality']
    
    if scan_type == "brain":
        if quality > 0.90:
            print("✅ Excellent brain tissue differentiation demonstrated")
        elif quality > 0.75:
            print("⚠️ Good brain structure understanding with some limitations")
        else:
            print("❌ Insufficient brain anatomy comprehension")
    
    return results

# Example usage
brain_mri_results = test_mri_comprehension("examples/brain_mri.jpg", "brain")
```

## 🏭 Industrial Quality Control

### Example 3: Manufacturing Defect Detection

```python
def detect_manufacturing_defects(product_image_path, reference_image_path):
    """
    Detect manufacturing defects by comparing reconstruction quality
    
    Insight: Defective products will be harder to reconstruct accurately
    because they deviate from learned patterns
    """
    
    product_image = cv2.imread(product_image_path)
    reference_image = cv2.imread(reference_image_path)
    
    print("🏭 MANUFACTURING DEFECT DETECTION")
    print("Method: Compare reconstruction quality between reference and product")
    
    engine = AutonomousReconstructionEngine(
        patch_size=32,
        context_size=96,
        device="cuda"
    )
    
    # Analyze reference product (should reconstruct well)
    print("\n📋 Analyzing reference product...")
    ref_results = engine.autonomous_analyze(
        image=reference_image,
        max_iterations=40,
        target_quality=0.85
    )
    
    ref_quality = ref_results['autonomous_reconstruction']['final_quality']
    print(f"Reference Quality: {ref_quality:.1%}")
    
    # Analyze test product
    print("\n🔍 Analyzing test product...")
    prod_results = engine.autonomous_analyze(
        image=product_image,
        max_iterations=40,
        target_quality=0.85
    )
    
    prod_quality = prod_results['autonomous_reconstruction']['final_quality']
    print(f"Product Quality: {prod_quality:.1%}")
    
    # Compare reconstruction qualities
    quality_difference = ref_quality - prod_quality
    
    print(f"\n📊 DEFECT ANALYSIS:")
    print(f"Quality Difference: {quality_difference:.1%}")
    
    if quality_difference > 0.15:
        defect_status = "MAJOR DEFECT DETECTED"
        confidence = "HIGH"
        action = "REJECT - Significant quality issues"
    elif quality_difference > 0.08:
        defect_status = "MINOR DEFECT DETECTED"
        confidence = "MODERATE"
        action = "REVIEW - Possible quality issues"
    elif quality_difference > 0.03:
        defect_status = "SLIGHT VARIATION"
        confidence = "LOW"
        action = "MONITOR - Within acceptable range"
    else:
        defect_status = "NO DEFECTS DETECTED"
        confidence = "HIGH"
        action = "ACCEPT - Quality matches reference"
    
    print(f"Status: {defect_status}")
    print(f"Confidence: {confidence}")
    print(f"Recommended Action: {action}")
    
    # Detailed analysis of reconstruction patterns
    ref_history = ref_results['reconstruction_history']
    prod_history = prod_results['reconstruction_history']
    
    if len(ref_history) != len(prod_history):
        print(f"\n⚠️ Convergence Difference: Reference took {len(ref_history)} iterations, "
              f"product took {len(prod_history)} iterations")
    
    return {
        'defect_status': defect_status,
        'quality_difference': quality_difference,
        'confidence': confidence,
        'recommended_action': action,
        'reference_quality': ref_quality,
        'product_quality': prod_quality
    }

# Example usage
defect_results = detect_manufacturing_defects(
    "examples/product_test.jpg", 
    "examples/product_reference.jpg"
)
```

## 🛰️ Satellite Imagery Analysis

### Example 4: Land Use Change Detection

```python
def detect_land_use_changes(before_image_path, after_image_path):
    """
    Detect land use changes by comparing reconstruction understanding
    
    Principle: Significant changes will result in different reconstruction patterns
    """
    
    before_image = cv2.imread(before_image_path)
    after_image = cv2.imread(after_image_path)
    
    print("🛰️ LAND USE CHANGE DETECTION")
    print("Method: Compare reconstruction patterns between time periods")
    
    engine = AutonomousReconstructionEngine(
        patch_size=48,      # Larger patches for satellite imagery
        context_size=144,   # More context for landscape features
        device="cuda"
    )
    
    # Analyze "before" image
    print("\n📅 Analyzing BEFORE image...")
    before_results = engine.autonomous_analyze(
        image=before_image,
        max_iterations=50,
        target_quality=0.80
    )
    
    # Analyze "after" image
    print("\n📅 Analyzing AFTER image...")
    after_results = engine.autonomous_analyze(
        image=after_image,
        max_iterations=50,
        target_quality=0.80
    )
    
    # Compare reconstruction patterns
    before_quality = before_results['autonomous_reconstruction']['final_quality']
    after_quality = after_results['autonomous_reconstruction']['final_quality']
    
    before_understanding = before_results['understanding_insights']['understanding_level']
    after_understanding = after_results['understanding_insights']['understanding_level']
    
    print(f"\n📊 TEMPORAL ANALYSIS:")
    print(f"Before - Quality: {before_quality:.1%}, Understanding: {before_understanding}")
    print(f"After - Quality: {after_quality:.1%}, Understanding: {after_understanding}")
    
    # Analyze reconstruction histories for change patterns
    before_history = before_results['reconstruction_history']
    after_history = after_results['reconstruction_history']
    
    # Calculate change indicators
    quality_change = abs(before_quality - after_quality)
    convergence_change = abs(len(before_history) - len(after_history))
    
    print(f"\n🔍 CHANGE INDICATORS:")
    print(f"Quality Change: {quality_change:.1%}")
    print(f"Convergence Difference: {convergence_change} iterations")
    
    # Determine change significance
    if quality_change > 0.20 or convergence_change > 15:
        change_level = "MAJOR CHANGES DETECTED"
        change_confidence = "HIGH"
    elif quality_change > 0.10 or convergence_change > 8:
        change_level = "MODERATE CHANGES DETECTED"
        change_confidence = "MODERATE"
    elif quality_change > 0.05 or convergence_change > 3:
        change_level = "MINOR CHANGES DETECTED"
        change_confidence = "LOW"
    else:
        change_level = "NO SIGNIFICANT CHANGES"
        change_confidence = "HIGH"
    
    print(f"\n🌍 CHANGE ASSESSMENT:")
    print(f"Change Level: {change_level}")
    print(f"Confidence: {change_confidence}")
    
    return {
        'change_level': change_level,
        'quality_change': quality_change,
        'convergence_change': convergence_change,
        'before_results': before_results,
        'after_results': after_results
    }

# Example usage
land_change_results = detect_land_use_changes(
    "examples/satellite_2020.jpg",
    "examples/satellite_2024.jpg"
)
```

## 🎨 Art and Cultural Analysis

### Example 5: Artistic Style Understanding

```python
def analyze_artistic_style(artwork_path, artist_name=None):
    """
    Analyze artistic style through reconstruction patterns
    
    Different artistic styles will show different reconstruction characteristics
    """
    
    artwork = cv2.imread(artwork_path)
    
    print(f"🎨 ARTISTIC STYLE ANALYSIS")
    if artist_name:
        print(f"Artist: {artist_name}")
    
    # Use comprehensive analysis for art
    analysis_engine = ComprehensiveAnalysisEngine()
    
    results = analysis_engine.comprehensive_analysis(
        image=artwork,
        metadata={
            'domain': 'art',
            'artist': artist_name,
            'type': 'painting'
        },
        enable_autonomous_reconstruction=True,
        enable_iterative_learning=True
    )
    
    # Extract artistic insights
    reconstruction_data = results['autonomous_reconstruction']
    quality = reconstruction_data['autonomous_reconstruction']['final_quality']
    understanding = reconstruction_data['understanding_insights']['understanding_level']
    
    print(f"\n🎭 STYLE ANALYSIS:")
    print(f"Reconstruction Quality: {quality:.1%}")
    print(f"Understanding Level: {understanding}")
    
    # Analyze reconstruction patterns for artistic insights
    history = reconstruction_data['reconstruction_history']
    
    # Calculate style characteristics
    quality_progression = [h['quality'] for h in history]
    confidence_progression = [h['confidence'] for h in history]
    
    quality_variance = np.var(quality_progression)
    confidence_variance = np.var(confidence_progression)
    
    print(f"\n🖼️ STYLE CHARACTERISTICS:")
    print(f"Quality Variance: {quality_variance:.4f}")
    print(f"Confidence Variance: {confidence_variance:.4f}")
    print(f"Iterations to Convergence: {len(history)}")
    
    # Artistic style interpretation
    if quality > 0.85 and quality_variance < 0.01:
        style_interpretation = "HIGHLY STRUCTURED - Realistic or classical style with clear patterns"
    elif quality > 0.70 and quality_variance > 0.02:
        style_interpretation = "MODERATELY COMPLEX - Impressionistic or mixed style"
    elif quality < 0.60 and confidence_variance > 0.03:
        style_interpretation = "HIGHLY ABSTRACT - Abstract or experimental style"
    else:
        style_interpretation = "UNIQUE STYLE - Distinctive artistic approach"
    
    print(f"\n🎨 STYLE INTERPRETATION:")
    print(f"{style_interpretation}")
    
    # Cross-validation insights
    if 'cross_validation' in results:
        cross_val = results['cross_validation']
        support_ratio = cross_val['understanding_validation']['support_ratio']
        
        if support_ratio > 0.8:
            print("✅ Style analysis strongly supported by multiple methods")
        elif support_ratio > 0.6:
            print("⚠️ Style analysis moderately supported")
        else:
            print("❌ Style analysis shows conflicting evidence")
    
    return {
        'style_interpretation': style_interpretation,
        'reconstruction_quality': quality,
        'style_characteristics': {
            'quality_variance': quality_variance,
            'confidence_variance': confidence_variance,
            'convergence_iterations': len(history)
        },
        'full_results': results
    }

# Example usage
art_results = analyze_artistic_style("examples/van_gogh_starry_night.jpg", "Van Gogh")
```

## 🔬 Scientific Research Applications

### Example 6: Microscopy Image Understanding

```python
def validate_microscopy_understanding(microscopy_path, magnification, specimen_type):
    """
    Validate understanding of microscopy images through reconstruction
    
    Critical for scientific research - can the AI truly see cellular structures?
    """
    
    microscopy_image = cv2.imread(microscopy_path)
    
    print(f"🔬 MICROSCOPY IMAGE UNDERSTANDING VALIDATION")
    print(f"Specimen: {specimen_type}")
    print(f"Magnification: {magnification}x")
    
    # Configure for microscopy analysis
    engine = AutonomousReconstructionEngine(
        patch_size=16,      # Small patches for cellular detail
        context_size=48,    # Limited context for microscopy
        device="cuda"
    )
    
    # High-precision analysis for scientific use
    results = engine.autonomous_analyze(
        image=microscopy_image,
        max_iterations=100,  # Many iterations for scientific precision
        target_quality=0.95  # Very high quality for research
    )
    
    quality = results['autonomous_reconstruction']['final_quality']
    understanding = results['understanding_insights']['understanding_level']
    
    print(f"\n🔬 SCIENTIFIC VALIDATION:")
    print(f"Reconstruction Quality: {quality:.1%}")
    print(f"Understanding Level: {understanding}")
    
    # Scientific assessment
    if quality > 0.95:
        scientific_confidence = "EXCELLENT - Suitable for quantitative analysis"
        research_grade = "PUBLICATION READY"
    elif quality > 0.90:
        scientific_confidence = "VERY GOOD - Suitable for qualitative analysis"
        research_grade = "RESEARCH SUITABLE"
    elif quality > 0.80:
        scientific_confidence = "GOOD - Suitable for preliminary analysis"
        research_grade = "PRELIMINARY ONLY"
    else:
        scientific_confidence = "INSUFFICIENT - Not suitable for scientific use"
        research_grade = "NOT SUITABLE"
    
    print(f"Scientific Confidence: {scientific_confidence}")
    print(f"Research Grade: {research_grade}")
    
    # Analyze cellular structure understanding
    history = results['reconstruction_history']
    
    # Look for patterns indicating cellular structure recognition
    quality_improvements = []
    for i in range(1, len(history)):
        improvement = history[i]['quality'] - history[i-1]['quality']
        quality_improvements.append(improvement)
    
    avg_improvement = np.mean(quality_improvements) if quality_improvements else 0
    
    print(f"\n📈 LEARNING ANALYSIS:")
    print(f"Average Quality Improvement per Iteration: {avg_improvement:.4f}")
    
    if avg_improvement > 0.01:
        print("✅ Strong learning progression - good cellular structure recognition")
    elif avg_improvement > 0.005:
        print("⚠️ Moderate learning - some cellular features recognized")
    else:
        print("❌ Limited learning - poor cellular structure recognition")
    
    return {
        'scientific_confidence': scientific_confidence,
        'research_grade': research_grade,
        'reconstruction_quality': quality,
        'learning_progression': avg_improvement,
        'full_results': results
    }

# Example usage
microscopy_results = validate_microscopy_understanding(
    "examples/cell_culture.jpg", 
    magnification=400, 
    specimen_type="HeLa cells"
)
```

## 🚗 Autonomous Vehicle Vision

### Example 7: Scene Understanding for Self-Driving Cars

```python
def validate_autonomous_vehicle_vision(street_scene_path):
    """
    Validate scene understanding for autonomous vehicles
    
    Critical safety application - can the AI truly understand the road scene?
    """
    
    street_scene = cv2.imread(street_scene_path)
    
    print("🚗 AUTONOMOUS VEHICLE VISION VALIDATION")
    print("Safety-critical application: Can AI truly see the road?")
    
    # Use comprehensive analysis for safety validation
    analysis_engine = ComprehensiveAnalysisEngine()
    
    results = analysis_engine.comprehensive_analysis(
        image=street_scene,
        metadata={
            'domain': 'autonomous_vehicle',
            'scene_type': 'street',
            'safety_critical': True
        },
        enable_autonomous_reconstruction=True,
        enable_iterative_learning=True
    )
    
    # Extract safety-critical metrics
    assessment = results['final_assessment']
    reconstruction_data = results['autonomous_reconstruction']
    
    quality = reconstruction_data['autonomous_reconstruction']['final_quality']
    understanding = reconstruction_data['understanding_insights']['understanding_level']
    
    print(f"\n🛡️ SAFETY ASSESSMENT:")
    print(f"Scene Understanding: {understanding}")
    print(f"Reconstruction Quality: {quality:.1%}")
    print(f"Understanding Demonstrated: {assessment['understanding_demonstrated']}")
    
    # Safety classification
    if quality > 0.95 and assessment['understanding_demonstrated']:
        safety_level = "SAFE FOR AUTONOMOUS OPERATION"
        confidence = "HIGH"
        recommendation = "APPROVED - System demonstrates complete scene understanding"
    elif quality > 0.85:
        safety_level = "CONDITIONAL SAFETY"
        confidence = "MODERATE"
        recommendation = "CAUTION - Good understanding but requires monitoring"
    elif quality > 0.70:
        safety_level = "LIMITED SAFETY"
        confidence = "LOW"
        recommendation = "RESTRICTED - Limited understanding, human oversight required"
    else:
        safety_level = "UNSAFE FOR AUTONOMOUS OPERATION"
        confidence = "VERY LOW"
        recommendation = "REJECTED - Insufficient scene understanding"
    
    print(f"\n🚦 SAFETY CLASSIFICATION:")
    print(f"Safety Level: {safety_level}")
    print(f"Confidence: {confidence}")
    print(f"Recommendation: {recommendation}")
    
    # Cross-validation for safety
    if 'cross_validation' in results:
        cross_val = results['cross_validation']
        support_ratio = cross_val['understanding_validation']['support_ratio']
        
        print(f"\n🔍 SAFETY CROSS-VALIDATION:")
        print(f"Support Ratio: {support_ratio:.1%}")
        
        if support_ratio > 0.9:
            print("✅ EXCELLENT - All validation methods support reconstruction insights")
        elif support_ratio > 0.7:
            print("⚠️ GOOD - Most validation methods support reconstruction insights")
        else:
            print("❌ POOR - Conflicting evidence from validation methods")
    
    return {
        'safety_level': safety_level,
        'reconstruction_quality': quality,
        'safety_confidence': confidence,
        'recommendation': recommendation,
        'full_results': results
    }

# Example usage
av_results = validate_autonomous_vehicle_vision("examples/street_scene.jpg")
```

## 📊 Batch Processing Example

### Example 8: Large-Scale Dataset Analysis

```python
def analyze_dataset_with_reconstruction(image_directory, output_directory):
    """
    Analyze entire datasets using autonomous reconstruction
    
    Demonstrates scalability of the reconstruction approach
    """
    
    import os
    import json
    from pathlib import Path
    
    print("📊 LARGE-SCALE DATASET ANALYSIS")
    print("Method: Autonomous reconstruction for every image")
    
    # Setup
    image_dir = Path(image_directory)
    output_dir = Path(output_directory)
    output_dir.mkdir(exist_ok=True)
    
    # Get all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(image_dir.glob(f"*{ext}"))
        image_paths.extend(image_dir.glob(f"*{ext.upper()}"))
    
    print(f"Found {len(image_paths)} images to analyze")
    
    # Initialize engines
    reconstruction_engine = AutonomousReconstructionEngine(
        patch_size=32,
        context_size=96,
        device="cuda"
    )
    
    # Batch analysis
    results = []
    failed_analyses = []
    
    for i, image_path in enumerate(image_paths):
        print(f"\nProcessing {i+1}/{len(image_paths)}: {image_path.name}")
        
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"❌ Failed to load {image_path.name}")
                continue
            
            # Analyze
            result = reconstruction_engine.autonomous_analyze(
                image=image,
                max_iterations=30,
                target_quality=0.80
            )
            
            # Add metadata
            result['image_path'] = str(image_path)
            result['image_name'] = image_path.name
            result['analysis_index'] = i
            
            results.append(result)
            
            # Save individual result
            result_file = output_dir / f"analysis_{i:04d}_{image_path.stem}.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
            # Progress update
            quality = result['autonomous_reconstruction']['final_quality']
            understanding = result['understanding_insights']['understanding_level']
            print(f"✅ Quality: {quality:.1%}, Understanding: {understanding}")
            
        except Exception as e:
            print(f"❌ Error processing {image_path.name}: {e}")
            failed_analyses.append({
                'image_path': str(image_path),
                'error': str(e)
            })
    
    # Generate dataset summary
    print(f"\n📈 GENERATING DATASET SUMMARY...")
    
    qualities = [r['autonomous_reconstruction']['final_quality'] for r in results]
    understanding_levels = [r['understanding_insights']['understanding_level'] for r in results]
    iterations = [r['autonomous_reconstruction']['reconstruction_iterations'] for r in results]
    
    summary = {
        'dataset_info': {
            'total_images': len(image_paths),
            'successfully_analyzed': len(results),
            'failed_analyses': len(failed_analyses),
            'success_rate': len(results) / len(image_paths) * 100
        },
        'reconstruction_statistics': {
            'average_quality': float(np.mean(qualities)),
            'quality_std': float(np.std(qualities)),
            'min_quality': float(np.min(qualities)),
            'max_quality': float(np.max(qualities)),
            'median_quality': float(np.median(qualities))
        },
        'understanding_distribution': {
            level: understanding_levels.count(level) 
            for level in set(understanding_levels)
        },
        'performance_metrics': {
            'average_iterations': float(np.mean(iterations)),
            'iteration_std': float(np.std(iterations)),
            'high_quality_count': sum(1 for q in qualities if q > 0.9),
            'low_quality_count': sum(1 for q in qualities if q < 0.6)
        }
    }
    
    # Save summary
    summary_file = output_dir / "dataset_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print(f"\n📊 DATASET ANALYSIS COMPLETE")
    print(f"Success Rate: {summary['dataset_info']['success_rate']:.1f}%")
    print(f"Average Quality: {summary['reconstruction_statistics']['average_quality']:.1%}")
    print(f"High Quality Images (>90%): {summary['performance_metrics']['high_quality_count']}")
    print(f"Understanding Distribution: {summary['understanding_distribution']}")
    
    return summary, results, failed_analyses

# Example usage
dataset_summary, all_results, failures = analyze_dataset_with_reconstruction(
    "datasets/test_images/",
    "outputs/dataset_analysis/"
)
```

---

<div style="background: #e8f4fd; padding: 1.5rem; border-radius: 8px; margin: 2rem 0; border-left: 4px solid #0366d6;">
<h3>💡 Key Insight from Examples</h3>
<p>Across all these diverse applications - from medical imaging to autonomous vehicles to art analysis - the same fundamental principle applies: <strong>reconstruction ability demonstrates understanding depth</strong>.</p>
<p>This universal metric works because it directly tests the AI's ability to "draw what it sees," providing an objective measure of visual comprehension that transcends domain-specific metrics.</p>
</div>

## 🎯 Running the Examples

### Prerequisites

```bash
# Install Helicopter with examples dependencies
pip install helicopter-cv[examples]

# Or install development version
git clone https://github.com/yourusername/helicopter.git
cd helicopter
pip install -e ".[examples]"
```

### Example Data

Download example datasets:

```bash
# Download example images
python -m helicopter.examples.download_data

# Or manually create examples directory
mkdir examples
# Add your own images to test with
```

### Running Individual Examples

```bash
# Medical imaging example
python examples/medical_imaging_demo.py

# Manufacturing quality control
python examples/manufacturing_demo.py

# Artistic style analysis
python examples/art_analysis_demo.py

# Full autonomous reconstruction demo
python examples/autonomous_reconstruction_demo.py
```

## 🔗 Related Documentation

- **[Getting Started](getting-started.html)** - Basic usage and installation
- **[Autonomous Reconstruction](autonomous-reconstruction.html)** - Core reconstruction engine
- **[Comprehensive Analysis](comprehensive-analysis.html)** - Full analysis pipeline
- **[API Reference](api-reference.html)** - Complete API documentation
