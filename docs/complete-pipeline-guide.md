# Complete Helicopter Pipeline Guide

This guide demonstrates how to use the complete Helicopter framework with all its modules working together for comprehensive image analysis.

## Overview

The Helicopter framework now includes seven major components that work together to provide comprehensive image analysis:

1. **MetacognitiveOrchestrator** - Ultimate coordination system
2. **AutonomousReconstructionEngine** - Primary reconstruction analysis
3. **SegmentAwareReconstructionEngine** - Complex scene processing
4. **ZengezaEngine** - Intelligent noise detection
5. **HatataEngine** - Probabilistic understanding verification
6. **NicotineContextValidator** - Context maintenance
7. **DiadochiCore** - Multi-domain expert combination

## Quick Start: Orchestrated Analysis

The **simplest and recommended approach** is to use the Metacognitive Orchestrator:

```python
from helicopter.core import MetacognitiveOrchestrator, AnalysisStrategy

# Initialize orchestrator (handles all modules automatically)
orchestrator = MetacognitiveOrchestrator()

# Single function call for comprehensive analysis
results = await orchestrator.orchestrated_analysis(
    image_path="path/to/your/image.jpg",
    analysis_goals=["comprehensive_understanding", "quality_assessment"],
    strategy=AnalysisStrategy.ADAPTIVE,  # Let orchestrator choose best approach
    time_budget=60.0,
    quality_threshold=0.85
)

# Review results
print(f"✅ Analysis Complete!")
print(f"📊 Overall Quality: {results['overall_quality']:.1%}")
print(f"🎯 Overall Confidence: {results['overall_confidence']:.1%}")
print(f"⚡ Strategy Used: {results['strategy_used']}")
print(f"🔧 Modules Executed: {results['modules_executed']}")
print(f"⏱️ Total Time: {results['execution_time']:.1f}s")
```

## Complete Manual Pipeline

For advanced users who want full control over the pipeline:

```python
from helicopter.core import (
    AutonomousReconstructionEngine,
    SegmentAwareReconstructionEngine,
    ZengezaEngine,
    HatataEngine,
    NicotineContextValidator,
    DiadochiCore
)

# Initialize all engines
autonomous_engine = AutonomousReconstructionEngine()
segment_engine = SegmentAwareReconstructionEngine()
zengeza_engine = ZengezaEngine()
hatata_engine = HatataEngine()
nicotine_validator = NicotineContextValidator()
diadochi_core = DiadochiCore()

# Step 1: Initialize context validation
print("🚬 Initializing context validation...")
validator_active = nicotine_validator.register_process(
    process_name="comprehensive_analysis",
    current_task="multi_module_image_analysis",
    objectives=["reconstruction", "understanding", "validation", "expert_synthesis"]
)

# Step 2: Noise analysis and prioritization
print("🔍 Analyzing noise and prioritizing segments...")
noise_results = await zengeza_engine.analyze_image_noise(
    image_path="path/to/image.jpg",
    analysis_depth="comprehensive"
)

print(f"   Overall Noise Level: {noise_results['overall_noise_level']:.1%}")
print(f"   Segments Identified: {noise_results['segments_count']}")
print(f"   Dominant Noise Type: {noise_results['dominant_noise_type']}")

# Step 3: Autonomous reconstruction
print("🎯 Performing autonomous reconstruction...")
auto_results = autonomous_engine.autonomous_analyze(
    image=image,
    max_iterations=30,
    target_quality=0.85,
    noise_analysis=noise_results  # Use noise analysis for guidance
)

print(f"   Reconstruction Quality: {auto_results['final_quality']:.1%}")
print(f"   Iterations Performed: {auto_results['iterations_performed']}")
print(f"   Understanding Level: {auto_results['understanding_level']}")

# Step 4: Segment-aware reconstruction (if needed)
if auto_results['final_quality'] < 0.8 or noise_results['overall_noise_level'] > 0.2:
    print("📊 Applying segment-aware reconstruction...")
    segment_results = segment_engine.segment_aware_reconstruction(
        image=image,
        description="Complex image requiring segment-wise analysis",
        noise_analysis=noise_results
    )
    
    print(f"   Segment-Aware Quality: {segment_results['overall_quality']:.1%}")
    print(f"   Segments Processed: {segment_results['segments_processed']}")
    print(f"   Successful Segments: {segment_results['successful_segments']}")
    
    # Use the better result
    if segment_results['overall_quality'] > auto_results['final_quality']:
        reconstruction_results = segment_results
        print("   ✅ Using segment-aware results (better quality)")
    else:
        reconstruction_results = auto_results
        print("   ✅ Using autonomous results (sufficient quality)")
else:
    reconstruction_results = auto_results
    print("   ✅ Autonomous reconstruction sufficient")

# Step 5: Probabilistic understanding verification
print("📈 Verifying understanding probabilistically...")
hatata_results = await hatata_engine.probabilistic_understanding_verification(
    image_path="path/to/image.jpg",
    reconstruction_data=reconstruction_results,
    confidence_threshold=0.8
)

print(f"   Understanding Probability: {hatata_results['understanding_probability']:.1%}")
print(f"   Confidence Bounds: [{hatata_results['confidence_lower']:.1%}, {hatata_results['confidence_upper']:.1%}]")
print(f"   Verification State: {hatata_results['final_state']}")

# Step 6: Expert synthesis
print("🏛️ Synthesizing expert analysis...")
synthesis_query = f"""
Analyze comprehensive image analysis results:
- Reconstruction quality: {reconstruction_results.get('final_quality', reconstruction_results.get('overall_quality', 0)):.1%}
- Noise level: {noise_results['overall_noise_level']:.1%}
- Understanding probability: {hatata_results['understanding_probability']:.1%}
- Context maintained: {validator_active}

Provide integrated assessment, key insights, and recommendations.
"""

expert_synthesis = await diadochi_core.generate(synthesis_query)
print(f"   Expert Analysis: {expert_synthesis[:200]}...")

# Step 7: Final context validation check
print("🚬 Final context validation...")
final_validation = nicotine_validator.get_validation_report()
print(f"   Context Maintained: {final_validation['pass_rate']:.1%}")
print(f"   Validation Sessions: {final_validation['total_sessions']}")

# Step 8: Compile comprehensive results
print("📋 Compiling comprehensive results...")
comprehensive_results = {
    'analysis_success': True,
    'overall_quality': max(
        reconstruction_results.get('final_quality', 0),
        reconstruction_results.get('overall_quality', 0)
    ),
    'noise_analysis': noise_results,
    'reconstruction_results': reconstruction_results,
    'understanding_verification': hatata_results,
    'expert_synthesis': expert_synthesis,
    'context_validation': final_validation,
    'processing_time': time.time() - start_time,
    'modules_used': ['zengeza', 'autonomous_reconstruction', 'hatata', 'diadochi', 'nicotine'],
    'recommendations': []
}

# Add recommendations based on results
if comprehensive_results['overall_quality'] > 0.9:
    comprehensive_results['recommendations'].append("Excellent analysis quality achieved")
elif comprehensive_results['overall_quality'] > 0.75:
    comprehensive_results['recommendations'].append("Good analysis quality, consider domain-specific optimization")
else:
    comprehensive_results['recommendations'].append("Consider additional processing or expert review")

if noise_results['overall_noise_level'] > 0.3:
    comprehensive_results['recommendations'].append("High noise detected, consider preprocessing")

if hatata_results['understanding_probability'] < 0.8:
    comprehensive_results['recommendations'].append("Understanding confidence low, consider additional validation")

print("\n🎉 Comprehensive Analysis Complete!")
print(f"📊 Final Quality: {comprehensive_results['overall_quality']:.1%}")
print(f"⏱️ Total Processing Time: {comprehensive_results['processing_time']:.1f}s")
print(f"🔧 Modules Used: {', '.join(comprehensive_results['modules_used'])}")
print(f"💡 Recommendations: {len(comprehensive_results['recommendations'])}")
for rec in comprehensive_results['recommendations']:
    print(f"   • {rec}")
```

## Domain-Specific Pipelines

### Medical Imaging Pipeline

```python
# Medical imaging requires highest quality and confidence
async def medical_imaging_pipeline(image_path, patient_info=None):
    orchestrator = MetacognitiveOrchestrator()
    
    # Configure for medical domain
    medical_config = {
        'quality_thresholds': {'minimum': 0.9, 'good': 0.95, 'excellent': 0.98},
        'confidence_thresholds': {'minimum': 0.8, 'good': 0.9, 'high': 0.95},
        'enable_all_modules': True,
        'domain_specific_validation': True
    }
    
    results = await orchestrator.orchestrated_analysis(
        image_path=image_path,
        analysis_goals=[
            "medical_diagnosis_support",
            "anomaly_detection", 
            "quality_assessment",
            "uncertainty_quantification"
        ],
        strategy=AnalysisStrategy.DEEP_ANALYSIS,
        time_budget=120.0,  # Allow more time for thorough analysis
        quality_threshold=0.95,
        config=medical_config
    )
    
    # Medical-specific reporting
    medical_report = {
        'diagnostic_confidence': results['overall_confidence'],
        'image_quality_score': results['overall_quality'],
        'uncertainty_bounds': results.get('uncertainty_analysis', {}),
        'anomaly_detection': results.get('anomaly_findings', []),
        'quality_indicators': results.get('quality_indicators', {}),
        'recommendations': results.get('recommendations', [])
    }
    
    return medical_report

# Usage
medical_results = await medical_imaging_pipeline("chest_xray.jpg")
print(f"Medical Analysis - Confidence: {medical_results['diagnostic_confidence']:.1%}")
```

### Real-Time Processing Pipeline

```python
# Real-time processing prioritizes speed
async def realtime_processing_pipeline(image_path):
    orchestrator = MetacognitiveOrchestrator()
    
    # Configure for real-time processing
    realtime_config = {
        'max_parallel_modules': 4,
        'enable_fast_mode': True,
        'quality_thresholds': {'minimum': 0.7, 'good': 0.8, 'excellent': 0.9}
    }
    
    results = await orchestrator.orchestrated_analysis(
        image_path=image_path,
        analysis_goals=["basic_understanding", "quality_assessment"],
        strategy=AnalysisStrategy.SPEED_OPTIMIZED,
        time_budget=5.0,  # Strict time limit
        quality_threshold=0.75,
        config=realtime_config
    )
    
    return {
        'success': results['success'],
        'quality': results['overall_quality'],
        'processing_time': results['execution_time'],
        'modules_used': results['modules_executed']
    }

# Usage
realtime_results = await realtime_processing_pipeline("webcam_frame.jpg")
print(f"Real-time Analysis - Quality: {realtime_results['quality']:.1%} in {realtime_results['processing_time']:.1f}s")
```

### Research Pipeline

```python
# Research pipeline uses all modules for maximum insight
async def research_pipeline(image_path, research_objectives):
    orchestrator = MetacognitiveOrchestrator()
    
    # Configure for research
    research_config = {
        'enable_learning': True,
        'comprehensive_logging': True,
        'save_intermediate_results': True,
        'generate_detailed_insights': True
    }
    
    results = await orchestrator.orchestrated_analysis(
        image_path=image_path,
        analysis_goals=research_objectives + [
            "comprehensive_understanding",
            "uncertainty_quantification",
            "method_validation",
            "insight_generation"
        ],
        strategy=AnalysisStrategy.DEEP_ANALYSIS,
        time_budget=300.0,  # Allow extensive processing
        quality_threshold=0.9,
        config=research_config
    )
    
    # Generate research report
    research_report = {
        'methodology': results.get('methodology_used', {}),
        'quantitative_results': results.get('quantitative_metrics', {}),
        'qualitative_insights': results.get('metacognitive_insights', []),
        'validation_results': results.get('validation_summary', {}),
        'limitations': results.get('analysis_limitations', []),
        'future_directions': results.get('recommendations', [])
    }
    
    return research_report

# Usage
research_results = await research_pipeline(
    "research_image.jpg",
    ["novel_method_validation", "comparative_analysis"]
)
```

## Batch Processing

```python
# Process multiple images efficiently
async def batch_processing_pipeline(image_paths, strategy=AnalysisStrategy.BALANCED):
    orchestrator = MetacognitiveOrchestrator()
    results = []
    
    for i, image_path in enumerate(image_paths):
        print(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
        
        result = await orchestrator.orchestrated_analysis(
            image_path=image_path,
            strategy=strategy,
            time_budget=30.0
        )
        
        results.append({
            'image_path': image_path,
            'quality': result['overall_quality'],
            'confidence': result['overall_confidence'],
            'processing_time': result['execution_time'],
            'success': result['success']
        })
        
        print(f"   Quality: {result['overall_quality']:.1%}, Time: {result['execution_time']:.1f}s")
    
    # Batch summary
    avg_quality = sum(r['quality'] for r in results) / len(results)
    avg_time = sum(r['processing_time'] for r in results) / len(results)
    success_rate = sum(r['success'] for r in results) / len(results)
    
    print(f"\n📊 Batch Summary:")
    print(f"   Average Quality: {avg_quality:.1%}")
    print(f"   Average Time: {avg_time:.1f}s")
    print(f"   Success Rate: {success_rate:.1%}")
    
    return results

# Usage
batch_results = await batch_processing_pipeline([
    "image1.jpg", "image2.jpg", "image3.jpg"
])
```

## Monitoring and Optimization

```python
# Monitor pipeline performance
class PipelineMonitor:
    def __init__(self):
        self.orchestrator = MetacognitiveOrchestrator()
        self.performance_history = []
    
    async def analyze_with_monitoring(self, image_path, **kwargs):
        start_time = time.time()
        
        results = await self.orchestrator.orchestrated_analysis(
            image_path=image_path,
            **kwargs
        )
        
        # Record performance
        performance_record = {
            'timestamp': time.time(),
            'image_path': image_path,
            'quality': results['overall_quality'],
            'confidence': results['overall_confidence'],
            'processing_time': results['execution_time'],
            'strategy': results['strategy_used'],
            'modules': results['modules_executed'],
            'success': results['success']
        }
        
        self.performance_history.append(performance_record)
        
        return results
    
    def get_performance_summary(self):
        if not self.performance_history:
            return {"message": "No analysis history available"}
        
        recent_analyses = self.performance_history[-10:]  # Last 10 analyses
        
        return {
            'total_analyses': len(self.performance_history),
            'recent_avg_quality': sum(a['quality'] for a in recent_analyses) / len(recent_analyses),
            'recent_avg_time': sum(a['processing_time'] for a in recent_analyses) / len(recent_analyses),
            'success_rate': sum(a['success'] for a in recent_analyses) / len(recent_analyses),
            'most_used_strategy': max(set(a['strategy'] for a in recent_analyses), 
                                    key=[a['strategy'] for a in recent_analyses].count),
            'optimization_recommendations': self._generate_optimization_recommendations()
        }
    
    def _generate_optimization_recommendations(self):
        if len(self.performance_history) < 5:
            return ["Collect more data for optimization recommendations"]
        
        recommendations = []
        recent = self.performance_history[-10:]
        
        avg_quality = sum(a['quality'] for a in recent) / len(recent)
        avg_time = sum(a['processing_time'] for a in recent) / len(recent)
        
        if avg_quality < 0.8:
            recommendations.append("Consider using QUALITY_OPTIMIZED strategy more often")
        
        if avg_time > 60:
            recommendations.append("Consider using SPEED_OPTIMIZED strategy for time-sensitive applications")
        
        if sum(a['success'] for a in recent) / len(recent) < 0.9:
            recommendations.append("Review input image quality and preprocessing")
        
        return recommendations

# Usage
monitor = PipelineMonitor()

# Analyze with monitoring
results = await monitor.analyze_with_monitoring(
    "test_image.jpg",
    strategy=AnalysisStrategy.ADAPTIVE
)

# Get performance insights
performance_summary = monitor.get_performance_summary()
print(f"Performance Summary: {performance_summary}")
```

## Best Practices

### 1. Strategy Selection
- **ADAPTIVE**: Default choice for unknown image types
- **SPEED_OPTIMIZED**: Real-time applications, live video processing
- **BALANCED**: General purpose, good quality-speed tradeoff
- **QUALITY_OPTIMIZED**: Critical applications, medical imaging
- **DEEP_ANALYSIS**: Research, comprehensive analysis needs

### 2. Time Budget Management
- Real-time: 1-5 seconds
- Interactive: 5-15 seconds
- Batch processing: 15-60 seconds
- Research: 60+ seconds

### 3. Quality Thresholds
- Basic applications: 0.7+
- General applications: 0.8+
- Critical applications: 0.9+
- Research applications: 0.95+

### 4. Module Selection
- **Always include**: Autonomous Reconstruction, Noise Detection
- **Complex images**: Add Segment-Aware Reconstruction
- **Critical applications**: Add Probabilistic Validation
- **Long processes**: Add Context Validation
- **Multi-domain**: Add Expert Synthesis

## Troubleshooting Guide

### Common Issues and Solutions

1. **Low Overall Quality**
   ```python
   # Check individual module results
   for module, result in results['module_results'].items():
       if result['quality_score'] < 0.7:
           print(f"Low quality in {module}: {result['quality_score']:.1%}")
   
   # Try higher quality strategy
   results = await orchestrator.orchestrated_analysis(
       image_path=image_path,
       strategy=AnalysisStrategy.QUALITY_OPTIMIZED
   )
   ```

2. **Long Processing Times**
   ```python
   # Use speed-optimized strategy
   results = await orchestrator.orchestrated_analysis(
       image_path=image_path,
       strategy=AnalysisStrategy.SPEED_OPTIMIZED,
       time_budget=10.0
   )
   ```

3. **Module Failures**
   ```python
   # Check module-specific errors
   for module, result in results['module_results'].items():
       if not result['success']:
           print(f"Module {module} failed: {result.get('error', 'Unknown error')}")
   ```

4. **Inconsistent Results**
   ```python
   # Enable learning for consistency
   orchestrator_config = {'enable_learning': True}
   orchestrator = MetacognitiveOrchestrator(config=orchestrator_config)
   ```

## Performance Optimization Tips

1. **Enable Parallel Processing**
   ```python
   config = {'max_parallel_modules': 4}
   orchestrator = MetacognitiveOrchestrator(config=config)
   ```

2. **Optimize for Your Domain**
   ```python
   # Medical imaging
   medical_config = {
       'quality_thresholds': {'minimum': 0.9},
       'enable_probabilistic_validation': True
   }
   
   # Real-time processing
   realtime_config = {
       'max_parallel_modules': 2,
       'enable_fast_mode': True
   }
   ```

3. **Use Appropriate Strategies**
   ```python
   # Pre-classify images and use appropriate strategies
   if image_complexity == 'simple':
       strategy = AnalysisStrategy.SPEED_OPTIMIZED
   elif image_complexity == 'complex':
       strategy = AnalysisStrategy.QUALITY_OPTIMIZED
   else:
       strategy = AnalysisStrategy.ADAPTIVE
   ```

## Conclusion

The complete Helicopter pipeline provides unprecedented capabilities for image analysis:

- **Intelligent Coordination**: Metacognitive orchestrator manages all modules
- **Comprehensive Analysis**: Seven specialized modules work together
- **Adaptive Processing**: Strategies adapt to image complexity and requirements
- **Quality Assurance**: Multiple validation and verification layers
- **Continuous Learning**: System improves over time through experience

For most users, the **Metacognitive Orchestrator with ADAPTIVE strategy** provides the best balance of quality, speed, and ease of use. Advanced users can customize the pipeline for specific domains and requirements.

The framework represents a complete solution for autonomous visual understanding that proves comprehension through reconstruction while providing comprehensive analysis through specialized modules. 