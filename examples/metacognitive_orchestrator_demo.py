"""
Metacognitive Orchestrator Demo

Demonstrates the comprehensive metacognitive orchestrator that intelligently
coordinates all Helicopter modules for optimal image analysis.

The orchestrator uses metacognitive principles to:
1. Assess image complexity and adapt analysis strategy
2. Intelligently select and coordinate modules
3. Learn from analysis outcomes
4. Provide comprehensive insights
"""

import asyncio
import logging
import json
from pathlib import Path
from typing import List, Dict, Any

# Import the metacognitive orchestrator and supporting classes
from helicopter.core import (
    MetacognitiveOrchestrator,
    AnalysisStrategy,
    ImageComplexity,
    AnalysisContext,
    ModuleResult,
    PipelineState,
    MetacognitiveInsight
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MetacognitiveDemo:
    """
    Comprehensive demonstration of the Metacognitive Orchestrator.
    
    Shows how to intelligently coordinate all Helicopter modules for
    optimal image analysis using metacognitive principles.
    """
    
    def __init__(self):
        """Initialize the demonstration."""
        self.orchestrator = MetacognitiveOrchestrator()
        self.demo_images = self._get_demo_images()
        
    def _get_demo_images(self) -> List[Dict[str, Any]]:
        """Get demo images with different complexity levels."""
        return [
            {
                "path": "examples/images/simple_object.jpg",
                "description": "Simple object against plain background",
                "expected_complexity": ImageComplexity.SIMPLE,
                "analysis_goals": ["object_recognition", "basic_understanding"]
            },
            {
                "path": "examples/images/complex_scene.jpg", 
                "description": "Complex scene with multiple objects",
                "expected_complexity": ImageComplexity.COMPLEX,
                "analysis_goals": ["scene_understanding", "object_relationships", "spatial_analysis"]
            },
            {
                "path": "examples/images/noisy_image.jpg",
                "description": "Image with significant noise and artifacts",
                "expected_complexity": ImageComplexity.MODERATE,
                "analysis_goals": ["noise_analysis", "signal_recovery", "quality_assessment"]
            },
            {
                "path": "examples/images/highly_complex.jpg",
                "description": "Highly complex image with fine details",
                "expected_complexity": ImageComplexity.HIGHLY_COMPLEX,
                "analysis_goals": ["detailed_analysis", "fine_structure_detection", "comprehensive_understanding"]
            }
        ]
    
    async def run_comprehensive_demo(self):
        """Run comprehensive demonstration of all orchestrator capabilities."""
        
        print("🧠 METACOGNITIVE ORCHESTRATOR DEMONSTRATION")
        print("=" * 60)
        
        # Demo 1: Basic orchestrated analysis
        await self._demo_basic_orchestration()
        
        # Demo 2: Strategy comparison
        await self._demo_strategy_comparison()
        
        # Demo 3: Adaptive analysis
        await self._demo_adaptive_analysis()
        
        # Demo 4: Learning and adaptation
        await self._demo_learning_capabilities()
        
        # Demo 5: Metacognitive insights
        await self._demo_metacognitive_insights()
        
        # Demo 6: Performance optimization
        await self._demo_performance_optimization()
        
        print("\n✅ Comprehensive demonstration completed!")
        
    async def _demo_basic_orchestration(self):
        """Demonstrate basic orchestrated analysis."""
        print("\n📋 DEMO 1: Basic Orchestrated Analysis")
        print("-" * 40)
        
        # Use first demo image
        demo_image = self.demo_images[0]
        
        print(f"Analyzing: {demo_image['description']}")
        print(f"Image: {demo_image['path']}")
        print(f"Goals: {demo_image['analysis_goals']}")
        
        try:
            # Perform orchestrated analysis
            results = await self.orchestrator.orchestrated_analysis(
                image_path=demo_image["path"],
                analysis_goals=demo_image["analysis_goals"],
                strategy=AnalysisStrategy.BALANCED
            )
            
            # Display results
            self._display_analysis_results("Basic Orchestration", results)
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            
    async def _demo_strategy_comparison(self):
        """Compare different analysis strategies."""
        print("\n⚖️ DEMO 2: Strategy Comparison")
        print("-" * 40)
        
        demo_image = self.demo_images[1]  # Complex scene
        strategies = [
            AnalysisStrategy.SPEED_OPTIMIZED,
            AnalysisStrategy.BALANCED,
            AnalysisStrategy.QUALITY_OPTIMIZED,
            AnalysisStrategy.DEEP_ANALYSIS
        ]
        
        strategy_results = {}
        
        for strategy in strategies:
            print(f"\n🎯 Testing {strategy.value} strategy...")
            
            try:
                results = await self.orchestrator.orchestrated_analysis(
                    image_path=demo_image["path"],
                    analysis_goals=demo_image["analysis_goals"],
                    strategy=strategy,
                    time_budget=30.0  # Limited time for comparison
                )
                
                strategy_results[strategy.value] = {
                    "success": results.get("success", False),
                    "quality": results.get("overall_quality", 0.0),
                    "confidence": results.get("overall_confidence", 0.0),
                    "execution_time": results.get("execution_time", 0.0),
                    "modules_executed": results.get("modules_executed", 0)
                }
                
                print(f"   ✅ Quality: {results.get('overall_quality', 0):.2%}")
                print(f"   ✅ Confidence: {results.get('overall_confidence', 0):.2%}")
                print(f"   ⏱️ Time: {results.get('execution_time', 0):.2f}s")
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                strategy_results[strategy.value] = {"error": str(e)}
        
        # Compare strategies
        print(f"\n📊 Strategy Comparison Summary:")
        self._display_strategy_comparison(strategy_results)
        
    async def _demo_adaptive_analysis(self):
        """Demonstrate adaptive analysis capabilities."""
        print("\n🔄 DEMO 3: Adaptive Analysis")
        print("-" * 40)
        
        print("The orchestrator will adaptively select the best strategy based on:")
        print("- Image complexity assessment")
        print("- Noise level detection")
        print("- Available time budget")
        print("- Quality requirements")
        
        for i, demo_image in enumerate(self.demo_images):
            print(f"\n🖼️ Image {i+1}: {demo_image['description']}")
            
            try:
                results = await self.orchestrator.orchestrated_analysis(
                    image_path=demo_image["path"],
                    analysis_goals=demo_image["analysis_goals"],
                    strategy=AnalysisStrategy.ADAPTIVE,  # Let orchestrator decide
                    time_budget=45.0
                )
                
                print(f"   🎯 Selected strategy: {results.get('strategy_used', 'unknown')}")
                print(f"   📊 Detected complexity: {results.get('image_complexity', 'unknown')}")
                print(f"   🔍 Noise level: {results.get('noise_level', 0):.1%}")
                print(f"   ✅ Final quality: {results.get('overall_quality', 0):.2%}")
                print(f"   📈 Strategy effectiveness: {results.get('final_assessment', {}).get('strategy_effectiveness', 'unknown')}")
                
            except Exception as e:
                print(f"   ❌ Analysis failed: {e}")
    
    async def _demo_learning_capabilities(self):
        """Demonstrate learning and adaptation capabilities."""
        print("\n🧠 DEMO 4: Learning and Adaptation")
        print("-" * 40)
        
        print("Running multiple analyses to demonstrate learning...")
        
        # Run multiple analyses to build learning data
        for iteration in range(1, 4):
            print(f"\n📚 Learning iteration {iteration}")
            
            for demo_image in self.demo_images[:2]:  # Use first 2 images
                try:
                    results = await self.orchestrator.orchestrated_analysis(
                        image_path=demo_image["path"],
                        analysis_goals=demo_image["analysis_goals"],
                        strategy=AnalysisStrategy.ADAPTIVE
                    )
                    
                    print(f"   ✅ Completed: {demo_image['description']}")
                    
                except Exception as e:
                    print(f"   ❌ Failed: {demo_image['description']} - {e}")
        
        # Show learning summary
        learning_summary = self.orchestrator.get_learning_summary()
        print(f"\n📈 Learning Summary:")
        print(f"   Executions completed: {learning_summary['executions_completed']}")
        print(f"   Strategy performance:")
        for strategy, performance in learning_summary['strategy_performance'].items():
            print(f"     {strategy}: {performance:.2%}")
        print(f"   Module reliability:")
        for module, reliability in learning_summary['module_reliability'].items():
            print(f"     {module}: {reliability:.2%}")
        
    async def _demo_metacognitive_insights(self):
        """Demonstrate metacognitive insights generation."""
        print("\n🎯 DEMO 5: Metacognitive Insights")
        print("-" * 40)
        
        # Perform analysis that will generate insights
        demo_image = self.demo_images[2]  # Noisy image
        
        try:
            results = await self.orchestrator.orchestrated_analysis(
                image_path=demo_image["path"],
                analysis_goals=demo_image["analysis_goals"],
                strategy=AnalysisStrategy.QUALITY_OPTIMIZED
            )
            
            # Display metacognitive insights
            insights = results.get("metacognitive_insights", [])
            
            if insights:
                print(f"Generated {len(insights)} metacognitive insights:")
                
                for i, insight in enumerate(insights, 1):
                    print(f"\n   🧠 Insight {i}: {insight.get('insight_type', 'unknown')}")
                    print(f"      Description: {insight.get('description', 'N/A')}")
                    print(f"      Confidence: {insight.get('confidence', 0):.2%}")
                    
                    evidence = insight.get('supporting_evidence', [])
                    if evidence:
                        print(f"      Evidence: {', '.join(evidence)}")
                    
                    recommendations = insight.get('recommendations', [])
                    if recommendations:
                        print(f"      Recommendations: {', '.join(recommendations)}")
            else:
                print("No metacognitive insights generated (need more learning data)")
                
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
    
    async def _demo_performance_optimization(self):
        """Demonstrate performance optimization capabilities."""
        print("\n⚡ DEMO 6: Performance Optimization")
        print("-" * 40)
        
        demo_image = self.demo_images[3]  # Highly complex image
        
        print("Comparing performance with different optimization approaches:")
        
        # Test 1: Speed-optimized
        print(f"\n🚀 Speed-optimized analysis:")
        try:
            results_speed = await self.orchestrator.orchestrated_analysis(
                image_path=demo_image["path"],
                analysis_goals=["basic_understanding"],
                strategy=AnalysisStrategy.SPEED_OPTIMIZED,
                time_budget=15.0
            )
            
            print(f"   ⏱️ Time: {results_speed.get('execution_time', 0):.2f}s")
            print(f"   ✅ Quality: {results_speed.get('overall_quality', 0):.2%}")
            print(f"   📊 Modules used: {results_speed.get('modules_executed', 0)}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results_speed = {"execution_time": float('inf'), "overall_quality": 0}
        
        # Test 2: Quality-optimized
        print(f"\n🎯 Quality-optimized analysis:")
        try:
            results_quality = await self.orchestrator.orchestrated_analysis(
                image_path=demo_image["path"],
                analysis_goals=demo_image["analysis_goals"],
                strategy=AnalysisStrategy.QUALITY_OPTIMIZED,
                time_budget=60.0
            )
            
            print(f"   ⏱️ Time: {results_quality.get('execution_time', 0):.2f}s")
            print(f"   ✅ Quality: {results_quality.get('overall_quality', 0):.2%}")
            print(f"   📊 Modules used: {results_quality.get('modules_executed', 0)}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results_quality = {"execution_time": 0, "overall_quality": 0}
        
        # Compare efficiency
        speed_time = results_speed.get('execution_time', float('inf'))
        quality_time = results_quality.get('execution_time', 0)
        speed_quality = results_speed.get('overall_quality', 0)
        quality_quality = results_quality.get('overall_quality', 0)
        
        if speed_time != float('inf') and quality_time > 0:
            speedup = quality_time / speed_time
            quality_gain = quality_quality - speed_quality
            
            print(f"\n📈 Performance Comparison:")
            print(f"   Speedup: {speedup:.1f}x faster")
            print(f"   Quality trade-off: {quality_gain:.2%} quality difference")
            print(f"   Efficiency ratio: {speed_quality/speed_time:.3f} quality/second (speed)")
            print(f"   Efficiency ratio: {quality_quality/quality_time:.3f} quality/second (quality)")
    
    def _display_analysis_results(self, title: str, results: Dict[str, Any]):
        """Display analysis results in a formatted way."""
        print(f"\n📊 {title} Results:")
        print(f"   Success: {'✅' if results.get('success', False) else '❌'}")
        print(f"   Overall Quality: {results.get('overall_quality', 0):.2%}")
        print(f"   Overall Confidence: {results.get('overall_confidence', 0):.2%}")
        print(f"   Execution Time: {results.get('execution_time', 0):.2f}s")
        print(f"   Modules Executed: {results.get('modules_executed', 0)}")
        print(f"   Successful Modules: {results.get('successful_modules', 0)}")
        
        # Show final assessment
        final_assessment = results.get('final_assessment', {})
        if final_assessment:
            print(f"   Quality Level: {final_assessment.get('quality_level', 'Unknown')}")
            print(f"   Confidence Level: {final_assessment.get('confidence_level', 'Unknown')}")
            print(f"   Summary: {final_assessment.get('summary', 'N/A')}")
        
        # Show key insights
        insights = results.get('all_insights', [])
        if insights:
            print(f"   Key Insights: {len(insights)} insights generated")
            for insight in insights[:3]:  # Show first 3
                print(f"     • {insight}")
    
    def _display_strategy_comparison(self, strategy_results: Dict[str, Any]):
        """Display strategy comparison results."""
        
        # Sort by quality
        sorted_strategies = sorted(
            [(name, data) for name, data in strategy_results.items() if not data.get('error')],
            key=lambda x: x[1].get('quality', 0),
            reverse=True
        )
        
        if sorted_strategies:
            print(f"   🥇 Best Quality: {sorted_strategies[0][0]} ({sorted_strategies[0][1]['quality']:.2%})")
            
            # Find fastest
            fastest = min(sorted_strategies, key=lambda x: x[1].get('execution_time', float('inf')))
            print(f"   ⚡ Fastest: {fastest[0]} ({fastest[1]['execution_time']:.2f}s)")
            
            # Find most comprehensive
            most_modules = max(sorted_strategies, key=lambda x: x[1].get('modules_executed', 0))
            print(f"   📊 Most Comprehensive: {most_modules[0]} ({most_modules[1]['modules_executed']} modules)")
        
        # Show detailed comparison table
        print(f"\n   Detailed Comparison:")
        print(f"   {'Strategy':<20} {'Quality':<10} {'Confidence':<12} {'Time':<8} {'Modules':<8}")
        print(f"   {'-'*60}")
        
        for strategy_name, data in strategy_results.items():
            if not data.get('error'):
                print(f"   {strategy_name:<20} {data.get('quality', 0):<10.2%} "
                      f"{data.get('confidence', 0):<12.2%} {data.get('execution_time', 0):<8.2f} "
                      f"{data.get('modules_executed', 0):<8}")
    
    def save_demo_results(self, filename: str = "metacognitive_demo_results.json"):
        """Save demonstration results for analysis."""
        
        learning_summary = self.orchestrator.get_learning_summary()
        
        demo_results = {
            "timestamp": asyncio.get_event_loop().time(),
            "demo_images": self.demo_images,
            "learning_summary": learning_summary,
            "orchestrator_config": self.orchestrator.config
        }
        
        with open(filename, 'w') as f:
            json.dump(demo_results, f, indent=2, default=str)
        
        print(f"💾 Demo results saved to {filename}")

# Interactive demonstration
async def interactive_demo():
    """Run interactive demonstration."""
    
    print("🎮 INTERACTIVE METACOGNITIVE ORCHESTRATOR DEMO")
    print("Choose a demonstration mode:")
    print("1. Full comprehensive demo")
    print("2. Quick strategy comparison")
    print("3. Single image analysis")
    print("4. Learning capabilities test")
    
    try:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        demo = MetacognitiveDemo()
        
        if choice == "1":
            await demo.run_comprehensive_demo()
        elif choice == "2":
            await demo._demo_strategy_comparison()
        elif choice == "3":
            image_path = input("Enter image path: ").strip()
            goals = input("Enter analysis goals (comma-separated): ").strip().split(',')
            goals = [g.strip() for g in goals if g.strip()]
            
            results = await demo.orchestrator.orchestrated_analysis(
                image_path=image_path,
                analysis_goals=goals,
                strategy=AnalysisStrategy.ADAPTIVE
            )
            
            demo._display_analysis_results("Single Image Analysis", results)
        elif choice == "4":
            await demo._demo_learning_capabilities()
        else:
            print("Invalid choice. Running full demo...")
            await demo.run_comprehensive_demo()
        
        # Save results
        demo.save_demo_results()
        
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"❌ Demo failed: {e}")

# Batch demonstration
async def batch_demo():
    """Run batch demonstration with multiple configurations."""
    
    print("🏭 BATCH METACOGNITIVE ORCHESTRATOR DEMO")
    
    demo = MetacognitiveDemo()
    
    # Test configurations
    configurations = [
        {"strategy": AnalysisStrategy.SPEED_OPTIMIZED, "time_budget": 15.0},
        {"strategy": AnalysisStrategy.BALANCED, "time_budget": 30.0},
        {"strategy": AnalysisStrategy.QUALITY_OPTIMIZED, "time_budget": 60.0},
        {"strategy": AnalysisStrategy.ADAPTIVE, "time_budget": 45.0},
    ]
    
    batch_results = {}
    
    for i, config in enumerate(configurations, 1):
        print(f"\n🔄 Batch {i}/{len(configurations)}: {config['strategy'].value}")
        
        config_results = []
        
        for demo_image in demo.demo_images:
            try:
                results = await demo.orchestrator.orchestrated_analysis(
                    image_path=demo_image["path"],
                    analysis_goals=demo_image["analysis_goals"],
                    **config
                )
                
                config_results.append({
                    "image": demo_image["description"],
                    "success": results.get("success", False),
                    "quality": results.get("overall_quality", 0.0),
                    "confidence": results.get("overall_confidence", 0.0),
                    "execution_time": results.get("execution_time", 0.0)
                })
                
            except Exception as e:
                config_results.append({
                    "image": demo_image["description"],
                    "error": str(e)
                })
        
        batch_results[config['strategy'].value] = config_results
    
    # Display batch results
    print(f"\n📊 BATCH RESULTS SUMMARY")
    print("=" * 60)
    
    for strategy, results in batch_results.items():
        successful_results = [r for r in results if not r.get('error')]
        
        if successful_results:
            avg_quality = sum(r['quality'] for r in successful_results) / len(successful_results)
            avg_confidence = sum(r['confidence'] for r in successful_results) / len(successful_results)
            avg_time = sum(r['execution_time'] for r in successful_results) / len(successful_results)
            
            print(f"\n🎯 {strategy}:")
            print(f"   Success rate: {len(successful_results)}/{len(results)} ({len(successful_results)/len(results):.1%})")
            print(f"   Average quality: {avg_quality:.2%}")
            print(f"   Average confidence: {avg_confidence:.2%}")
            print(f"   Average time: {avg_time:.2f}s")
    
    # Save batch results
    with open("batch_demo_results.json", 'w') as f:
        json.dump(batch_results, f, indent=2, default=str)
    
    print(f"\n💾 Batch results saved to batch_demo_results.json")

if __name__ == "__main__":
    print("🧠 Metacognitive Orchestrator Demo")
    print("Choose demo type:")
    print("1. Interactive demo")
    print("2. Batch demo")
    print("3. Full comprehensive demo")
    
    try:
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            asyncio.run(interactive_demo())
        elif choice == "2":
            asyncio.run(batch_demo())
        else:
            demo = MetacognitiveDemo()
            asyncio.run(demo.run_comprehensive_demo())
            
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted")
    except Exception as e:
        print(f"❌ Demo failed: {e}") 