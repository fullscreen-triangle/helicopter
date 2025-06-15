"""
Metacognitive Orchestrator Demo

Demonstrates the comprehensive metacognitive orchestrator that intelligently
coordinates all Helicopter modules for optimal image analysis.
"""

import asyncio
import logging
from typing import List, Dict, Any

from helicopter.core import (
    MetacognitiveOrchestrator,
    AnalysisStrategy,
    ImageComplexity
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OrchestratorDemo:
    """Demonstration of the Metacognitive Orchestrator."""
    
    def __init__(self):
        """Initialize the demonstration."""
        self.orchestrator = MetacognitiveOrchestrator()
        
    async def run_demo(self):
        """Run comprehensive demonstration."""
        
        print("🧠 METACOGNITIVE ORCHESTRATOR DEMONSTRATION")
        print("=" * 60)
        
        # Demo image paths (these would be actual image files in practice)
        demo_images = [
            "examples/images/simple_object.jpg",
            "examples/images/complex_scene.jpg", 
            "examples/images/noisy_image.jpg"
        ]
        
        for i, image_path in enumerate(demo_images, 1):
            print(f"\n📋 Demo {i}: Analyzing {image_path}")
            print("-" * 40)
            
            try:
                # Perform orchestrated analysis
                results = await self.orchestrator.orchestrated_analysis(
                    image_path=image_path,
                    analysis_goals=["comprehensive_understanding", "quality_assessment"],
                    strategy=AnalysisStrategy.ADAPTIVE
                )
                
                # Display results
                self._display_results(results)
                
            except Exception as e:
                print(f"❌ Analysis failed: {e}")
        
        # Show learning summary
        print(f"\n📈 Learning Summary:")
        learning_summary = self.orchestrator.get_learning_summary()
        print(f"   Executions completed: {learning_summary['executions_completed']}")
        
    def _display_results(self, results: Dict[str, Any]):
        """Display analysis results."""
        print(f"   Success: {'✅' if results.get('success', False) else '❌'}")
        print(f"   Overall Quality: {results.get('overall_quality', 0):.2%}")
        print(f"   Overall Confidence: {results.get('overall_confidence', 0):.2%}")
        print(f"   Execution Time: {results.get('execution_time', 0):.2f}s")
        print(f"   Strategy Used: {results.get('strategy_used', 'unknown')}")
        print(f"   Modules Executed: {results.get('modules_executed', 0)}")

async def main():
    """Main demo function."""
    demo = OrchestratorDemo()
    await demo.run_demo()

if __name__ == "__main__":
    asyncio.run(main()) 