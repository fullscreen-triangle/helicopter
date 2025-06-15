#!/usr/bin/env python3
"""
Diadochi Framework Demonstration

This demo showcases the complete Diadochi framework for intelligent model combination,
demonstrating all five architectural patterns with practical examples.

Named after Alexander the Great's successors who divided his empire into specialized domains,
Diadochi intelligently combines domain-expert models to produce superior expertise.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add the helicopter module to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from helicopter.core.diadochi import (
    DiadochiCore, 
    DomainExpertise, 
    IntegrationPattern
)
from helicopter.core.diadochi_models import (
    ModelFactory, 
    MockModel
)
from helicopter.core.diadochi_evaluation import (
    DiadochiEvaluator,
    EvaluationQuery,
    create_domain_specific_queries,
    create_cross_domain_queries
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DiadochiDemo:
    """Main demo class showcasing Diadochi capabilities."""
    
    def __init__(self):
        self.diadochi = DiadochiCore()
        self.setup_complete = False
    
    async def setup_demo_environment(self):
        """Setup the demo environment with mock models and domain expertise."""
        logger.info("🚀 Setting up Diadochi Demo Environment")
        
        # Define domain expertise areas
        domain_expertise = [
            DomainExpertise(
                domain="computer_vision",
                description="Computer vision focuses on enabling machines to interpret and understand visual information from images and videos, including object detection, image classification, and visual recognition.",
                keywords=["image", "vision", "visual", "pixel", "detection", "classification", "opencv", "cnn"],
                specialized_prompts={
                    "system": "You are a computer vision expert with deep knowledge of image processing, neural networks for vision tasks, and visual recognition systems."
                }
            ),
            DomainExpertise(
                domain="machine_learning",
                description="Machine learning involves developing algorithms and statistical models that enable computers to learn and make decisions from data without explicit programming.",
                keywords=["algorithm", "model", "training", "neural", "learning", "prediction", "tensorflow", "sklearn"],
                specialized_prompts={
                    "system": "You are a machine learning expert specializing in algorithm design, model training, and predictive analytics."
                }
            ),
            DomainExpertise(
                domain="natural_language",
                description="Natural language processing enables computers to understand, interpret, and generate human language in a meaningful way.",
                keywords=["text", "language", "nlp", "words", "sentence", "grammar", "tokenization", "transformer"],
                specialized_prompts={
                    "system": "You are a natural language processing expert with expertise in text analysis, language models, and linguistic AI systems."
                }
            ),
            DomainExpertise(
                domain="data_science",
                description="Data science combines statistics, programming, and domain expertise to extract insights and knowledge from structured and unstructured data.",
                keywords=["data", "analysis", "statistics", "correlation", "pandas", "numpy", "visualization", "insight"],
                specialized_prompts={
                    "system": "You are a data science expert skilled in statistical analysis, data visualization, and extracting actionable insights from complex datasets."
                }
            )
        ]
        
        # Add domain expertise to Diadochi
        for expertise in domain_expertise:
            self.diadochi.add_domain_expertise(expertise)
        
        # Create mock models with specialized responses
        cv_responses = {
            "image": "In computer vision, images are processed as multidimensional arrays where each pixel contains color information. Common preprocessing steps include normalization, augmentation, and feature extraction using convolutional neural networks.",
            "detection": "Object detection combines classification and localization to identify and locate objects within images. Popular architectures include YOLO, R-CNN, and SSD, each offering different trade-offs between speed and accuracy.",
            "vision": "Computer vision systems analyze visual data through hierarchical feature learning, starting from edge detection at lower layers to complex object recognition at higher layers."
        }
        
        ml_responses = {
            "algorithm": "Machine learning algorithms can be categorized into supervised, unsupervised, and reinforcement learning. Each category addresses different types of problems and requires different training approaches.",
            "model": "ML models are mathematical representations that learn patterns from data. The choice of model depends on the problem type, data size, and interpretability requirements.",
            "training": "Model training involves optimizing parameters to minimize a loss function. This process requires careful consideration of learning rates, regularization, and validation strategies."
        }
        
        nlp_responses = {
            "language": "Natural language processing leverages computational linguistics and machine learning to enable computers to process human language. Modern approaches use transformer architectures for superior context understanding.",
            "text": "Text processing involves tokenization, normalization, and feature extraction. Advanced techniques include attention mechanisms and contextualized embeddings for better semantic understanding.",
            "nlp": "NLP applications range from sentiment analysis to machine translation, each requiring specialized preprocessing and model architectures optimized for linguistic patterns."
        }
        
        ds_responses = {
            "data": "Data science workflows typically follow the CRISP-DM methodology: data understanding, preparation, modeling, evaluation, and deployment. Each phase requires different analytical approaches.",
            "analysis": "Statistical analysis in data science involves hypothesis testing, correlation analysis, and predictive modeling to uncover patterns and relationships in data.",
            "insight": "Generating actionable insights requires combining domain expertise with statistical analysis, often involving data visualization and storytelling to communicate findings effectively."
        }
        
        # Register models with domain-specific knowledge
        cv_model = MockModel("computer_vision_expert", cv_responses)
        ml_model = MockModel("machine_learning_expert", ml_responses) 
        nlp_model = MockModel("nlp_expert", nlp_responses)
        ds_model = MockModel("data_science_expert", ds_responses)
        
        # General model for synthesis and fallback
        general_responses = {
            "combine": "Integrating insights from multiple domains requires understanding the connections between different fields and synthesizing information coherently.",
            "analysis": "Multi-domain analysis benefits from considering various perspectives and methodological approaches to provide comprehensive solutions."
        }
        general_model = MockModel("general_expert", general_responses)
        
        # Register all models
        self.diadochi.register_model("cv_expert", cv_model, ["computer_vision"])
        self.diadochi.register_model("ml_expert", ml_model, ["machine_learning"])
        self.diadochi.register_model("nlp_expert", nlp_model, ["natural_language"])
        self.diadochi.register_model("ds_expert", ds_model, ["data_science"])
        self.diadochi.register_model("general", general_model)
        
        self.setup_complete = True
        logger.info("✅ Demo environment setup complete!")
    
    async def demonstrate_router_ensemble(self):
        """Demonstrate Router-Based Ensemble pattern."""
        logger.info("\n🎯 Demonstrating Router-Based Ensemble Pattern")
        
        # Configure router ensemble
        self.diadochi.configure_router_ensemble(
            router_type="embedding",
            embedding_model="general",
            threshold=0.5,
            mixer_type="synthesis"
        )
        
        # Test queries
        test_queries = [
            "How can I improve object detection accuracy in low-light images?",
            "What machine learning algorithm should I use for time series prediction?",
            "How do I extract sentiment from customer reviews?",
            "What's the best way to visualize correlation between multiple variables?"
        ]
        
        for query in test_queries:
            print(f"\n📝 Query: {query}")
            response = await self.diadochi.generate(query)
            print(f"🤖 Response: {response}")
            
            # Demonstrate multi-domain routing
            multi_response = await self.diadochi.generate(query, top_k=2)
            print(f"🔄 Multi-domain response: {multi_response}")
    
    async def demonstrate_sequential_chain(self):
        """Demonstrate Sequential Chaining pattern."""
        logger.info("\n🔗 Demonstrating Sequential Chaining Pattern")
        
        # Define chain sequence and templates
        chain_sequence = ["cv_expert", "ml_expert", "ds_expert", "general"]
        
        prompt_templates = {
            "cv_expert": "From a computer vision perspective, analyze this query: {query}",
            "ml_expert": """Computer vision analysis: {responses[0]}
            
Original query: {query}

From a machine learning perspective, build upon the computer vision analysis:""",
            "ds_expert": """Previous analyses:
Computer Vision: {responses[0]}
Machine Learning: {responses[1]}

Original query: {query}

From a data science perspective, integrate these insights:""",
            "general": """Domain expert analyses:
1. Computer Vision: {responses[0]}
2. Machine Learning: {responses[1]} 
3. Data Science: {responses[2]}

Original query: {query}

Synthesize these perspectives into a comprehensive response:"""
        }
        
        # Configure sequential chain
        self.diadochi.configure_sequential_chain(chain_sequence, prompt_templates)
        
        # Test with complex query
        complex_query = "How can I build an AI system that automatically analyzes customer feedback from images and text to predict satisfaction trends?"
        
        print(f"\n📝 Complex Query: {complex_query}")
        response = await self.diadochi.generate(complex_query)
        print(f"🔗 Chained Response: {response}")
    
    async def demonstrate_mixture_of_experts(self):
        """Demonstrate Mixture of Experts pattern."""
        logger.info("\n🎪 Demonstrating Mixture of Experts Pattern")
        
        # Configure mixture of experts
        self.diadochi.configure_mixture_of_experts(
            mixer_model="general",
            threshold=0.2,
            temperature=0.7
        )
        
        # Test cross-domain queries
        cross_domain_queries = [
            "How do computer vision and NLP techniques combine in document analysis systems?",
            "What role does data science play in improving machine learning model performance?",
            "How can I use both image analysis and text processing for social media sentiment analysis?"
        ]
        
        for query in cross_domain_queries:
            print(f"\n📝 Cross-domain Query: {query}")
            response = await self.diadochi.generate(query)
            print(f"🎪 MoE Response: {response}")
    
    async def demonstrate_system_prompts(self):
        """Demonstrate Specialized System Prompts pattern."""
        logger.info("\n💬 Demonstrating Specialized System Prompts Pattern")
        
        # Configure system prompts
        self.diadochi.configure_system_prompts(
            base_model="general",
            integration_prompt="""You are a multi-domain AI expert with specialized knowledge in:
- Computer Vision: Image processing and visual recognition
- Machine Learning: Algorithm development and model training  
- Natural Language Processing: Text analysis and language understanding
- Data Science: Statistical analysis and insight generation

For the following query, determine relevant domains and provide an integrated response that combines insights from all applicable areas:

Query: {query}

Provide a comprehensive response that demonstrates expertise across relevant domains:"""
        )
        
        # Test various query types
        test_queries = [
            "Explain convolutional neural networks", # Single domain
            "How do attention mechanisms work in both vision and language models?", # Multi-domain
            "What's the complete pipeline for building an AI-powered content moderation system?" # Full integration
        ]
        
        for query in test_queries:
            print(f"\n📝 Query: {query}")
            response = await self.diadochi.generate(query)
            print(f"💬 System Prompt Response: {response}")
    
    async def demonstrate_evaluation_framework(self):
        """Demonstrate the evaluation capabilities."""
        logger.info("\n📊 Demonstrating Evaluation Framework")
        
        # Create evaluation queries
        evaluation_queries = []
        
        # Add domain-specific queries
        for domain in ["computer_vision", "machine_learning", "natural_language", "data_science"]:
            domain_queries = create_domain_specific_queries(domain, 3)
            evaluation_queries.extend(domain_queries)
        
        # Add cross-domain queries  
        domains = ["computer_vision", "machine_learning", "natural_language", "data_science"]
        cross_queries = create_cross_domain_queries(domains, 5)
        evaluation_queries.extend(cross_queries)
        
        # Setup evaluator with mock domain experts
        mock_experts = {
            "computer_vision": MockModel("cv_eval", {"image": "Expert CV analysis"}),
            "machine_learning": MockModel("ml_eval", {"model": "Expert ML analysis"}),
            "natural_language": MockModel("nlp_eval", {"text": "Expert NLP analysis"}),
            "data_science": MockModel("ds_eval", {"data": "Expert DS analysis"})
        }
        
        evaluator = DiadochiEvaluator(mock_experts)
        
        # Evaluate different patterns
        patterns_to_test = [
            IntegrationPattern.ROUTER_ENSEMBLE,
            IntegrationPattern.MIXTURE_OF_EXPERTS,
            IntegrationPattern.SYSTEM_PROMPTS
        ]
        
        pattern_metrics = {}
        
        for pattern in patterns_to_test:
            logger.info(f"Evaluating {pattern.value}")
            
            # Configure pattern
            if pattern == IntegrationPattern.ROUTER_ENSEMBLE:
                self.diadochi.configure_router_ensemble()
            elif pattern == IntegrationPattern.MIXTURE_OF_EXPERTS:
                self.diadochi.configure_mixture_of_experts()
            elif pattern == IntegrationPattern.SYSTEM_PROMPTS:
                self.diadochi.configure_system_prompts("general")
            
            # Evaluate subset of queries (for demo speed)
            test_queries = evaluation_queries[:5]
            metrics = await evaluator.evaluate_dataset(self.diadochi, test_queries, parallel=False)
            pattern_metrics[pattern] = metrics
            
            print(f"\n📈 {pattern.value} Results:")
            print(f"   Response Quality: {metrics.response_quality:.3f}")
            print(f"   Cross-Domain Accuracy: {metrics.cross_domain_accuracy:.3f}")
            print(f"   Mean Latency: {metrics.latency_metrics.get('mean_latency', 0):.2f}s")
        
        # Generate comparison report
        comparison_report = evaluator.compare_patterns(pattern_metrics)
        print(f"\n📋 Pattern Comparison:\n{comparison_report}")
    
    async def demonstrate_real_world_scenarios(self):
        """Demonstrate real-world application scenarios."""
        logger.info("\n🌍 Demonstrating Real-World Scenarios")
        
        scenarios = [
            {
                "title": "AI-Powered Content Moderation System",
                "query": "Design a comprehensive content moderation system that can analyze both images and text for inappropriate content, learn from user feedback, and provide detailed reporting dashboard.",
                "pattern": IntegrationPattern.SEQUENTIAL_CHAIN
            },
            {
                "title": "Medical Image Analysis Pipeline", 
                "query": "Create an end-to-end pipeline for analyzing medical images that includes preprocessing, feature extraction, diagnosis prediction, and confidence scoring with explainable results.",
                "pattern": IntegrationPattern.MIXTURE_OF_EXPERTS
            },
            {
                "title": "Social Media Analytics Platform",
                "query": "Build a platform that analyzes social media posts (text, images, videos) to track brand sentiment, identify trending topics, and predict viral content using multi-modal AI approaches.",
                "pattern": IntegrationPattern.ROUTER_ENSEMBLE
            },
            {
                "title": "Autonomous Driving Perception System",
                "query": "Design a perception system for autonomous vehicles that combines computer vision for object detection, machine learning for behavior prediction, and data analysis for performance optimization.",
                "pattern": IntegrationPattern.SYSTEM_PROMPTS
            }
        ]
        
        for scenario in scenarios:
            print(f"\n🎯 Scenario: {scenario['title']}")
            print(f"📋 Challenge: {scenario['query']}")
            
            # Configure appropriate pattern
            if scenario['pattern'] == IntegrationPattern.ROUTER_ENSEMBLE:
                self.diadochi.configure_router_ensemble()
            elif scenario['pattern'] == IntegrationPattern.SEQUENTIAL_CHAIN:
                chain_sequence = ["cv_expert", "ml_expert", "ds_expert", "general"]
                self.diadochi.configure_sequential_chain(chain_sequence)
            elif scenario['pattern'] == IntegrationPattern.MIXTURE_OF_EXPERTS:
                self.diadochi.configure_mixture_of_experts()
            elif scenario['pattern'] == IntegrationPattern.SYSTEM_PROMPTS:
                self.diadochi.configure_system_prompts("general")
            
            # Generate solution
            solution = await self.diadochi.generate(scenario['query'])
            print(f"🤖 Solution ({scenario['pattern'].value}):\n{solution}\n")
    
    async def run_full_demo(self):
        """Run the complete Diadochi demonstration."""
        print("=" * 80)
        print("🏛️  DIADOCHI: INTELLIGENT MODEL COMBINATION FRAMEWORK")
        print("    Named after Alexander's successors who mastered specialized domains")
        print("=" * 80)
        
        # Setup
        if not self.setup_complete:
            await self.setup_demo_environment()
        
        # Display system status
        status = self.diadochi.get_status()
        print(f"\n📊 System Status:")
        print(f"   Registered Models: {len(status['registered_models'])}")
        print(f"   Available Domains: {len(status['available_domains'])}")
        print(f"   Active Pattern: {status['active_pattern'] or 'None'}")
        
        try:
            # Demonstrate all patterns
            await self.demonstrate_router_ensemble()
            await self.demonstrate_sequential_chain()
            await self.demonstrate_mixture_of_experts()
            await self.demonstrate_system_prompts()
            
            # Evaluation framework
            await self.demonstrate_evaluation_framework()
            
            # Real-world scenarios
            await self.demonstrate_real_world_scenarios()
            
            print("\n" + "=" * 80)
            print("✅ DIADOCHI DEMONSTRATION COMPLETED SUCCESSFULLY!")
            print("🚀 Ready for production deployment with real models")
            print("=" * 80)
            
        except Exception as e:
            logger.error(f"Demo error: {e}")
            print(f"\n❌ Demo encountered an error: {e}")

async def run_interactive_demo():
    """Run an interactive demo allowing user queries."""
    demo = DiadochiDemo()
    await demo.setup_demo_environment()
    
    print("\n🎮 Interactive Diadochi Demo")
    print("Enter queries to test different patterns, or 'quit' to exit")
    print("Available patterns: router, chain, mixture, prompts")
    
    current_pattern = "router"
    demo.diadochi.configure_router_ensemble()
    
    while True:
        print(f"\n[Current: {current_pattern}] ", end="")
        user_input = input("Query: ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() in ['router', 'chain', 'mixture', 'prompts']:
            current_pattern = user_input.lower()
            if current_pattern == 'router':
                demo.diadochi.configure_router_ensemble()
            elif current_pattern == 'chain':
                demo.diadochi.configure_sequential_chain(["cv_expert", "ml_expert", "general"])
            elif current_pattern == 'mixture':
                demo.diadochi.configure_mixture_of_experts()
            elif current_pattern == 'prompts':
                demo.diadochi.configure_system_prompts("general")
            print(f"✅ Switched to {current_pattern} pattern")
            continue
        
        if user_input:
            try:
                response = await demo.diadochi.generate(user_input)
                print(f"🤖 {response}")
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main demo entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Diadochi Framework Demonstration")
    parser.add_argument("--interactive", "-i", action="store_true", 
                       help="Run interactive demo mode")
    args = parser.parse_args()
    
    if args.interactive:
        asyncio.run(run_interactive_demo())
    else:
        demo = DiadochiDemo()
        asyncio.run(demo.run_full_demo())

if __name__ == "__main__":
    main() 