"""
Metacognitive Orchestrated Pipeline

The ultimate coordination layer that intelligently orchestrates all Helicopter modules:
- Autonomous Reconstruction Engine
- Segment-Aware Reconstruction  
- Nicotine Context Validator
- Hatata MDP Engine (Probabilistic Understanding)
- Zengeza Noise Detector
- Diadochi Model Combination
- Traditional CV methods (Vibrio, Moriarty, Homo-veloce, Pakati)

This orchestrator uses metacognitive principles to decide which modules to use when,
adapt strategies based on image type and analysis goals, and learn from outcomes.
"""

import asyncio
import json
import logging
import numpy as np
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from pathlib import Path
from enum import Enum
import statistics

# Import all Helicopter modules
from .autonomous_reconstruction_engine import AutonomousReconstructionEngine
from .segment_aware_reconstruction import SegmentAwareReconstructionEngine, SegmentType
from .nicotine_context_validator import NicotineContextValidator, NicotinePuzzle, PuzzleType
from .hatata_mdp_engine import HatataEngine, HatataMDPModel, UnderstandingState, HatataAction
from .zengeza_noise_detector import ZengezaEngine, ZengezaNoiseAnalyzer, NoiseType, NoiseLevel
from .diadochi import DiadochiCore, DomainExpertise, IntegrationPattern
from .diadochi_models import ModelFactory, MockModel
from .comprehensive_analysis_engine import ComprehensiveAnalysisEngine

logger = logging.getLogger(__name__)

class AnalysisStrategy(Enum):
    """Available analysis strategies."""
    SPEED_OPTIMIZED = "speed_optimized"
    QUALITY_OPTIMIZED = "quality_optimized"
    BALANCED = "balanced"
    DEEP_ANALYSIS = "deep_analysis"
    ADAPTIVE = "adaptive"

class ImageComplexity(Enum):
    """Image complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    HIGHLY_COMPLEX = "highly_complex"

class AnalysisPhase(Enum):
    """Analysis pipeline phases."""
    INITIAL_ASSESSMENT = "initial_assessment"
    NOISE_DETECTION = "noise_detection"
    STRATEGY_SELECTION = "strategy_selection"
    RECONSTRUCTION_ANALYSIS = "reconstruction_analysis"
    PROBABILISTIC_VALIDATION = "probabilistic_validation"
    CONTEXT_VALIDATION = "context_validation"
    EXPERT_SYNTHESIS = "expert_synthesis"
    FINAL_INTEGRATION = "final_integration"
    METACOGNITIVE_REVIEW = "metacognitive_review"

@dataclass
class AnalysisContext:
    """Comprehensive context for the analysis pipeline."""
    image_path: str
    image_type: str = "unknown"
    analysis_goals: List[str] = field(default_factory=list)
    strategy: AnalysisStrategy = AnalysisStrategy.ADAPTIVE
    complexity: ImageComplexity = ImageComplexity.MODERATE
    time_budget: float = 60.0  # seconds
    quality_threshold: float = 0.8
    confidence_threshold: float = 0.7
    metadata: Dict[str, Any] = field(default_factory=dict)
    
@dataclass
class ModuleResult:
    """Result from a single module analysis."""
    module_name: str
    success: bool
    confidence: float
    quality_score: float
    execution_time: float
    insights: List[str] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
@dataclass
class PipelineState:
    """Current state of the pipeline execution."""
    current_phase: AnalysisPhase
    completed_modules: List[str] = field(default_factory=list)
    active_modules: List[str] = field(default_factory=list)
    module_results: Dict[str, ModuleResult] = field(default_factory=dict)
    overall_confidence: float = 0.0
    overall_quality: float = 0.0
    time_elapsed: float = 0.0
    context_valid: bool = True
    noise_level: float = 0.0
    
@dataclass
class MetacognitiveInsight:
    """Insights from metacognitive analysis."""
    insight_type: str
    confidence: float
    description: str
    supporting_evidence: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

class MetacognitiveOrchestrator:
    """
    The ultimate orchestrator that coordinates all Helicopter modules using
    metacognitive principles to optimize analysis strategy and execution.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the metacognitive orchestrator."""
        self.config = config or self._default_config()
        
        # Initialize all module engines
        self._initialize_modules()
        
        # Learning and adaptation
        self.execution_history: List[Dict[str, Any]] = []
        self.strategy_performance: Dict[AnalysisStrategy, List[float]] = {}
        self.module_reliability: Dict[str, List[float]] = {}
        
        # Metacognitive state
        self.metacognitive_insights: List[MetacognitiveInsight] = []
        self.learning_enabled = True
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for the orchestrator."""
        return {
            "enable_learning": True,
            "max_parallel_modules": 3,
            "default_time_budget": 60.0,
            "quality_thresholds": {
                "minimum": 0.6,
                "good": 0.75,
                "excellent": 0.9
            },
            "confidence_thresholds": {
                "minimum": 0.5,
                "good": 0.7,
                "high": 0.85
            },
            "module_priorities": {
                "autonomous_reconstruction": 1.0,
                "segment_aware": 0.9,
                "zengeza_noise": 0.8,
                "hatata_mdp": 0.7,
                "diadochi": 0.9,
                "nicotine_validation": 0.6
            }
        }
    
    def _initialize_modules(self):
        """Initialize all analysis modules."""
        try:
            # Core reconstruction engines
            self.autonomous_engine = AutonomousReconstructionEngine()
            self.segment_engine = SegmentAwareReconstructionEngine()
            self.comprehensive_engine = ComprehensiveAnalysisEngine()
            
            # Specialized analysis modules
            self.zengeza_engine = ZengezaEngine()
            self.hatata_engine = HatataEngine()
            
            # Context and validation
            self.nicotine_validator = NicotineContextValidator(
                trigger_interval=5,
                puzzle_count=2,
                pass_threshold=0.7
            )
            
            # Expert model combination
            self.diadochi_core = DiadochiCore()
            self._setup_diadochi()
            
            logger.info("✅ All modules initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing modules: {e}")
            raise
    
    def _setup_diadochi(self):
        """Setup Diadochi with domain experts."""
        # Define expertise domains
        domains = [
            DomainExpertise(
                domain="image_reconstruction",
                description="Expert in autonomous image reconstruction and visual understanding through reconstruction",
                keywords=["reconstruction", "autonomous", "visual", "understanding", "quality"]
            ),
            DomainExpertise(
                domain="noise_analysis", 
                description="Expert in detecting and analyzing noise patterns in visual data",
                keywords=["noise", "artifacts", "quality", "distortion", "cleanup"]
            ),
            DomainExpertise(
                domain="probabilistic_reasoning",
                description="Expert in probabilistic analysis and uncertainty quantification",
                keywords=["probability", "uncertainty", "confidence", "bayesian", "markov"]
            ),
            DomainExpertise(
                domain="context_validation",
                description="Expert in maintaining context and validating system understanding",
                keywords=["context", "validation", "consistency", "awareness", "focus"]
            )
        ]
        
        for expertise in domains:
            self.diadochi_core.add_domain_expertise(expertise)
        
        # Register mock models for each domain (in production, use real models)
        for domain in ["image_reconstruction", "noise_analysis", "probabilistic_reasoning", "context_validation"]:
            model = MockModel(f"{domain}_expert", {
                "analyze": f"Expert {domain} analysis provides comprehensive insights",
                "recommend": f"Based on {domain} expertise, I recommend specific approaches"
            })
            self.diadochi_core.register_model(f"{domain}_expert", model, [domain])
        
        # Configure with mixture of experts for multi-domain integration
        self.diadochi_core.configure_mixture_of_experts(threshold=0.2, temperature=0.6)
    
    async def orchestrated_analysis(self, 
                                  image_path: str,
                                  analysis_goals: Optional[List[str]] = None,
                                  strategy: AnalysisStrategy = AnalysisStrategy.ADAPTIVE,
                                  **kwargs) -> Dict[str, Any]:
        """
        Perform comprehensive orchestrated analysis using all available modules.
        """
        # Initialize analysis context
        context = AnalysisContext(
            image_path=image_path,
            analysis_goals=analysis_goals or ["comprehensive_understanding"],
            strategy=strategy,
            **kwargs
        )
        
        # Initialize pipeline state
        state = PipelineState(current_phase=AnalysisPhase.INITIAL_ASSESSMENT)
        start_time = time.time()
        
        try:
            logger.info(f"🚀 Starting orchestrated analysis: {image_path}")
            logger.info(f"📋 Strategy: {strategy.value}, Goals: {context.analysis_goals}")
            
            # Execute analysis pipeline
            results = await self._execute_pipeline(context, state)
            
            # Metacognitive review and learning
            await self._metacognitive_review(context, state, results)
            
            # Update execution history
            execution_record = {
                "timestamp": time.time(),
                "context": context,
                "results": results,
                "execution_time": time.time() - start_time,
                "success": results.get("success", False)
            }
            self.execution_history.append(execution_record)
            
            logger.info(f"✅ Analysis completed in {time.time() - start_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"❌ Pipeline execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time,
                "partial_results": state.module_results
            }
    
    async def _execute_pipeline(self, context: AnalysisContext, state: PipelineState) -> Dict[str, Any]:
        """Execute the complete analysis pipeline."""
        
        # Phase 1: Initial Assessment
        state.current_phase = AnalysisPhase.INITIAL_ASSESSMENT
        await self._initial_assessment(context, state)
        
        # Phase 2: Noise Detection
        state.current_phase = AnalysisPhase.NOISE_DETECTION
        await self._noise_detection_phase(context, state)
        
        # Phase 3: Strategy Selection (Adaptive)
        state.current_phase = AnalysisPhase.STRATEGY_SELECTION
        await self._adaptive_strategy_selection(context, state)
        
        # Phase 4: Reconstruction Analysis
        state.current_phase = AnalysisPhase.RECONSTRUCTION_ANALYSIS
        await self._reconstruction_analysis_phase(context, state)
        
        # Phase 5: Probabilistic Validation
        state.current_phase = AnalysisPhase.PROBABILISTIC_VALIDATION
        await self._probabilistic_validation_phase(context, state)
        
        # Phase 6: Context Validation
        state.current_phase = AnalysisPhase.CONTEXT_VALIDATION
        await self._context_validation_phase(context, state)
        
        # Phase 7: Expert Synthesis
        state.current_phase = AnalysisPhase.EXPERT_SYNTHESIS
        await self._expert_synthesis_phase(context, state)
        
        # Phase 8: Final Integration
        state.current_phase = AnalysisPhase.FINAL_INTEGRATION
        final_results = await self._final_integration_phase(context, state)
        
        return final_results
    
    async def _initial_assessment(self, context: AnalysisContext, state: PipelineState):
        """Perform initial image assessment to guide strategy."""
        logger.info("🔍 Phase 1: Initial Assessment")
        
        try:
            # Quick complexity assessment using comprehensive engine
            quick_results = self.comprehensive_engine.quick_assessment(context.image_path)
            
            # Determine image complexity
            complexity_score = quick_results.get("complexity_score", 0.5)
            if complexity_score < 0.3:
                context.complexity = ImageComplexity.SIMPLE
            elif complexity_score < 0.6:
                context.complexity = ImageComplexity.MODERATE
            elif complexity_score < 0.8:
                context.complexity = ImageComplexity.COMPLEX
            else:
                context.complexity = ImageComplexity.HIGHLY_COMPLEX
            
            # Record initial assessment
            assessment_result = ModuleResult(
                module_name="initial_assessment",
                success=True,
                confidence=quick_results.get("confidence", 0.7),
                quality_score=quick_results.get("quality", 0.7),
                execution_time=quick_results.get("execution_time", 1.0),
                insights=[f"Image complexity: {context.complexity.value}"],
                data=quick_results
            )
            
            state.module_results["initial_assessment"] = assessment_result
            state.completed_modules.append("initial_assessment")
            
            logger.info(f"📊 Complexity: {context.complexity.value}")
            
        except Exception as e:
            logger.error(f"Initial assessment failed: {e}")
    
    async def _noise_detection_phase(self, context: AnalysisContext, state: PipelineState):
        """Detect and analyze noise in the image."""
        logger.info("🔍 Phase 2: Noise Detection")
        
        try:
            start_time = time.time()
            
            # Use Zengeza for noise analysis
            noise_result = await self.zengeza_engine.analyze_image_noise(context.image_path)
            
            execution_time = time.time() - start_time
            
            # Extract noise metrics
            overall_noise = noise_result.get("overall_noise_level", 0.0)
            noise_confidence = noise_result.get("confidence", 0.0)
            noisy_segments = noise_result.get("noisy_segments", [])
            
            state.noise_level = overall_noise
            
            # Create result
            noise_module_result = ModuleResult(
                module_name="zengeza_noise",
                success=True,
                confidence=noise_confidence,
                quality_score=1.0 - overall_noise,  # Lower noise = higher quality
                execution_time=execution_time,
                insights=[
                    f"Overall noise level: {overall_noise:.2%}",
                    f"Noisy segments detected: {len(noisy_segments)}",
                    f"Noise analysis confidence: {noise_confidence:.2%}"
                ],
                data=noise_result,
                recommendations=[
                    "Apply noise reduction if noise > 30%",
                    "Use segment-aware reconstruction for noisy regions",
                    "Increase quality thresholds for noisy images"
                ]
            )
            
            state.module_results["zengeza_noise"] = noise_module_result
            state.completed_modules.append("zengeza_noise")
            
            # Adjust strategy based on noise level
            if overall_noise > 0.4:
                context.quality_threshold = min(context.quality_threshold + 0.1, 0.95)
                logger.info(f"🔧 Adjusted quality threshold to {context.quality_threshold} due to high noise")
            
            logger.info(f"🔍 Noise level: {overall_noise:.1%}")
            
        except Exception as e:
            logger.error(f"Noise detection failed: {e}")
    
    async def _adaptive_strategy_selection(self, context: AnalysisContext, state: PipelineState):
        """Adaptively select the best strategy based on current information."""
        logger.info("🔍 Phase 3: Adaptive Strategy Selection")
        
        if context.strategy != AnalysisStrategy.ADAPTIVE:
            return  # Use predetermined strategy
        
        # Strategy selection based on complexity and noise
        complexity_weight = {
            ImageComplexity.SIMPLE: 0.2,
            ImageComplexity.MODERATE: 0.5,
            ImageComplexity.COMPLEX: 0.8,
            ImageComplexity.HIGHLY_COMPLEX: 1.0
        }[context.complexity]
        
        noise_weight = min(state.noise_level * 2, 1.0)  # Scale noise impact
        
        # Calculate strategy scores
        strategy_scores = {
            AnalysisStrategy.SPEED_OPTIMIZED: 1.0 - complexity_weight - noise_weight,
            AnalysisStrategy.BALANCED: 0.8 - abs(complexity_weight - 0.5) - noise_weight * 0.5,
            AnalysisStrategy.QUALITY_OPTIMIZED: complexity_weight + noise_weight,
            AnalysisStrategy.DEEP_ANALYSIS: complexity_weight * 1.2 + noise_weight
        }
        
        # Select best strategy
        best_strategy = max(strategy_scores, key=strategy_scores.get)
        context.strategy = best_strategy
        
        logger.info(f"🎯 Selected strategy: {best_strategy.value}")
        
        # Adjust parameters based on selected strategy
        if best_strategy == AnalysisStrategy.SPEED_OPTIMIZED:
            context.time_budget = min(context.time_budget, 30.0)
            context.quality_threshold = max(context.quality_threshold - 0.1, 0.6)
        elif best_strategy == AnalysisStrategy.QUALITY_OPTIMIZED:
            context.time_budget = context.time_budget * 1.5
            context.quality_threshold = min(context.quality_threshold + 0.1, 0.95)
        elif best_strategy == AnalysisStrategy.DEEP_ANALYSIS:
            context.time_budget = context.time_budget * 2.0
            context.quality_threshold = min(context.quality_threshold + 0.15, 0.98)
    
    async def _reconstruction_analysis_phase(self, context: AnalysisContext, state: PipelineState):
        """Perform reconstruction-based analysis."""
        logger.info("🔍 Phase 4: Reconstruction Analysis")
        
        reconstruction_tasks = []
        
        # Always run autonomous reconstruction
        reconstruction_tasks.append(self._run_autonomous_reconstruction(context))
        
        # Add segment-aware reconstruction for complex images or high noise
        if context.complexity in [ImageComplexity.COMPLEX, ImageComplexity.HIGHLY_COMPLEX] or state.noise_level > 0.3:
            reconstruction_tasks.append(self._run_segment_aware_reconstruction(context))
        
        # Execute reconstruction tasks
        reconstruction_results = await asyncio.gather(*reconstruction_tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(reconstruction_results):
            if isinstance(result, Exception):
                logger.error(f"Reconstruction task {i} failed: {result}")
            else:
                state.module_results[result.module_name] = result
                state.completed_modules.append(result.module_name)
    
    async def _run_autonomous_reconstruction(self, context: AnalysisContext) -> ModuleResult:
        """Run autonomous reconstruction analysis."""
        start_time = time.time()
        
        try:
            # Configure based on strategy
            max_iterations = {
                AnalysisStrategy.SPEED_OPTIMIZED: 15,
                AnalysisStrategy.BALANCED: 25,
                AnalysisStrategy.QUALITY_OPTIMIZED: 40,
                AnalysisStrategy.DEEP_ANALYSIS: 60
            }.get(context.strategy, 25)
            
            results = self.autonomous_engine.autonomous_analyze(
                image=context.image_path,
                max_iterations=max_iterations,
                target_quality=context.quality_threshold
            )
            
            execution_time = time.time() - start_time
            
            return ModuleResult(
                module_name="autonomous_reconstruction",
                success=True,
                confidence=results.get("final_confidence", 0.0),
                quality_score=results.get("final_quality", 0.0),
                execution_time=execution_time,
                insights=[
                    f"Reconstruction quality: {results.get('final_quality', 0):.2%}",
                    f"Understanding level: {results.get('understanding_level', 'unknown')}",
                    f"Iterations performed: {results.get('iterations_performed', 0)}"
                ],
                data=results,
                recommendations=[
                    "High reconstruction quality indicates good understanding",
                    "Use reconstruction insights for further analysis"
                ]
            )
            
        except Exception as e:
            return ModuleResult(
                module_name="autonomous_reconstruction",
                success=False,
                confidence=0.0,
                quality_score=0.0,
                execution_time=time.time() - start_time,
                data={"error": str(e)}
            )
    
    async def _run_segment_aware_reconstruction(self, context: AnalysisContext) -> ModuleResult:
        """Run segment-aware reconstruction analysis."""
        start_time = time.time()
        
        try:
            results = self.segment_engine.segment_aware_reconstruction(
                image=context.image_path,
                description=f"Complex {context.image_type} image"
            )
            
            execution_time = time.time() - start_time
            
            return ModuleResult(
                module_name="segment_aware_reconstruction",
                success=True,
                confidence=results.get("overall_confidence", 0.0),
                quality_score=results.get("overall_quality", 0.0),
                execution_time=execution_time,
                insights=[
                    f"Segments processed: {results.get('segments_processed', 0)}",
                    f"Successful segments: {results.get('successful_segments', 0)}",
                    f"Overall quality: {results.get('overall_quality', 0):.2%}"
                ],
                data=results,
                recommendations=[
                    "Segment-wise analysis provides detailed understanding",
                    "Focus on segments with low reconstruction quality"
                ]
            )
            
        except Exception as e:
            return ModuleResult(
                module_name="segment_aware_reconstruction",
                success=False,
                confidence=0.0,
                quality_score=0.0,
                execution_time=time.time() - start_time,
                data={"error": str(e)}
            )
    
    async def _probabilistic_validation_phase(self, context: AnalysisContext, state: PipelineState):
        """Perform probabilistic validation using Hatata MDP."""
        logger.info("🔍 Phase 5: Probabilistic Validation")
        
        # Only run for quality/deep analysis strategies or when confidence is uncertain
        if context.strategy in [AnalysisStrategy.SPEED_OPTIMIZED]:
            return
        
        try:
            start_time = time.time()
            
            # Get reconstruction results for probabilistic analysis
            reconstruction_data = {}
            if "autonomous_reconstruction" in state.module_results:
                reconstruction_data.update(state.module_results["autonomous_reconstruction"].data)
            
            # Run Hatata probabilistic analysis
            hatata_results = await self.hatata_engine.probabilistic_understanding_verification(
                image_path=context.image_path,
                reconstruction_data=reconstruction_data,
                confidence_threshold=context.confidence_threshold
            )
            
            execution_time = time.time() - start_time
            
            hatata_result = ModuleResult(
                module_name="hatata_mdp",
                success=True,
                confidence=hatata_results.get("understanding_probability", 0.0),
                quality_score=hatata_results.get("verification_score", 0.0),
                execution_time=execution_time,
                insights=[
                    f"Understanding probability: {hatata_results.get('understanding_probability', 0):.2%}",
                    f"Verification confidence: {hatata_results.get('verification_score', 0):.2%}",
                    f"Probabilistic state: {hatata_results.get('final_state', 'unknown')}"
                ],
                data=hatata_results,
                recommendations=[
                    "High probability indicates reliable understanding",
                    "Use probabilistic bounds for confidence intervals"
                ]
            )
            
            state.module_results["hatata_mdp"] = hatata_result
            state.completed_modules.append("hatata_mdp")
            
        except Exception as e:
            logger.error(f"Probabilistic validation failed: {e}")
    
    async def _context_validation_phase(self, context: AnalysisContext, state: PipelineState):
        """Validate system context and focus."""
        logger.info("🔍 Phase 6: Context Validation")
        
        try:
            # Register current analysis process
            can_continue = self.nicotine_validator.register_process(
                process_name=f"orchestrated_analysis_{int(time.time())}",
                current_task="comprehensive_image_analysis",
                objectives=context.analysis_goals,
                system_state={
                    "completed_modules": len(state.completed_modules),
                    "overall_confidence": state.overall_confidence,
                    "time_elapsed": state.time_elapsed
                }
            )
            
            validation_report = self.nicotine_validator.get_validation_report()
            
            context_result = ModuleResult(
                module_name="nicotine_validation",
                success=can_continue,
                confidence=validation_report.get("pass_rate", 0.0),
                quality_score=1.0 if can_continue else 0.5,
                execution_time=0.5,  # Quick validation
                insights=[
                    f"Context validation: {'✅ Passed' if can_continue else '❌ Failed'}",
                    f"Pass rate: {validation_report.get('pass_rate', 0):.1%}",
                    f"Sessions completed: {validation_report.get('total_sessions', 0)}"
                ],
                data=validation_report,
                recommendations=[
                    "Context validation ensures focus maintenance",
                    "Failed validation indicates potential analysis drift"
                ]
            )
            
            state.module_results["nicotine_validation"] = context_result
            state.completed_modules.append("nicotine_validation")
            state.context_valid = can_continue
            
        except Exception as e:
            logger.error(f"Context validation failed: {e}")
            state.context_valid = True  # Continue on validation failure
    
    async def _expert_synthesis_phase(self, context: AnalysisContext, state: PipelineState):
        """Synthesize insights using Diadochi expert combination."""
        logger.info("🔍 Phase 7: Expert Synthesis")
        
        try:
            # Prepare synthesis query
            completed_modules = ", ".join(state.completed_modules)
            overall_quality = statistics.mean([r.quality_score for r in state.module_results.values() if r.success])
            overall_confidence = statistics.mean([r.confidence for r in state.module_results.values() if r.success])
            
            synthesis_query = f"""
            Analyze and synthesize insights from comprehensive image analysis:
            
            Image: {context.image_path}
            Completed modules: {completed_modules}
            Overall quality: {overall_quality:.2%}
            Overall confidence: {overall_confidence:.2%}
            Noise level: {state.noise_level:.2%}
            Strategy used: {context.strategy.value}
            
            Provide expert synthesis of findings and recommendations.
            """
            
            start_time = time.time()
            expert_response = await self.diadochi_core.generate(synthesis_query)
            execution_time = time.time() - start_time
            
            synthesis_result = ModuleResult(
                module_name="diadochi_synthesis",
                success=True,
                confidence=overall_confidence,
                quality_score=overall_quality,
                execution_time=execution_time,
                insights=[
                    "Expert synthesis combines all module insights",
                    f"Multi-domain analysis confidence: {overall_confidence:.2%}",
                    "Integrated recommendations provided"
                ],
                data={"expert_response": expert_response},
                recommendations=[
                    "Expert synthesis provides integrated understanding",
                    "Use synthesized insights for decision making"
                ]
            )
            
            state.module_results["diadochi_synthesis"] = synthesis_result
            state.completed_modules.append("diadochi_synthesis")
            
        except Exception as e:
            logger.error(f"Expert synthesis failed: {e}")
    
    async def _final_integration_phase(self, context: AnalysisContext, state: PipelineState) -> Dict[str, Any]:
        """Integrate all results into final comprehensive analysis."""
        logger.info("🔍 Phase 8: Final Integration")
        
        # Calculate overall metrics
        successful_modules = [r for r in state.module_results.values() if r.success]
        
        if successful_modules:
            state.overall_quality = statistics.mean([r.quality_score for r in successful_modules])
            state.overall_confidence = statistics.mean([r.confidence for r in successful_modules])
        
        total_execution_time = sum([r.execution_time for r in state.module_results.values()])
        
        # Collect all insights
        all_insights = []
        all_recommendations = []
        
        for result in successful_modules:
            all_insights.extend(result.insights)
            all_recommendations.extend(result.recommendations)
        
        # Generate final assessment
        final_assessment = self._generate_final_assessment(context, state, successful_modules)
        
        return {
            "success": len(successful_modules) > 0,
            "overall_quality": state.overall_quality,
            "overall_confidence": state.overall_confidence,
            "execution_time": total_execution_time,
            "modules_executed": len(state.completed_modules),
            "successful_modules": len(successful_modules),
            "context_maintained": state.context_valid,
            "noise_level": state.noise_level,
            "strategy_used": context.strategy.value,
            "image_complexity": context.complexity.value,
            "final_assessment": final_assessment,
            "all_insights": list(set(all_insights)),  # Remove duplicates
            "all_recommendations": list(set(all_recommendations)),
            "module_results": {name: result for name, result in state.module_results.items()},
            "metacognitive_insights": [insight.__dict__ for insight in self.metacognitive_insights]
        }
    
    def _generate_final_assessment(self, 
                                 context: AnalysisContext, 
                                 state: PipelineState, 
                                 successful_modules: List[ModuleResult]) -> Dict[str, Any]:
        """Generate final comprehensive assessment."""
        
        # Quality assessment
        if state.overall_quality >= 0.9:
            quality_level = "Excellent"
        elif state.overall_quality >= 0.75:
            quality_level = "Good"
        elif state.overall_quality >= 0.6:
            quality_level = "Acceptable"
        else:
            quality_level = "Poor"
        
        # Confidence assessment
        if state.overall_confidence >= 0.85:
            confidence_level = "High"
        elif state.overall_confidence >= 0.7:
            confidence_level = "Moderate"
        else:
            confidence_level = "Low"
        
        # Understanding level
        understanding_indicators = []
        if "autonomous_reconstruction" in state.module_results:
            recon_quality = state.module_results["autonomous_reconstruction"].quality_score
            if recon_quality >= 0.8:
                understanding_indicators.append("Strong reconstruction ability")
        
        if "segment_aware_reconstruction" in state.module_results:
            segment_quality = state.module_results["segment_aware_reconstruction"].quality_score
            if segment_quality >= 0.75:
                understanding_indicators.append("Good segment-level understanding")
        
        if "hatata_mdp" in state.module_results:
            prob_confidence = state.module_results["hatata_mdp"].confidence
            if prob_confidence >= 0.7:
                understanding_indicators.append("Probabilistically validated understanding")
        
        # Generate recommendations
        recommendations = []
        if state.overall_quality < 0.7:
            recommendations.append("Consider additional analysis or different approach")
        if state.noise_level > 0.4:
            recommendations.append("Apply noise reduction preprocessing")
        if not state.context_valid:
            recommendations.append("Review analysis focus and objectives")
        if state.overall_confidence < 0.6:
            recommendations.append("Gather additional evidence or expert validation")
        
        return {
            "quality_level": quality_level,
            "confidence_level": confidence_level,
            "understanding_indicators": understanding_indicators,
            "analysis_completeness": len(successful_modules) / len(state.module_results),
            "strategy_effectiveness": self._assess_strategy_effectiveness(context, state),
            "recommendations": recommendations,
            "summary": f"{quality_level} quality analysis with {confidence_level.lower()} confidence using {context.strategy.value} strategy"
        }
    
    def _assess_strategy_effectiveness(self, context: AnalysisContext, state: PipelineState) -> str:
        """Assess how effective the chosen strategy was."""
        quality_score = state.overall_quality
        time_efficiency = 1.0 - min(state.time_elapsed / context.time_budget, 1.0)
        
        effectiveness_score = (quality_score + time_efficiency) / 2
        
        if effectiveness_score >= 0.8:
            return "Highly effective"
        elif effectiveness_score >= 0.6:
            return "Moderately effective"
        else:
            return "Less effective"
    
    async def _metacognitive_review(self, 
                                  context: AnalysisContext, 
                                  state: PipelineState, 
                                  results: Dict[str, Any]):
        """Perform metacognitive review and learning."""
        logger.info("🔍 Phase 9: Metacognitive Review")
        
        if not self.learning_enabled:
            return
        
        # Analyze strategy performance
        strategy_score = results["overall_quality"] * results["overall_confidence"]
        
        if context.strategy not in self.strategy_performance:
            self.strategy_performance[context.strategy] = []
        self.strategy_performance[context.strategy].append(strategy_score)
        
        # Analyze module reliability
        for module_name, result in state.module_results.items():
            if module_name not in self.module_reliability:
                self.module_reliability[module_name] = []
            
            reliability_score = result.confidence if result.success else 0.0
            self.module_reliability[module_name].append(reliability_score)
        
        # Generate metacognitive insights
        await self._generate_metacognitive_insights(context, state, results)
        
        # Update configuration based on learning
        self._adaptive_configuration_update()
    
    async def _generate_metacognitive_insights(self, 
                                             context: AnalysisContext, 
                                             state: PipelineState, 
                                             results: Dict[str, Any]):
        """Generate insights about the analysis process itself."""
        
        insights = []
        
        # Strategy effectiveness insight
        if len(self.strategy_performance.get(context.strategy, [])) > 3:
            avg_performance = statistics.mean(self.strategy_performance[context.strategy][-5:])
            if avg_performance > 0.8:
                insights.append(MetacognitiveInsight(
                    insight_type="strategy_effectiveness",
                    confidence=0.8,
                    description=f"{context.strategy.value} strategy shows consistently high performance",
                    supporting_evidence=[f"Average performance: {avg_performance:.2%}"],
                    recommendations=[f"Continue using {context.strategy.value} for similar cases"]
                ))
        
        # Module reliability insight
        unreliable_modules = []
        for module, scores in self.module_reliability.items():
            if len(scores) > 3:
                avg_reliability = statistics.mean(scores[-5:])
                if avg_reliability < 0.6:
                    unreliable_modules.append((module, avg_reliability))
        
        if unreliable_modules:
            insights.append(MetacognitiveInsight(
                insight_type="module_reliability",
                confidence=0.7,
                description="Some modules showing reduced reliability",
                supporting_evidence=[f"{mod}: {score:.2%}" for mod, score in unreliable_modules],
                recommendations=["Review module configurations", "Consider alternative approaches"]
            ))
        
        # Quality-complexity relationship
        if context.complexity == ImageComplexity.HIGHLY_COMPLEX and results["overall_quality"] > 0.8:
            insights.append(MetacognitiveInsight(
                insight_type="complexity_handling",
                confidence=0.9,
                description="System handles complex images effectively",
                supporting_evidence=[f"Quality {results['overall_quality']:.2%} on {context.complexity.value} image"],
                recommendations=["Maintain current approach for complex images"]
            ))
        
        self.metacognitive_insights.extend(insights)
        
        # Keep only recent insights
        if len(self.metacognitive_insights) > 20:
            self.metacognitive_insights = self.metacognitive_insights[-20:]
    
    def _adaptive_configuration_update(self):
        """Update configuration based on learning."""
        
        # Adjust module priorities based on reliability
        for module, scores in self.module_reliability.items():
            if len(scores) > 5:
                avg_reliability = statistics.mean(scores[-10:])
                if module in self.config["module_priorities"]:
                    # Adjust priority based on reliability
                    current_priority = self.config["module_priorities"][module]
                    new_priority = current_priority * 0.9 + avg_reliability * 0.1
                    self.config["module_priorities"][module] = new_priority
        
        logger.info("🧠 Configuration updated based on learning")
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of learning and adaptation."""
        return {
            "executions_completed": len(self.execution_history),
            "strategy_performance": {
                strategy.value: statistics.mean(scores) if scores else 0.0
                for strategy, scores in self.strategy_performance.items()
            },
            "module_reliability": {
                module: statistics.mean(scores[-10:]) if scores else 0.0
                for module, scores in self.module_reliability.items()
            },
            "recent_insights": [insight.__dict__ for insight in self.metacognitive_insights[-5:]],
            "configuration_updates": self.config.get("last_update_time", "Never")
        }
    
    def save_learning_state(self, filepath: str):
        """Save learning state to file."""
        learning_state = {
            "execution_history": self.execution_history[-50:],  # Keep recent history
            "strategy_performance": self.strategy_performance,
            "module_reliability": self.module_reliability,
            "metacognitive_insights": [insight.__dict__ for insight in self.metacognitive_insights],
            "config": self.config
        }
        
        with open(filepath, 'w') as f:
            json.dump(learning_state, f, indent=2, default=str)
        
        logger.info(f"💾 Learning state saved to {filepath}")
    
    def load_learning_state(self, filepath: str):
        """Load learning state from file."""
        try:
            with open(filepath, 'r') as f:
                learning_state = json.load(f)
            
            self.execution_history = learning_state.get("execution_history", [])
            self.strategy_performance = learning_state.get("strategy_performance", {})
            self.module_reliability = learning_state.get("module_reliability", {})
            
            # Reconstruct metacognitive insights
            insights_data = learning_state.get("metacognitive_insights", [])
            self.metacognitive_insights = [
                MetacognitiveInsight(**insight_data) for insight_data in insights_data
            ]
            
            # Update configuration
            saved_config = learning_state.get("config", {})
            self.config.update(saved_config)
            
            logger.info(f"📂 Learning state loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load learning state: {e}")

# Export main classes
__all__ = [
    "MetacognitiveOrchestrator",
    "AnalysisStrategy",
    "ImageComplexity",
    "AnalysisContext",
    "ModuleResult",
    "PipelineState",
    "MetacognitiveInsight"
] 