"""
Iterative Expert Research System

This implements the user's brilliant research-oriented approach:
1. Build domain expert LLM from scientific literature
2. Use metacognitive orchestrator to guide analysis
3. Continuously learn and improve with each new image
4. Iterate until confidence threshold is reached
5. Get smarter with every research dataset processed

This is the true "Reverse Helicopter" for research - not just template matching,
but building genuine scientific expertise that improves over time.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
import faiss
from sentence_transformers import SentenceTransformer
import cv2
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class ScientificKnowledge:
    """Represents extracted knowledge from scientific literature"""
    domain: str
    concept: str
    description: str
    expected_visual_features: List[str]
    typical_variations: List[str]
    confidence_indicators: List[str]
    source_papers: List[str]
    embedding: Optional[np.ndarray] = None


@dataclass
class AnalysisResult:
    """Result of analyzing a single image"""
    image_id: str
    iteration: int
    confidence_score: float
    findings: List[Dict[str, Any]]
    expert_reasoning: str
    suggested_annotations: List[Dict[str, Any]]
    learning_feedback: Dict[str, Any]
    needs_reanalysis: bool


@dataclass
class LearningState:
    """Current state of the learning system"""
    total_images_seen: int
    current_confidence: float
    knowledge_base_size: int
    iteration_count: int
    performance_metrics: Dict[str, float]
    convergence_indicators: Dict[str, float]


class DomainExpertLLM:
    """
    Domain-specific LLM built from scientific literature
    This becomes the "expert brain" of the system
    """
    
    def __init__(
        self,
        domain: str,
        base_model: str = "microsoft/DialoGPT-medium",
        device: str = "cuda"
    ):
        self.domain = domain
        self.device = device
        
        # Load base model
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForCausalLM.from_pretrained(base_model)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Knowledge base for scientific facts
        self.knowledge_base: List[ScientificKnowledge] = []
        
        # Embedding model for semantic search
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # FAISS index for fast knowledge retrieval
        self.knowledge_index = None
        
        logger.info(f"Initialized Domain Expert LLM for: {domain}")
    
    def ingest_scientific_literature(
        self,
        papers: List[Dict[str, Any]],
        extract_visual_knowledge: bool = True
    ):
        """
        Ingest scientific papers and extract domain knowledge
        
        Args:
            papers: List of papers with 'title', 'abstract', 'content', 'domain_tags'
            extract_visual_knowledge: Whether to focus on visual analysis knowledge
        """
        
        logger.info(f"Ingesting {len(papers)} papers for domain: {self.domain}")
        
        for paper in papers:
            # Extract domain-specific knowledge
            knowledge_items = self._extract_knowledge_from_paper(
                paper, extract_visual_knowledge
            )
            
            self.knowledge_base.extend(knowledge_items)
        
        # Build semantic search index
        self._build_knowledge_index()
        
        # Fine-tune the LLM on domain knowledge
        self._fine_tune_on_domain_knowledge()
        
        logger.info(f"Knowledge base now contains {len(self.knowledge_base)} concepts")
    
    def _extract_knowledge_from_paper(
        self,
        paper: Dict[str, Any],
        focus_visual: bool
    ) -> List[ScientificKnowledge]:
        """Extract structured knowledge from a single paper"""
        
        knowledge_items = []
        
        # Simple knowledge extraction (in practice, use more sophisticated NLP)
        content = paper.get('content', '') + ' ' + paper.get('abstract', '')
        
        # Look for visual analysis patterns
        if focus_visual:
            visual_keywords = [
                'microscopy', 'imaging', 'visualization', 'morphology',
                'structure', 'appearance', 'feature', 'pattern',
                'texture', 'shape', 'size', 'intensity', 'contrast'
            ]
            
            sentences = content.split('.')
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in visual_keywords):
                    # Extract expected features
                    concept = self._extract_concept_from_sentence(sentence)
                    if concept:
                        knowledge = ScientificKnowledge(
                            domain=self.domain,
                            concept=concept,
                            description=sentence.strip(),
                            expected_visual_features=self._extract_visual_features(sentence),
                            typical_variations=self._extract_variations(sentence),
                            confidence_indicators=self._extract_confidence_indicators(sentence),
                            source_papers=[paper.get('title', 'Unknown')]
                        )
                        knowledge_items.append(knowledge)
        
        return knowledge_items
    
    def _extract_concept_from_sentence(self, sentence: str) -> Optional[str]:
        """Extract the main concept from a sentence"""
        # Simplified concept extraction
        words = sentence.lower().split()
        
        # Look for key domain terms
        domain_terms = {
            'medical': ['cell', 'tissue', 'organ', 'lesion', 'abnormality'],
            'biology': ['organism', 'structure', 'protein', 'membrane'],
            'materials': ['crystal', 'grain', 'defect', 'surface', 'interface']
        }
        
        for term in domain_terms.get(self.domain, []):
            if term in words:
                return term
        
        return None
    
    def _extract_visual_features(self, sentence: str) -> List[str]:
        """Extract visual features mentioned in sentence"""
        features = []
        
        feature_patterns = [
            'bright', 'dark', 'round', 'elongated', 'smooth', 'rough',
            'uniform', 'heterogeneous', 'dense', 'sparse', 'linear', 'curved'
        ]
        
        for pattern in feature_patterns:
            if pattern in sentence.lower():
                features.append(pattern)
        
        return features
    
    def _extract_variations(self, sentence: str) -> List[str]:
        """Extract typical variations mentioned"""
        variations = []
        
        variation_keywords = ['varies', 'different', 'range', 'variable', 'diverse']
        
        if any(keyword in sentence.lower() for keyword in variation_keywords):
            variations.append("Natural variation expected")
        
        return variations
    
    def _extract_confidence_indicators(self, sentence: str) -> List[str]:
        """Extract confidence indicators from literature"""
        indicators = []
        
        confidence_patterns = [
            'typically', 'usually', 'often', 'commonly', 'characteristic',
            'diagnostic', 'pathognomonic', 'specific', 'consistent'
        ]
        
        for pattern in confidence_patterns:
            if pattern in sentence.lower():
                indicators.append(pattern)
        
        return indicators
    
    def _build_knowledge_index(self):
        """Build FAISS index for fast knowledge retrieval"""
        
        if not self.knowledge_base:
            return
        
        # Create embeddings for all knowledge items
        descriptions = [k.description for k in self.knowledge_base]
        embeddings = self.embedding_model.encode(descriptions)
        
        # Store embeddings in knowledge items
        for i, knowledge in enumerate(self.knowledge_base):
            knowledge.embedding = embeddings[i]
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.knowledge_index = faiss.IndexFlatIP(dimension)
        self.knowledge_index.add(embeddings.astype('float32'))
        
        logger.info(f"Built knowledge index with {len(self.knowledge_base)} items")
    
    def _fine_tune_on_domain_knowledge(self):
        """Fine-tune the LLM on domain-specific knowledge"""
        
        if not self.knowledge_base:
            return
        
        # Prepare training data
        training_texts = []
        for knowledge in self.knowledge_base:
            # Create training examples
            example = f"Domain: {knowledge.domain}\nConcept: {knowledge.concept}\nDescription: {knowledge.description}\nExpected features: {', '.join(knowledge.expected_visual_features)}"
            training_texts.append(example)
        
        # Tokenize training data
        tokenized_data = self.tokenizer(
            training_texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Create dataset
        dataset = Dataset.from_dict({
            'input_ids': tokenized_data['input_ids'],
            'attention_mask': tokenized_data['attention_mask'],
            'labels': tokenized_data['input_ids']
        })
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"./expert_llm_{self.domain}",
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            save_steps=10_000,
            save_total_limit=2,
            logging_steps=500,
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
        )
        
        # Fine-tune
        logger.info("Fine-tuning expert LLM on domain knowledge...")
        trainer.train()
        
        logger.info("Fine-tuning completed!")
    
    def query_knowledge(
        self,
        query: str,
        top_k: int = 5
    ) -> List[ScientificKnowledge]:
        """Query the knowledge base for relevant information"""
        
        if not self.knowledge_index:
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        
        # Search knowledge base
        scores, indices = self.knowledge_index.search(
            query_embedding.astype('float32'), top_k
        )
        
        # Return relevant knowledge
        relevant_knowledge = []
        for i, score in zip(indices[0], scores[0]):
            if score > 0.5:  # Relevance threshold
                relevant_knowledge.append(self.knowledge_base[i])
        
        return relevant_knowledge
    
    def generate_expert_analysis(
        self,
        observation: str,
        context: str = ""
    ) -> str:
        """Generate expert analysis using domain knowledge"""
        
        # Query relevant knowledge
        relevant_knowledge = self.query_knowledge(observation)
        
        # Construct prompt with domain expertise
        knowledge_context = "\n".join([
            f"- {k.concept}: {k.description}" 
            for k in relevant_knowledge[:3]
        ])
        
        prompt = f"""
As a {self.domain} expert, analyze this observation:

Observation: {observation}
Context: {context}

Relevant scientific knowledge:
{knowledge_context}

Expert analysis:"""
        
        # Generate response
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + 150,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        analysis = response[len(prompt):].strip()
        
        return analysis


class MetacognitiveOrchestrator:
    """
    High-level orchestrator that uses expert LLM to guide analysis
    This is the "thinking brain" that decides what to do next
    """
    
    def __init__(self, expert_llm: DomainExpertLLM):
        self.expert_llm = expert_llm
        self.learning_state = LearningState(
            total_images_seen=0,
            current_confidence=0.0,
            knowledge_base_size=len(expert_llm.knowledge_base),
            iteration_count=0,
            performance_metrics={},
            convergence_indicators={}
        )
        
    def should_continue_iterating(self, results: List[AnalysisResult]) -> bool:
        """Decide if we need another iteration"""
        
        if not results:
            return True
        
        # Calculate average confidence
        avg_confidence = np.mean([r.confidence_score for r in results])
        
        # Check for convergence
        needs_reanalysis = any(r.needs_reanalysis for r in results)
        
        # Confidence threshold
        confidence_threshold = 0.85
        
        # Continue if confidence is low or reanalysis needed
        should_continue = (
            avg_confidence < confidence_threshold or 
            needs_reanalysis or
            self.learning_state.iteration_count < 2  # Minimum iterations
        )
        
        logger.info(f"Iteration {self.learning_state.iteration_count}: "
                   f"Confidence={avg_confidence:.3f}, Continue={should_continue}")
        
        return should_continue
    
    def plan_next_iteration(
        self,
        previous_results: List[AnalysisResult],
        new_images: List[str]
    ) -> Dict[str, Any]:
        """Plan the next iteration based on previous results"""
        
        plan = {
            "iteration_number": self.learning_state.iteration_count + 1,
            "focus_areas": [],
            "analysis_priorities": [],
            "learning_updates": [],
            "confidence_targets": {}
        }
        
        # Analyze what went wrong in previous iteration
        if previous_results:
            low_confidence_areas = [
                r for r in previous_results 
                if r.confidence_score < 0.7
            ]
            
            for result in low_confidence_areas:
                plan["focus_areas"].append(result.image_id)
                plan["analysis_priorities"].append("detailed_reexamination")
        
        # Plan learning updates
        if new_images:
            plan["learning_updates"].append("incorporate_new_samples")
            plan["confidence_targets"]["new_images"] = 0.8
        
        return plan
    
    def orchestrate_analysis_iteration(
        self,
        images: List[str],
        iteration_plan: Dict[str, Any]
    ) -> List[AnalysisResult]:
        """Orchestrate a complete analysis iteration"""
        
        results = []
        
        for image_path in images:
            logger.info(f"Analyzing {image_path} (iteration {iteration_plan['iteration_number']})")
            
            # Analyze image with current expert knowledge
            result = self._analyze_single_image(
                image_path, 
                iteration_plan['iteration_number']
            )
            
            results.append(result)
            
            # Update learning state
            self._update_learning_state(result)
        
        return results
    
    def _analyze_single_image(
        self,
        image_path: str,
        iteration: int
    ) -> AnalysisResult:
        """Analyze a single image using expert LLM guidance"""
        
        # Load and preprocess image
        image = cv2.imread(image_path)
        
        # Extract basic visual features (simplified)
        features = self._extract_basic_features(image)
        
        # Query expert knowledge
        observation = f"Image shows: {features}"
        expert_analysis = self.expert_llm.generate_expert_analysis(observation)
        
        # Calculate confidence based on expert knowledge match
        confidence = self._calculate_confidence(features, expert_analysis)
        
        # Generate findings
        findings = self._extract_findings(expert_analysis, features)
        
        # Suggest annotations
        suggestions = self._suggest_annotations(expert_analysis, image.shape)
        
        # Determine if reanalysis is needed
        needs_reanalysis = confidence < 0.7 or len(findings) < 2
        
        return AnalysisResult(
            image_id=Path(image_path).stem,
            iteration=iteration,
            confidence_score=confidence,
            findings=findings,
            expert_reasoning=expert_analysis,
            suggested_annotations=suggestions,
            learning_feedback={
                "feature_extraction_quality": len(features),
                "expert_knowledge_relevance": confidence
            },
            needs_reanalysis=needs_reanalysis
        )
    
    def _extract_basic_features(self, image: np.ndarray) -> str:
        """Extract basic visual features (placeholder)"""
        
        h, w = image.shape[:2]
        
        # Basic statistics
        mean_intensity = np.mean(image)
        std_intensity = np.std(image)
        
        # Simple texture analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (h * w)
        
        # Color analysis
        color_channels = cv2.split(image)
        dominant_channel = np.argmax([np.mean(ch) for ch in color_channels])
        channel_names = ['blue', 'green', 'red']
        
        features = f"size {w}x{h}, mean intensity {mean_intensity:.1f}, " \
                  f"texture density {edge_density:.3f}, " \
                  f"dominant color {channel_names[dominant_channel]}"
        
        return features
    
    def _calculate_confidence(self, features: str, expert_analysis: str) -> float:
        """Calculate confidence based on feature-analysis alignment"""
        
        # Simple confidence calculation
        feature_words = set(features.lower().split())
        analysis_words = set(expert_analysis.lower().split())
        
        overlap = len(feature_words & analysis_words)
        total = len(feature_words | analysis_words)
        
        confidence = overlap / total if total > 0 else 0.0
        
        # Add bonus for expert terminology
        expert_terms = ['characteristic', 'typical', 'consistent', 'diagnostic']
        if any(term in expert_analysis.lower() for term in expert_terms):
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def _extract_findings(
        self,
        expert_analysis: str,
        features: str
    ) -> List[Dict[str, Any]]:
        """Extract structured findings from expert analysis"""
        
        findings = []
        
        # Simple finding extraction
        sentences = expert_analysis.split('.')
        for sentence in sentences:
            if len(sentence.strip()) > 10:  # Skip short fragments
                finding = {
                    "description": sentence.strip(),
                    "confidence": 0.7,  # Placeholder
                    "type": "observation"
                }
                findings.append(finding)
        
        return findings
    
    def _suggest_annotations(
        self,
        expert_analysis: str,
        image_shape: Tuple[int, int, int]
    ) -> List[Dict[str, Any]]:
        """Suggest annotations based on expert analysis"""
        
        suggestions = []
        
        h, w, _ = image_shape
        
        # Simple annotation suggestions
        if "center" in expert_analysis.lower():
            suggestions.append({
                "type": "region",
                "location": (w//4, h//4, 3*w//4, 3*h//4),
                "label": "central_region",
                "reason": "Expert mentioned central features"
            })
        
        if "edge" in expert_analysis.lower():
            suggestions.append({
                "type": "boundary",
                "location": (10, 10, w-10, h-10),
                "label": "boundary_region",
                "reason": "Expert mentioned edge features"
            })
        
        return suggestions
    
    def _update_learning_state(self, result: AnalysisResult):
        """Update learning state based on analysis result"""
        
        self.learning_state.total_images_seen += 1
        
        # Update running confidence average
        total_weight = self.learning_state.total_images_seen
        prev_weight = total_weight - 1
        
        self.learning_state.current_confidence = (
            (prev_weight * self.learning_state.current_confidence + result.confidence_score) 
            / total_weight
        )
        
        # Update performance metrics
        self.learning_state.performance_metrics[result.image_id] = result.confidence_score


class IterativeResearchSystem:
    """
    Main research system that orchestrates iterative learning
    """
    
    def __init__(self, domain: str):
        self.domain = domain
        self.expert_llm = DomainExpertLLM(domain)
        self.orchestrator = MetacognitiveOrchestrator(self.expert_llm)
        self.iteration_history: List[List[AnalysisResult]] = []
    
    def ingest_domain_literature(self, papers: List[Dict[str, Any]]):
        """Ingest scientific literature to build expertise"""
        self.expert_llm.ingest_scientific_literature(papers)
    
    def analyze_research_dataset(
        self,
        image_paths: List[str],
        max_iterations: int = 5
    ) -> Dict[str, Any]:
        """
        Analyze research dataset with iterative learning
        
        This is the main research function that:
        1. Analyzes all images
        2. Learns from results
        3. Re-analyzes with improved knowledge
        4. Continues until confident or max iterations
        """
        
        logger.info(f"Starting iterative analysis of {len(image_paths)} images")
        
        current_images = image_paths.copy()
        
        for iteration in range(max_iterations):
            logger.info(f"\n=== ITERATION {iteration + 1} ===")
            
            # Plan this iteration
            previous_results = self.iteration_history[-1] if self.iteration_history else []
            iteration_plan = self.orchestrator.plan_next_iteration(
                previous_results, current_images
            )
            
            # Execute analysis
            results = self.orchestrator.orchestrate_analysis_iteration(
                current_images, iteration_plan
            )
            
            self.iteration_history.append(results)
            
            # Check if we should continue
            if not self.orchestrator.should_continue_iterating(results):
                logger.info("Convergence achieved! Stopping iterations.")
                break
            
            # Update expert knowledge based on results
            self._update_expert_knowledge(results)
            
            # Prepare for next iteration (might add new images or focus on problematic ones)
            current_images = self._select_images_for_next_iteration(results, image_paths)
        
        # Generate final analysis
        final_analysis = self._generate_final_analysis()
        
        return final_analysis
    
    def _update_expert_knowledge(self, results: List[AnalysisResult]):
        """Update expert knowledge based on analysis results"""
        
        # Extract new knowledge from successful analyses
        high_confidence_results = [r for r in results if r.confidence_score > 0.8]
        
        for result in high_confidence_results:
            # Create new knowledge items from successful analyses
            new_knowledge = ScientificKnowledge(
                domain=self.domain,
                concept=f"learned_pattern_{result.image_id}",
                description=result.expert_reasoning,
                expected_visual_features=[f.get('description', '') for f in result.findings],
                typical_variations=["Observed variation"],
                confidence_indicators=["High confidence analysis"],
                source_papers=["Learned from data"]
            )
            
            self.expert_llm.knowledge_base.append(new_knowledge)
        
        # Rebuild knowledge index with new information
        if high_confidence_results:
            self.expert_llm._build_knowledge_index()
            logger.info(f"Updated knowledge base with {len(high_confidence_results)} new patterns")
    
    def _select_images_for_next_iteration(
        self,
        results: List[AnalysisResult],
        all_images: List[str]
    ) -> List[str]:
        """Select images for next iteration based on results"""
        
        # Focus on images that need reanalysis
        problematic_ids = [r.image_id for r in results if r.needs_reanalysis]
        
        # Map back to full paths
        selected_images = []
        for image_path in all_images:
            image_id = Path(image_path).stem
            if image_id in problematic_ids:
                selected_images.append(image_path)
        
        # If no problematic images, analyze all again with improved knowledge
        if not selected_images:
            selected_images = all_images
        
        return selected_images
    
    def _generate_final_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive final analysis"""
        
        final_results = self.iteration_history[-1] if self.iteration_history else []
        
        analysis = {
            "total_iterations": len(self.iteration_history),
            "final_confidence": np.mean([r.confidence_score for r in final_results]),
            "total_images_analyzed": len(final_results),
            "convergence_achieved": not any(r.needs_reanalysis for r in final_results),
            "learning_progression": [
                np.mean([r.confidence_score for r in iteration_results])
                for iteration_results in self.iteration_history
            ],
            "final_findings": final_results,
            "expert_knowledge_growth": {
                "initial_knowledge": 0,  # Would track initial size
                "final_knowledge": len(self.expert_llm.knowledge_base),
                "learned_patterns": len([k for k in self.expert_llm.knowledge_base 
                                       if "learned_pattern" in k.concept])
            }
        }
        
        return analysis


# Convenience functions for research workflows

def analyze_microscopy_dataset(
    image_directory: str,
    literature_papers: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Analyze microscopy research dataset"""
    
    system = IterativeResearchSystem("microscopy")
    
    # Build expertise from literature
    system.ingest_domain_literature(literature_papers)
    
    # Find all images
    image_paths = list(Path(image_directory).glob("*.jpg")) + \
                  list(Path(image_directory).glob("*.png"))
    
    # Analyze with iterative learning
    results = system.analyze_research_dataset([str(p) for p in image_paths])
    
    return results


def analyze_medical_imaging_study(
    dicom_directory: str,
    clinical_protocols: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Analyze medical imaging research study"""
    
    system = IterativeResearchSystem("medical_imaging")
    
    # Build expertise from clinical protocols and papers
    system.ingest_domain_literature(clinical_protocols)
    
    # Process DICOM files (simplified)
    image_paths = list(Path(dicom_directory).glob("*.dcm"))
    
    results = system.analyze_research_dataset([str(p) for p in image_paths])
    
    return results 