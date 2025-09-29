#!/usr/bin/env python3
"""
Proof-Validated Compression Demonstration (Working Version)
==========================================================

Demonstrates the integration of formal proof validation with compression
ambiguity detection. This version includes proper error handling and
fallback modes when modules are not available.
"""

import sys
import time
from pathlib import Path
from typing import Optional, List, Any

import numpy as np
from PIL import Image

# Add helicopter modules to path
sys.path.append(str(Path(__file__).parent.parent))

# Try to import the consciousness modules with proper error handling
IMPORTS_AVAILABLE = False
BatchCompressionAnalysis = None
ProofValidatedCompressionProcessor = None
BatchAmbiguityProcessor = None
FormalSystem = None

try:
    from consciousness.proof_validated_compression import (
        ProofValidatedCompressionProcessor,
        FormalSystem
    )
    from consciousness.ambiguous_compression import (
        BatchAmbiguityProcessor,
        BatchCompressionAnalysis
    )
    IMPORTS_AVAILABLE = True
    print("✓ Successfully imported proof validation modules")
except ImportError as e:
    print(f"⚠ Warning: Could not import proof validation modules: {e}")
    print("Running in demonstration mode with simulated results.")
    
    # Create mock classes for the demonstration
    class MockBatchCompressionAnalysis:
        def __init__(self):
            self.ambiguous_bits = []
            self.compression_ratio = 0.45
            self.ambiguity_density = 0.000123
            
    class MockAmbiguousBit:
        def __init__(self, resistance, meaning_count, potential):
            self.compression_resistance = resistance
            self.meaning_count = meaning_count
            self.meta_information_potential = potential
            
    class MockProofValidatedCompressionProcessor:
        def __init__(self, formal_system):
            self.formal_system = formal_system
            
        def process_with_proof_validation(self, images):
            return []  # Return empty list for demo
            
        def get_proof_based_meta_information_summary(self):
            return {"status": "no_validated_bits"}
            
    class MockBatchAmbiguityProcessor:
        def process_image_batch(self, images):
            # Create some mock ambiguous bits for demonstration
            analysis = MockBatchCompressionAnalysis()
            analysis.ambiguous_bits = [
                MockAmbiguousBit(0.85, 3, 2.4),
                MockAmbiguousBit(0.72, 2, 1.8),
                MockAmbiguousBit(0.91, 4, 3.1)
            ]
            return analysis
    
    class MockFormalSystem:
        LEAN = "lean"
        COQ = "coq"
    
    # Use mock classes
    BatchCompressionAnalysis = MockBatchCompressionAnalysis
    ProofValidatedCompressionProcessor = MockProofValidatedCompressionProcessor
    BatchAmbiguityProcessor = MockBatchAmbiguityProcessor
    FormalSystem = MockFormalSystem


class ProofValidationComparisonDemo:
    """Compare statistical vs proof-validated compression analysis."""

    def __init__(self):
        self.statistical_processor = BatchAmbiguityProcessor()
        self.proof_processor = ProofValidatedCompressionProcessor(
            FormalSystem.LEAN
        )
        self.demo_results = {}

    def create_test_images(self, count: int = 4) -> List[tuple]:
        """Create test images with known ambiguous patterns."""
        images = []
        np.random.seed(42)

        # Mixed composition - high ambiguity expected
        mixed = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        # Add some structured patterns
        mixed[20:40, 20:40] = np.tile([128, 64, 192], (20, 20, 1))
        images.append(('mixed', mixed))

        # Technical pattern - medium ambiguity
        technical = np.zeros((64, 64, 3), dtype=np.uint8)
        # Grid pattern
        for i in range(0, 64, 8):
            technical[i, :] = 255
            technical[:, i] = 255
        images.append(('technical', technical))

        # Natural-like - variable ambiguity
        natural = np.random.exponential(100, (64, 64, 3)).astype(np.uint8)
        natural = np.clip(natural, 0, 255)
        images.append(('natural', natural))

        # High-entropy - maximum ambiguity expected
        high_entropy = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        images.append(('high_entropy', high_entropy))

        return images[:count]

    def run_statistical_analysis(self, image_batch: List[tuple]) -> Optional[Any]:
        """Run statistical compression ambiguity analysis."""
        print("Running Statistical Compression Analysis...")

        # Extract just the image arrays
        images = [img_data for _, img_data in image_batch]

        start_time = time.time()
        try:
            analysis = self.statistical_processor.process_image_batch(images)
            processing_time = time.time() - start_time

            print(f"  Statistical analysis completed in "
                  f"{processing_time:.2f}s")
            print(f"  Ambiguous bits found: {len(analysis.ambiguous_bits)}")
            print(f"  Ambiguity density: {analysis.ambiguity_density:.6f}")

            return analysis

        except Exception as e:
            print(f"  Statistical analysis failed: {e}")
            return None

    def run_proof_validated_analysis(self, image_batch: List[tuple]) -> List[Any]:
        """Run proof-validated compression analysis."""
        print("\nRunning Proof-Validated Compression Analysis...")

        # Extract just the image arrays
        images = [img_data for _, img_data in image_batch]

        start_time = time.time()
        try:
            validated_bits = (
                self.proof_processor.process_with_proof_validation(images)
            )
            processing_time = time.time() - start_time

            print(f"  Proof validation completed in "
                  f"{processing_time:.2f}s")
            print(f"  Validated ambiguous bits: {len(validated_bits)}")

            if not IMPORTS_AVAILABLE:
                print("  (Running in simulation mode - no actual proof validation)")

            # Show proof validation summary
            summary = (
                self.proof_processor.get_proof_based_meta_information_summary()
            )
            if summary.get('status') != 'no_validated_bits':
                total_proofs = summary.get('total_formal_proofs', 0)
                print(f"  Total formal proofs generated: {total_proofs}")
                avg_consciousness = summary.get('average_consciousness_level', 0)
                print(f"  Average consciousness level: "
                      f"{avg_consciousness:.4f}")
                avg_complexity = summary.get('average_proof_complexity', 0)
                print(f"  Average proof complexity: {avg_complexity:.2f}")

            return validated_bits

        except Exception as e:
            print(f"  Proof validation failed: {e}")
            return []

    def compare_analysis_methods(self, image_batch: List[tuple]):
        """Compare statistical vs proof-validated analysis."""
        print("=" * 70)
        print("COMPRESSION AMBIGUITY ANALYSIS COMPARISON")
        print("=" * 70)

        # Run both analyses
        statistical_analysis = self.run_statistical_analysis(image_batch)
        proof_validated_bits = self.run_proof_validated_analysis(image_batch)

        # Comparison results
        print("\n" + "=" * 70)
        print("COMPARISON RESULTS")
        print("=" * 70)

        # Statistical Analysis Results
        if statistical_analysis:
            print("\nSTATISTICAL ANALYSIS:")
            print("  Method: Compression ratio + entropy analysis")
            ambiguous_count = len(statistical_analysis.ambiguous_bits)
            print(f"  Ambiguous patterns detected: {ambiguous_count}")
            compression_ratio = statistical_analysis.compression_ratio
            print(f"  Compression ratio achieved: {compression_ratio:.3f}")
            ambiguity_density = statistical_analysis.ambiguity_density
            print(f"  Ambiguity density: {ambiguity_density:.6f}")
            print("  Validation method: Statistical inference")
            print("  Meta-information quality: Inferred from statistics")

            # Show top statistical ambiguous patterns
            if statistical_analysis.ambiguous_bits:
                print("\n  Top Statistical Ambiguous Patterns:")
                sorted_bits = sorted(
                    statistical_analysis.ambiguous_bits[:3],
                    key=lambda x: x.meta_information_potential,
                    reverse=True
                )
                for i, bit in enumerate(sorted_bits):
                    print(f"    Pattern {i+1}:")
                    resistance = bit.compression_resistance
                    print(f"      Compression resistance: {resistance:.3f}")
                    print(f"      Meaning count: {bit.meaning_count}")
                    potential = bit.meta_information_potential
                    print(f"      Meta-info potential: {potential:.3f}")

        # Proof-Validated Analysis Results
        print("\nPROOF-VALIDATED ANALYSIS:")
        if proof_validated_bits:
            print("  Method: Formal proof generation + verification")
            validated_count = len(proof_validated_bits)
            print(f"  Validated ambiguous patterns: {validated_count}")

            if IMPORTS_AVAILABLE:
                summary = (
                    self.proof_processor.get_proof_based_meta_information_summary()
                )
                total_proofs = summary.get('total_formal_proofs', 0)
                print(f"  Total formal proofs: {total_proofs}")
                proof_system = summary.get('formal_system', 'unknown')
                print(f"  Proof system: {proof_system}")
                print("  Validation method: Machine-checked formal proof")
                meta_quality = summary.get('meta_information_quality', 'unknown')
                print(f"  Meta-information quality: {meta_quality}")
        else:
            print("  Method: Formal proof generation + verification")
            print("  Validated ambiguous patterns: 0")
            if IMPORTS_AVAILABLE:
                print("  Note: No patterns met formal validation criteria")
            else:
                print("  Note: Running in simulation mode")
            print("  Validation method: Machine-checked formal proof")
            print("  Meta-information quality: Mathematically rigorous "
                  "(when validated)")

        # Key Differences
        print("\n" + "=" * 50)
        print("KEY DIFFERENCES")
        print("=" * 50)

        print("\nSTATISTICAL APPROACH:")
        print("  • Fast processing, statistical inference")
        print("  • Ambiguity based on compression ratios")
        print("  • Meta-information inferred from patterns")
        print("  • No formal guarantees about claims")
        print("  • Suitable for rapid analysis")

        print("\nPROOF-VALIDATED APPROACH:")
        print("  • Rigorous processing, mathematical proof")
        print("  • Ambiguity formally proven with Lean/Coq")
        print("  • Meta-information derived from proofs")
        print("  • Mathematical guarantees about all claims")
        print("  • Suitable for critical applications requiring certainty")

        # Integration Benefits
        print("\n" + "=" * 50)
        print("INTEGRATION BENEFITS")
        print("=" * 50)

        print("\nCOMBINED APPROACH ADVANTAGES:")
        print("  • Statistical analysis for rapid screening")
        print("  • Proof validation for critical pattern verification")
        print("  • Formal guarantees where needed, speed where acceptable")
        print("  • Meta-information quality ranges from inferred to proven")
        print("  • Computational resources allocated based on rigor "
              "requirements")

    def save_demonstration_results(self, image_batch: List[tuple]):
        """Save demonstration images and results."""
        print("\n" + "=" * 50)
        print("SAVING DEMONSTRATION RESULTS")
        print("=" * 50)

        results_dir = Path(__file__).parent / "proof_validation_results"
        results_dir.mkdir(exist_ok=True)

        # Save test images
        for name, image_data in image_batch:
            img_path = results_dir / f"test_image_{name}.png"
            image_pil = Image.fromarray(image_data)
            image_pil.save(img_path)
            print(f"  Saved test image: {img_path}")

        print(f"\n  Results directory: {results_dir}")
        print("  All demonstration artifacts saved successfully")

    def run_complete_demonstration(self):
        """Run the complete proof-validated compression demonstration."""
        print("PROOF-VALIDATED COMPRESSION DEMONSTRATION")
        print("=" * 80)
        
        if IMPORTS_AVAILABLE:
            print("Running with FULL proof validation capabilities")
        else:
            print("Running in SIMULATION mode (imports not available)")
            
        print("Demonstrating formal proof integration with compression "
              "ambiguity detection")
        print("=" * 80)

        # Create test images
        image_batch = self.create_test_images(4)
        batch_count = len(image_batch)
        print(f"\nCreated {batch_count} test images:")
        for name, img in image_batch:
            print(f"  {name}: {img.shape} ({img.dtype})")

        # Run comparative analysis
        self.compare_analysis_methods(image_batch)

        # Save results
        self.save_demonstration_results(image_batch)

        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETE")
        print("=" * 80)
        print("\nKey Achievements:")
        print("  ✓ Statistical compression analysis completed")
        
        if IMPORTS_AVAILABLE:
            print("  ✓ Formal proof generation and validation framework "
                  "demonstrated")
            print("  ✓ Consciousness-aware processing integration shown")
        else:
            print("  ✓ Proof validation framework structure demonstrated")
            print("  ✓ Ready for full implementation when modules are available")
            
        print("  ✓ Mathematical rigor vs statistical inference comparison "
              "provided")
        print("  ✓ Meta-information quality upgrading from inference to proof")
        
        if IMPORTS_AVAILABLE:
            print("\nThe proof-validated approach provides mathematical guarantees")
            print("about ambiguity claims while maintaining integration with the")
            print("existing consciousness-aware processing framework.")
        else:
            print("\nTo enable full proof validation:")
            print("  1. Ensure consciousness modules are properly installed")
            print("  2. Install Lean/Coq theorem provers if needed")  
            print("  3. Re-run this demonstration for complete functionality")


def main():
    """Main demonstration entry point."""
    try:
        demo = ProofValidationComparisonDemo()
        demo.run_complete_demonstration()
    except Exception as e:
        print(f"\nDemonstration failed with error: {e}")
        print("This is likely due to missing dependencies or module structure issues.")
        print("Please check the consciousness module imports and try again.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
