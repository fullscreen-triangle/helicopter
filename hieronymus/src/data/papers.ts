export interface Paper {
  slug: string;
  number: number;
  title: string;
  subtitle: string;
  lines: number;
  theorems: number;
  refs: number;
  panels: number;
  results: number;
  keyResult: string;
  abstract: string;
  keyTheorems: string[];
  validation: Record<string, string>;
  panelNames: string[];
}

export const papers: Paper[] = [
  {
    slug: 'measurement-modalities-stereogram',
    number: 1,
    title: 'Measurement-Modality Stereograms',
    subtitle:
      'Dual-Path Pixel Validation Through Optical and Oxygen-Mediated Categorical Observation',
    lines: 1572,
    theorems: 53,
    refs: 56,
    panels: 5,
    results: 3,
    keyResult:
      'Every pixel is a dual object — visible (optical) + invisible (O\u2082 categorical). Consistency between the two paths provides cross-validation without ground truth.',
    abstract:
      'Every pixel in a biological microscopy image carries an implicit duality: the value recorded by the camera sensor through external photon collection (the visible pixel) and the value that the intracellular oxygen distribution at that same spatial location would encode through molecular state transitions (the invisible pixel). We develop a rigorous mathematical framework — measurement-modality stereograms — that formalises this duality from two foundational axioms: bounded phase space and categorical observation. From these axioms we derive hierarchical partition coordinates (n, l, m, s) for spherical phase space, a conserved entropy coordinate system (S_k, S_t, S_e), and a commutation theorem establishing that categorical observables commute with physical observables. We prove that molecular oxygen admits exactly three categorical states — absorption, ground, and emission — forming a complete ternary basis. Both visible and invisible pixels independently encode the same partition signature, enabling dual-pixel cross-validation.',
    keyTheorems: [
      'Dual-Pixel Consistency Theorem: visible and invisible partition signatures must agree for reliable measurements',
      'Information Gain Theorem: the fused dual pixel carries strictly greater information than either single-modality pixel',
      'Resolution Enhancement Theorem: molecular oxygen detectors at ~0.1 nm break the optical diffraction limit at ~200 nm',
      'S-Entropy Conservation: S_k + S_t + S_e = 1 holds independently for both modalities',
    ],
    validation: {
      dice: '0.785',
      conservation: '1.000000',
      MI: '4.47 bits',
      consistency: '89.0%',
    },
    panelNames: [
      'Segmentation Performance',
      'Entropy Conservation',
      'Information Theory',
      'Consistency Analysis',
      'Ternary Resolution',
    ],
  },
  {
    slug: 'image-harmonic-coupling',
    number: 2,
    title: 'Image Harmonic Matching Circuits',
    subtitle:
      'Oscillatory Pixel Dynamics and Interference-Based Visual Comparison Without Algorithmic Computation',
    lines: 1393,
    theorems: 46,
    refs: 47,
    panels: 5,
    results: 10,
    keyResult:
      'Image comparison IS interference, not computation. Constructive interference signals a match, destructive interference signals a mismatch — O(1) wall-clock time.',
    abstract:
      'We establish a framework in which image comparison is performed by physical interference rather than algorithmic computation. Starting from two axioms — bounded phase space for persistent dynamical systems and finite-resolution categorical observation — we prove by elimination that oscillatory dynamics is the unique valid mode for bounded persistent systems. This Oscillatory Necessity Theorem implies that the fundamental constituents of any physical representation are oscillators. Each pixel in a digital image is mapped to an information gas molecule inheriting thermodynamic and oscillatory properties — frequency, phase, and a partition signature. Image comparison then reduces to superposition of pixel wavefunctions: constructive interference signals a match and destructive interference signals a mismatch, with no sequential instruction execution required. Harmonically coupled pixel-oscillations form networks whose closed loops constitute matching circuits — wallless resonant cavities where the round-trip phase condition distinguishes sustained resonance from radiative decay.',
    keyTheorems: [
      'Oscillatory Necessity Theorem: oscillatory dynamics is the unique valid persistent mode in bounded phase space',
      'Triple Equivalence: oscillation, categorical enumeration, and partition function evaluation yield identical entropy S = k_B M ln n',
      'Von Neumann Bottleneck Elimination: O(n^2 d) algorithmic comparison replaced by O(1) wall-clock interference',
      'Matching Circuit Theorem: closed harmonic loops with round-trip phase 2pi*n sustain resonance (verified match)',
    ],
    validation: {
      matchAccuracy: '0.892',
      noiseRobust: '0.847 at SNR=5dB',
      segDice: '0.791',
      conservation: '1.000000',
    },
    panelNames: [
      'Matching Accuracy',
      'Noise Robustness',
      'Segmentation',
      'Harmonic Network',
      'Entropy Triple',
    ],
  },
  {
    slug: 'universal-spectral-matching',
    number: 3,
    title: 'Universal Spectral Matching',
    subtitle:
      'Reducing All Comparison to Computer Vision Through Oscillatory Representation and GPU-Parallel Interference',
    lines: 2525,
    theorems: 42,
    refs: 72,
    panels: 5,
    results: 13,
    keyResult:
      'ALL comparison reduces to computer vision on GPU. Every bounded system has a spectral image, and comparing systems means comparing images via interference.',
    abstract:
      'We prove from first principles that every comparison problem — matching molecules, signals, structures, sequences, or arbitrary data — reduces to a computer vision problem. The argument proceeds through a chain of mathematical identities, not analogies. First, the Oscillatory Necessity Theorem establishes that every persistent system in bounded phase space oscillates. Second, Koopman operator theory guarantees a complete spectral decomposition into frequency-amplitude-phase triples. Third, the Spectral Image Theorem proves this spectrum is isomorphic to a two-dimensional image with frequency mapped to horizontal position, phase to vertical position, and amplitude to pixel intensity. Fourth, the Universal Reduction Theorem establishes that categorical distance between any two bounded systems equals the L2 image distance between their spectral images. GPU fragment shaders implement massively parallel per-pixel interference, comparing two spectral images in a single render pass with O(1) wall-clock time on commodity hardware.',
    keyTheorems: [
      'Spectral Image Theorem: every bounded oscillatory system maps injectively to a 2D image preserving metric structure',
      'Universal Reduction Theorem: d(X,Y) = d_CV(I_X, I_Y) — categorical distance equals computer vision distance',
      'Five-Pass GPU Pipeline: encode, partition, interfere, entropy, display — complete comparison in one frame',
      'Domain Encoder Universality: microscopy, molecular spectra, chromatography, time series, text, genomics, graphs',
    ],
    validation: {
      selfConsistency: 'd(X,X) < 1e-6',
      triangleInequality: '100% satisfied',
      crossDomain: '0.934 AUC',
      throughput: '847 pairs/sec',
    },
    panelNames: [
      'Spectral Image Construction',
      'Cross-Domain Matching',
      'Interference Similarity',
      'S-Entropy Conservation',
      'Throughput Network',
    ],
  },
  {
    slug: 'gpu-observation-architecture',
    number: 4,
    title: 'Fragment Shader as Observation Apparatus',
    subtitle:
      'O(1)-Memory Universal Computation Through the Rendering-Measurement Identity',
    lines: 3268,
    theorems: 52,
    refs: 55,
    panels: 5,
    results: 16,
    keyResult:
      'Rendering = measurement. The fragment shader IS a physical observation apparatus. O(1) memory, ~13 MB working set, GPU-supervised training without human labels.',
    abstract:
      'We prove from first principles that a GPU fragment shader is not a visualization tool but a physical observation apparatus: when it writes a pixel value it performs a measurement, and the rendered texture is the computed result in categorical representation, not a picture of it. The argument rests on three pillars. First, the Oscillatory Necessity Theorem: every persistent dynamical system in bounded phase space necessarily oscillates. Second, the Triple Equivalence Theorem: oscillatory, categorical, and partitional descriptions yield identical state counts and entropies, connected by explicit bijective maps. Third, the Rendering-Measurement Identity: for a fragment shader implementing partition observation, rendering a texture is identical to measuring the categorical state. From these we derive: an O(1) GPU-memory streaming protocol reducing database search to ~13 MB working set independent of database size; a GPU-supervised training framework where physical observables serve as training signals without human labels; and proof that integrated GPUs with ~25 MB working set are sufficient for the complete pipeline.',
    keyTheorems: [
      'Rendering-Measurement Identity: rendering a texture IS performing a measurement — mathematical identity, not analogy',
      'O(1) Memory Theorem: streaming observation with constant ~13 MB working set regardless of database size',
      'GPU-Supervised Training: partition sharpness, phase coherence, interference visibility as label-free training signals',
      'Integrated GPU Sufficiency: Intel UHD / AMD Radeon / Apple M-series at ~1-2 TFLOPS sufficient for complete pipeline',
    ],
    validation: {
      memoryScaling: 'O(1) verified to 10M items',
      renderMeasureGap: '< 1e-7',
      observableCorr: '> 0.95 with ground truth',
      throughput: '1,240 obs/sec on integrated GPU',
    },
    panelNames: [
      'Rendering-Measurement Identity',
      'Memory Scaling',
      'Physical Observables',
      'GPU-Supervised Training',
      'Throughput by Hardware',
    ],
  },
  {
    slug: 'ray-tracing-cellular-computing',
    number: 5,
    title: 'Ray-Tracing as Cellular Computation',
    subtitle:
      'Simultaneous Optical, Chromatographic, and Circuit Observation Through Volumetric Partition Traversal',
    lines: 3104,
    theorems: 18,
    refs: 64,
    panels: 5,
    results: 19,
    keyResult:
      'A single ray march simultaneously computes optical absorption, chromatographic retention, and circuit current flow — three observations from one traversal.',
    abstract:
      'We prove from first principles that a single ray marching through cellular cytoplasm simultaneously computes three physically distinct observations at every integration step: optical absorption (light transport via Beer-Lambert attenuation), chromatographic retention (molecular interaction via categorical distance), and circuit current flow (biochemical reaction network via Kirchhoff\'s laws). The Triple Observation Identity establishes these three computations are not independent processes but a single partition observation expressed in three equivalent representations. The cytoplasm functions as a three-dimensional chromatographic column whose local retention properties are encoded by partition coordinates (n, l, m, s) at each spatial position. The intracellular reaction network is isomorphic to an electrical circuit whose node potentials are chemical potentials and whose conductances are reaction rate constants. Interference between multiple rays yields a visibility parameter identical to the cellular coherence index: high visibility indicates healthy cells, low visibility indicates pathology.',
    keyTheorems: [
      'Triple Observation Identity: optical absorption, chromatographic retention, and circuit current are one partition observation in three representations',
      'Cellular Chromatography Theorem: cytoplasm is a 3D chromatographic column with partition coordinates at each voxel',
      'Circuit-Reaction Isomorphism: biochemical network maps to electrical circuit — potentials are chemical potentials, conductances are rate constants',
      'Coherence Diagnostic: ray interference visibility = cellular health index — synchronized oscillators indicate viability',
    ],
    validation: {
      tripleConsistency: 'r > 0.99 across all three observations',
      holographicError: 'RMSE < 0.02',
      coherenceSensitivity: '0.96 AUC for viability',
      flowRecovery: 'r = 0.94 vs ground truth',
    },
    panelNames: [
      'Triple Consistency',
      'Holographic Reconstruction',
      'Coherence Diagnostic',
      'Flow Recovery',
      'Throughput & Conservation',
    ],
  },
];

export const totalStats = {
  papers: papers.length,
  lines: papers.reduce((s, p) => s + p.lines, 0),
  theorems: papers.reduce((s, p) => s + p.theorems, 0),
  refs: papers.reduce((s, p) => s + p.refs, 0),
  panels: papers.reduce((s, p) => s + p.panels, 0),
  results: papers.reduce((s, p) => s + p.results, 0),
};

export function getPaperBySlug(slug: string): Paper | undefined {
  return papers.find((p) => p.slug === slug);
}
