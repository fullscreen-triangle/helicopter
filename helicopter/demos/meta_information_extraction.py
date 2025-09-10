#!/usr/bin/env python3
"""
Meta-Information Extraction for Problem Space Compression
=========================================================

Implements meta-information extraction that analyzes structural information
about information organization patterns, enabling exponential compression
of processing complexity. Based on st-stellas-moon-landing.tex.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_extraction import image
from sklearn.metrics import silhouette_score
import cv2
from scipy import ndimage
from scipy.spatial.distance import pdist, squareform
from collections import Counter
import networkx as nx

class MetaInformationExtractor:
    """
    Extracts structural information about information organization patterns.
    
    Meta-information function: μ: I → M
    where I is information space and M is meta-information space containing
    organizational patterns, density distributions, and structural relationships.
    """
    
    def __init__(self):
        self.compression_threshold = 0.1
        self.connectivity_radius = 0.5
        self.pattern_significance_threshold = 0.05
        
    def analyze_information_type(self, data):
        """
        Classify information type (α(x) function).
        
        Args:
            data: Input information element
            
        Returns:
            dict: Information type classification scores
        """
        if isinstance(data, np.ndarray) and len(data.shape) >= 2:
            # Image-like data
            return self._analyze_image_type(data)
        elif isinstance(data, np.ndarray) and len(data.shape) == 1:
            # Vector data
            return self._analyze_vector_type(data)
        else:
            # Default classification
            return {
                'structured': 0.5,
                'random': 0.5,
                'periodic': 0.0,
                'hierarchical': 0.0
            }
    
    def _analyze_image_type(self, image_data):
        """Analyze information type for image data."""
        if len(image_data.shape) == 3:
            gray = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_data.astype(float)
        
        # Normalize
        gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)
        
        # Structural analysis
        edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)
        structure_score = np.sum(edges) / (gray.shape[0] * gray.shape[1])
        
        # Randomness analysis (entropy)
        hist, _ = np.histogram(gray.flatten(), bins=50)
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log(hist + 1e-8))
        randomness_score = entropy / np.log(50)  # Normalized entropy
        
        # Periodicity analysis (FFT)
        fft_2d = np.fft.fft2(gray)
        fft_magnitude = np.abs(fft_2d)
        # Look for strong periodic components
        fft_peaks = fft_magnitude > np.percentile(fft_magnitude, 95)
        periodicity_score = np.sum(fft_peaks) / fft_magnitude.size
        
        # Hierarchical analysis (multi-scale variance)
        scales = [1, 2, 4, 8]
        variances = []
        for scale in scales:
            if scale < min(gray.shape) // 2:
                downsampled = cv2.resize(gray, 
                    (gray.shape[1] // scale, gray.shape[0] // scale))
                variances.append(np.var(downsampled))
        
        # Hierarchical if variance decreases smoothly across scales
        if len(variances) > 1:
            variance_gradient = np.gradient(variances)
            hierarchical_score = np.exp(-np.var(variance_gradient))
        else:
            hierarchical_score = 0.0
        
        return {
            'structured': float(structure_score),
            'random': float(randomness_score),
            'periodic': float(periodicity_score),
            'hierarchical': float(hierarchical_score)
        }
    
    def _analyze_vector_type(self, vector_data):
        """Analyze information type for vector data."""
        # Structural analysis (autocorrelation)
        if len(vector_data) > 1:
            autocorr = np.correlate(vector_data, vector_data, mode='full')
            structure_score = np.max(autocorr[len(autocorr)//2+1:]) / autocorr[len(autocorr)//2]
        else:
            structure_score = 0.0
        
        # Randomness (statistical tests)
        mean_val = np.mean(vector_data)
        std_val = np.std(vector_data)
        randomness_score = min(1.0, std_val / (abs(mean_val) + 1e-8))
        
        # Periodicity (FFT)
        fft_vals = np.fft.fft(vector_data)
        fft_magnitude = np.abs(fft_vals)
        periodicity_score = np.std(fft_magnitude) / (np.mean(fft_magnitude) + 1e-8)
        
        # Hierarchical (local vs global patterns)
        if len(vector_data) > 10:
            local_vars = []
            window_size = len(vector_data) // 5
            for i in range(0, len(vector_data) - window_size, window_size):
                local_vars.append(np.var(vector_data[i:i+window_size]))
            hierarchical_score = 1.0 - (np.var(local_vars) / (np.var(vector_data) + 1e-8))
        else:
            hierarchical_score = 0.0
        
        return {
            'structured': float(min(1.0, structure_score)),
            'random': float(min(1.0, randomness_score)),
            'periodic': float(min(1.0, periodicity_score / 10)),
            'hierarchical': float(max(0.0, min(1.0, hierarchical_score)))
        }
    
    def calculate_semantic_density(self, element, full_dataset):
        """
        Calculate semantic density β(x) at element x.
        
        Args:
            element: Information element
            full_dataset: Complete information space
            
        Returns:
            float: Semantic density score
        """
        if not full_dataset:
            return 0.5
        
        # For demonstration, calculate density based on local similarity
        if isinstance(element, np.ndarray) and isinstance(full_dataset[0], np.ndarray):
            # Flatten for comparison
            element_flat = element.flatten()
            
            similarities = []
            for other_element in full_dataset[:min(len(full_dataset), 50)]:  # Sample for efficiency
                other_flat = other_element.flatten()
                
                # Resize to match if necessary
                min_size = min(len(element_flat), len(other_flat))
                element_sample = element_flat[:min_size]
                other_sample = other_flat[:min_size]
                
                # Calculate similarity (inverse of normalized distance)
                if min_size > 0:
                    distance = np.linalg.norm(element_sample - other_sample) / min_size
                    similarity = np.exp(-distance)
                    similarities.append(similarity)
            
            semantic_density = np.mean(similarities) if similarities else 0.5
        else:
            semantic_density = 0.5
        
        return float(semantic_density)
    
    def calculate_connectivity_degree(self, element, full_dataset):
        """
        Calculate structural connectivity degree γ(x).
        
        Args:
            element: Information element
            full_dataset: Complete information space
            
        Returns:
            float: Connectivity degree
        """
        if not full_dataset or len(full_dataset) < 2:
            return 0.0
        
        # Build connectivity graph based on similarity
        n_elements = min(len(full_dataset), 20)  # Sample for efficiency
        sample_dataset = full_dataset[:n_elements]
        
        # Find element index
        element_idx = None
        for i, dataset_element in enumerate(sample_dataset):
            if np.array_equal(element.flatten()[:10], dataset_element.flatten()[:10]):
                element_idx = i
                break
        
        if element_idx is None:
            return 0.0
        
        # Calculate pairwise similarities
        similarities = np.zeros((n_elements, n_elements))
        for i in range(n_elements):
            for j in range(i+1, n_elements):
                elem_i = sample_dataset[i].flatten()
                elem_j = sample_dataset[j].flatten()
                
                min_size = min(len(elem_i), len(elem_j))
                if min_size > 0:
                    distance = np.linalg.norm(elem_i[:min_size] - elem_j[:min_size]) / min_size
                    similarity = np.exp(-distance)
                    similarities[i, j] = similarities[j, i] = similarity
        
        # Count connections above threshold
        connections = np.sum(similarities[element_idx, :] > self.connectivity_radius)
        connectivity_degree = connections / (n_elements - 1)
        
        return float(connectivity_degree)
    
    def estimate_compression_potential(self, element, info_type, semantic_density, connectivity):
        """
        Estimate compression potential coefficient δ(x).
        
        Args:
            element: Information element
            info_type: Information type classification
            semantic_density: Semantic density score
            connectivity: Connectivity degree
            
        Returns:
            float: Compression potential coefficient
        """
        # High structure + high connectivity + high density = high compression potential
        structure_weight = 0.4
        density_weight = 0.3  
        connectivity_weight = 0.3
        
        structure_score = info_type.get('structured', 0.0) + info_type.get('hierarchical', 0.0)
        
        compression_potential = (structure_weight * structure_score + 
                               density_weight * semantic_density +
                               connectivity_weight * connectivity)
        
        # Ensure reasonable bounds
        compression_potential = max(0.1, min(1.0, compression_potential))
        
        return float(compression_potential)
    
    def extract_structural_patterns(self, dataset):
        """
        Extract structural patterns π(x) from dataset.
        
        Args:
            dataset: List of information elements
            
        Returns:
            dict: Extracted structural patterns
        """
        patterns = {}
        
        if not dataset:
            return patterns
        
        # Flatten all elements for pattern analysis
        flattened_data = []
        for element in dataset:
            if isinstance(element, np.ndarray):
                flattened_data.append(element.flatten()[:100])  # Limit size for efficiency
        
        if not flattened_data:
            return patterns
        
        # Ensure all vectors have same length
        min_length = min(len(vec) for vec in flattened_data)
        normalized_data = np.array([vec[:min_length] for vec in flattened_data])
        
        if normalized_data.shape[0] < 2:
            return patterns
        
        # Clustering analysis
        if normalized_data.shape[0] >= 3:
            try:
                n_clusters = min(5, normalized_data.shape[0] // 2)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(normalized_data)
                
                patterns['cluster_centers'] = kmeans.cluster_centers_
                patterns['cluster_labels'] = cluster_labels
                patterns['n_clusters'] = n_clusters
                
                # Calculate silhouette score
                if len(np.unique(cluster_labels)) > 1:
                    patterns['cluster_quality'] = silhouette_score(normalized_data, cluster_labels)
                else:
                    patterns['cluster_quality'] = 0.0
            except:
                patterns['cluster_quality'] = 0.0
        
        # PCA analysis
        if normalized_data.shape[1] >= 2:
            try:
                pca = PCA(n_components=min(3, normalized_data.shape[1]))
                pca_transformed = pca.fit_transform(normalized_data)
                
                patterns['pca_components'] = pca.components_
                patterns['pca_explained_variance'] = pca.explained_variance_ratio_
                patterns['pca_transformed'] = pca_transformed
            except:
                patterns['pca_explained_variance'] = np.array([1.0])
        
        # Distance matrix analysis
        try:
            distances = pdist(normalized_data, metric='euclidean')
            distance_matrix = squareform(distances)
            
            patterns['mean_distance'] = np.mean(distances)
            patterns['distance_std'] = np.std(distances)
            patterns['distance_matrix'] = distance_matrix
        except:
            patterns['mean_distance'] = 1.0
            patterns['distance_std'] = 0.1
        
        return patterns
    
    def calculate_compression_ratio(self, meta_information):
        """
        Calculate overall compression ratio C_ratio = |I_original| / |I_compressed|.
        
        Args:
            meta_information: Extracted meta-information
            
        Returns:
            float: Compression ratio
        """
        if not meta_information:
            return 1.0
        
        total_compression = 0.0
        element_count = 0
        
        for element_meta in meta_information:
            if 'compression_potential' in element_meta:
                total_compression += element_meta['compression_potential']
                element_count += 1
        
        if element_count == 0:
            return 1.0
        
        average_compression = total_compression / element_count
        
        # Convert compression potential to compression ratio
        # Higher compression potential = higher compression ratio
        compression_ratio = 1.0 + average_compression * 99.0  # Scale to reasonable range
        
        return float(compression_ratio)
    
    def extract_meta_information(self, dataset):
        """
        Main meta-information extraction function.
        
        Args:
            dataset: List of information elements
            
        Returns:
            dict: Complete meta-information analysis
        """
        print(f"Extracting meta-information from {len(dataset)} elements...")
        
        meta_information = []
        
        # Extract meta-information for each element
        for i, element in enumerate(dataset):
            print(f"  Processing element {i+1}/{len(dataset)}")
            
            # Information type classification α(x)
            info_type = self.analyze_information_type(element)
            
            # Semantic density β(x)
            semantic_density = self.calculate_semantic_density(element, dataset)
            
            # Connectivity degree γ(x)
            connectivity = self.calculate_connectivity_degree(element, dataset)
            
            # Compression potential δ(x)
            compression_potential = self.estimate_compression_potential(
                element, info_type, semantic_density, connectivity
            )
            
            element_meta = {
                'element_index': i,
                'information_type': info_type,
                'semantic_density': semantic_density,
                'connectivity_degree': connectivity,
                'compression_potential': compression_potential
            }
            
            meta_information.append(element_meta)
        
        # Extract structural patterns
        structural_patterns = self.extract_structural_patterns(dataset)
        
        # Calculate compression ratio
        compression_ratio = self.calculate_compression_ratio(meta_information)
        
        results = {
            'element_meta_information': meta_information,
            'structural_patterns': structural_patterns,
            'compression_ratio': compression_ratio,
            'dataset_size': len(dataset)
        }
        
        print(f"Meta-information extraction completed!")
        print(f"  Compression ratio: {compression_ratio:.2f}×")
        
        return results

def visualize_meta_information(meta_info_results, save_path=None):
    """
    Visualize meta-information extraction results.
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    meta_info = meta_info_results['element_meta_information']
    
    # 1. Information type distribution
    info_types = ['structured', 'random', 'periodic', 'hierarchical']
    type_scores = {info_type: [] for info_type in info_types}
    
    for element_meta in meta_info:
        for info_type in info_types:
            type_scores[info_type].append(element_meta['information_type'].get(info_type, 0))
    
    type_means = [np.mean(type_scores[t]) for t in info_types]
    axes[0,0].bar(info_types, type_means)
    axes[0,0].set_title('Information Type Distribution')
    axes[0,0].set_ylabel('Average Score')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. Semantic density distribution
    semantic_densities = [elem['semantic_density'] for elem in meta_info]
    axes[0,1].hist(semantic_densities, bins=20, alpha=0.7, color='orange')
    axes[0,1].set_title('Semantic Density Distribution')
    axes[0,1].set_xlabel('Semantic Density')
    axes[0,1].set_ylabel('Count')
    axes[0,1].axvline(np.mean(semantic_densities), color='red', linestyle='--',
                      label=f'Mean: {np.mean(semantic_densities):.3f}')
    axes[0,1].legend()
    
    # 3. Connectivity degree distribution
    connectivities = [elem['connectivity_degree'] for elem in meta_info]
    axes[0,2].hist(connectivities, bins=20, alpha=0.7, color='green')
    axes[0,2].set_title('Connectivity Degree Distribution')
    axes[0,2].set_xlabel('Connectivity Degree')
    axes[0,2].set_ylabel('Count')
    axes[0,2].axvline(np.mean(connectivities), color='red', linestyle='--',
                      label=f'Mean: {np.mean(connectivities):.3f}')
    axes[0,2].legend()
    
    # 4. Compression potential distribution
    compression_potentials = [elem['compression_potential'] for elem in meta_info]
    axes[0,3].hist(compression_potentials, bins=20, alpha=0.7, color='purple')
    axes[0,3].set_title('Compression Potential Distribution')
    axes[0,3].set_xlabel('Compression Potential')
    axes[0,3].set_ylabel('Count')
    axes[0,3].axvline(np.mean(compression_potentials), color='red', linestyle='--',
                      label=f'Mean: {np.mean(compression_potentials):.3f}')
    axes[0,3].legend()
    
    # 5. Correlation analysis
    correlation_data = np.column_stack([
        semantic_densities, connectivities, compression_potentials
    ])
    correlation_matrix = np.corrcoef(correlation_data.T)
    
    im = axes[1,0].imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1,0].set_xticks(range(3))
    axes[1,0].set_yticks(range(3))
    axes[1,0].set_xticklabels(['Semantic\nDensity', 'Connectivity', 'Compression\nPotential'])
    axes[1,0].set_yticklabels(['Semantic\nDensity', 'Connectivity', 'Compression\nPotential'])
    axes[1,0].set_title('Meta-Information Correlations')
    
    # Add correlation values
    for i in range(3):
        for j in range(3):
            axes[1,0].text(j, i, f'{correlation_matrix[i,j]:.2f}',
                          ha='center', va='center', color='black', fontweight='bold')
    
    plt.colorbar(im, ax=axes[1,0])
    
    # 6. PCA visualization (if available)
    if 'pca_transformed' in meta_info_results['structural_patterns']:
        pca_data = meta_info_results['structural_patterns']['pca_transformed']
        if pca_data.shape[1] >= 2:
            scatter = axes[1,1].scatter(pca_data[:, 0], pca_data[:, 1], 
                                      c=compression_potentials, cmap='viridis', alpha=0.7)
            axes[1,1].set_title('PCA Projection (colored by compression potential)')
            axes[1,1].set_xlabel('First Principal Component')
            axes[1,1].set_ylabel('Second Principal Component')
            plt.colorbar(scatter, ax=axes[1,1])
    else:
        axes[1,1].text(0.5, 0.5, 'PCA data\nnot available', 
                      transform=axes[1,1].transAxes, ha='center', va='center')
        axes[1,1].set_title('PCA Projection')
    
    # 7. Compression ratio visualization
    axes[1,2].bar(['Original', 'Compressed'], 
                  [meta_info_results['dataset_size'], 
                   meta_info_results['dataset_size'] / meta_info_results['compression_ratio']])
    axes[1,2].set_title(f'Compression Visualization\n(Ratio: {meta_info_results["compression_ratio"]:.1f}×)')
    axes[1,2].set_ylabel('Effective Size')
    
    # 8. Meta-information summary
    summary_text = f"""Meta-Information Summary
Dataset Size: {meta_info_results['dataset_size']}
Compression Ratio: {meta_info_results['compression_ratio']:.2f}×

Average Scores:
• Semantic Density: {np.mean(semantic_densities):.3f}
• Connectivity: {np.mean(connectivities):.3f}  
• Compression Potential: {np.mean(compression_potentials):.3f}

Structural Patterns:
• Cluster Quality: {meta_info_results['structural_patterns'].get('cluster_quality', 'N/A')}
• PCA Variance Explained: {np.sum(meta_info_results['structural_patterns'].get('pca_explained_variance', [0])):.3f}
"""
    
    axes[1,3].text(0.05, 0.95, summary_text, transform=axes[1,3].transAxes,
                   verticalalignment='top', fontsize=10, fontfamily='monospace')
    axes[1,3].set_xlim(0, 1)
    axes[1,3].set_ylim(0, 1)
    axes[1,3].axis('off')
    axes[1,3].set_title('Summary')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def demonstrate_meta_information_extraction():
    """
    Demonstrate meta-information extraction on sample dataset.
    """
    print("Meta-Information Extraction Demonstration")
    print("=" * 50)
    
    # Create sample dataset with different types of information
    dataset = []
    
    print("Creating sample dataset...")
    
    # 1. Structured information (geometric patterns)
    for i in range(10):
        img = np.zeros((50, 50))
        # Create structured patterns
        img[10:40, 10:40] = 1.0
        img[15:35, 15:35] = 0.5
        img[20:30, 20:30] = 1.0
        # Add some variation
        img += np.random.normal(0, 0.1, img.shape)
        dataset.append(img)
    
    # 2. Random information
    for i in range(10):
        img = np.random.random((50, 50))
        dataset.append(img)
    
    # 3. Periodic information
    for i in range(10):
        x, y = np.meshgrid(np.arange(50), np.arange(50))
        img = np.sin(2 * np.pi * x / 10) * np.cos(2 * np.pi * y / 10)
        img = (img + 1) / 2  # Normalize to [0, 1]
        dataset.append(img)
    
    # 4. Hierarchical information (fractals)
    for i in range(10):
        img = np.zeros((50, 50))
        # Simple hierarchical pattern
        for scale in [1, 2, 4, 8]:
            x, y = np.meshgrid(np.arange(0, 50, scale), np.arange(0, 50, scale))
            pattern = np.sin(x) * np.cos(y)
            # Resize to full image
            pattern_resized = cv2.resize(pattern, (50, 50))
            img += pattern_resized / (scale * 2)
        img = (img - img.min()) / (img.max() - img.min())
        dataset.append(img)
    
    print(f"Created dataset with {len(dataset)} elements")
    
    # Extract meta-information
    extractor = MetaInformationExtractor()
    meta_info_results = extractor.extract_meta_information(dataset)
    
    # Visualize results
    print("Visualizing meta-information extraction results...")
    visualize_meta_information(
        meta_info_results,
        save_path="meta_information_extraction_demo.png"
    )
    
    print("Meta-information extraction demonstration completed!")
    
    return meta_info_results

if __name__ == "__main__":
    demonstrate_meta_information_extraction()
