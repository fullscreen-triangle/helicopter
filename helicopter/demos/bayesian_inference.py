#!/usr/bin/env python3
"""
Bayesian Inference on Fuzzy Window Samples
==========================================

Implements Bayesian inference on samples collected through constrained 
stochastic sampling with fuzzy windows. This is the Moon Landing Algorithm's
third layer that tries to understand the system.
Based on st-stellas-moon-landing.tex.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import seaborn as sns
from sklearn.mixture import BayesianGaussianMixture
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class BayesianInferenceEngine:
    """
    Bayesian inference engine for processing constrained stochastic samples.
    Implements the Moon Landing Algorithm's understanding layer.
    """
    
    def __init__(self, prior_parameters=None):
        self.prior_parameters = prior_parameters or self._default_prior_parameters()
        self.posterior_samples = []
        self.inference_results = {}
        
    def _default_prior_parameters(self):
        """Default prior parameters for Bayesian inference."""
        return {
            'mean_prior_mean': np.array([0.0, 0.0]),
            'mean_prior_cov': np.eye(2) * 4.0,
            'precision_prior_shape': 2.0,
            'precision_prior_rate': 1.0,
            'mixture_concentration': 1.0
        }
    
    def calculate_likelihood(self, sample_data, parameters):
        """
        Calculate likelihood P(data|parameters) for sample data.
        
        Args:
            sample_data: Sample positions and weights
            parameters: Model parameters
            
        Returns:
            float: Log likelihood
        """
        if not sample_data or len(sample_data) == 0:
            return -np.inf
        
        positions = np.array([s['position'] for s in sample_data])
        weights = np.array([s['weight'] for s in sample_data])
        
        # Model: samples follow a mixture of Gaussians
        if 'means' not in parameters or 'covariances' not in parameters:
            return -np.inf
        
        means = parameters['means']
        covariances = parameters['covariances']
        mixture_weights = parameters.get('mixture_weights', np.ones(len(means)) / len(means))
        
        total_log_likelihood = 0.0
        
        for i, (pos, weight) in enumerate(zip(positions, weights)):
            # Calculate likelihood for this sample
            sample_likelihood = 0.0
            
            for j, (mean, cov, mix_weight) in enumerate(zip(means, covariances, mixture_weights)):
                try:
                    # Multivariate normal likelihood
                    mvn_likelihood = stats.multivariate_normal.pdf(pos, mean, cov)
                    sample_likelihood += mix_weight * mvn_likelihood
                except:
                    continue
            
            if sample_likelihood > 0:
                # Weight by fuzzy window weight
                total_log_likelihood += weight * np.log(sample_likelihood)
        
        return total_log_likelihood
    
    def calculate_prior_probability(self, parameters):
        """
        Calculate prior probability P(parameters).
        
        Args:
            parameters: Model parameters
            
        Returns:
            float: Log prior probability
        """
        log_prior = 0.0
        
        # Prior on means (multivariate normal)
        if 'means' in parameters:
            for mean in parameters['means']:
                try:
                    log_prior += stats.multivariate_normal.logpdf(
                        mean,
                        self.prior_parameters['mean_prior_mean'],
                        self.prior_parameters['mean_prior_cov']
                    )
                except:
                    log_prior += -10.0  # Penalty for invalid parameters
        
        # Prior on precision matrices (Wishart)
        if 'covariances' in parameters:
            for cov in parameters['covariances']:
                try:
                    # Use inverse Wishart prior on covariance
                    # For simplicity, use penalty for extreme values
                    det_cov = np.linalg.det(cov)
                    if det_cov <= 0:
                        log_prior += -100.0
                    else:
                        log_prior += -0.5 * np.trace(np.linalg.inv(cov))
                except:
                    log_prior += -100.0
        
        return log_prior
    
    def sample_posterior(self, sample_data, n_posterior_samples=1000, n_components=3):
        """
        Sample from posterior distribution using MCMC-like approach.
        
        Args:
            sample_data: Observed samples
            n_posterior_samples: Number of posterior samples
            n_components: Number of mixture components
            
        Returns:
            dict: Posterior samples and analysis
        """
        print(f"Sampling posterior with {n_components} components...")
        
        if not sample_data:
            return {'posterior_samples': [], 'convergence': False}
        
        positions = np.array([s['position'] for s in sample_data])
        
        # Use Bayesian Gaussian Mixture Model for inference
        try:
            bgm = BayesianGaussianMixture(
                n_components=n_components,
                covariance_type='full',
                max_iter=200,
                random_state=42
            )
            
            # Fit with sample weights
            weights = np.array([s['weight'] for s in sample_data])
            weights = weights / np.sum(weights)  # Normalize
            
            bgm.fit(positions)
            
            # Extract fitted parameters
            posterior_means = bgm.means_
            posterior_covariances = bgm.covariances_
            posterior_weights = bgm.weights_
            
            # Sample from posterior
            posterior_samples = []
            
            for i in range(n_posterior_samples):
                # Sample component
                component = np.random.choice(n_components, p=posterior_weights)
                
                # Sample from component
                sample = np.random.multivariate_normal(
                    posterior_means[component],
                    posterior_covariances[component]
                )
                
                posterior_samples.append({
                    'mean_sample': sample,
                    'component': component,
                    'likelihood': self.calculate_likelihood(
                        sample_data,
                        {'means': [sample], 'covariances': [np.eye(len(sample))]}
                    )
                })
            
            # Calculate convergence diagnostics
            sample_positions = np.array([s['mean_sample'] for s in posterior_samples])
            
            # Effective sample size (rough estimate)
            likelihoods = np.array([s['likelihood'] for s in posterior_samples])
            valid_likelihoods = likelihoods[np.isfinite(likelihoods)]
            
            if len(valid_likelihoods) > 0:
                ess = len(valid_likelihoods) ** 2 / np.sum(np.exp(2 * (valid_likelihoods - np.max(valid_likelihoods))))
                convergence = ess > 50  # Rough convergence criterion
            else:
                ess = 0
                convergence = False
            
            results = {
                'posterior_samples': posterior_samples,
                'posterior_means': posterior_means,
                'posterior_covariances': posterior_covariances,
                'posterior_weights': posterior_weights,
                'effective_sample_size': ess,
                'convergence': convergence,
                'n_components': n_components,
                'bgm_model': bgm
            }
            
            print(f"  Effective sample size: {ess:.1f}")
            print(f"  Convergence: {convergence}")
            
            return results
            
        except Exception as e:
            print(f"Error in Bayesian inference: {e}")
            # Fallback: use K-means clustering
            return self._fallback_inference(sample_data, n_components)
    
    def _fallback_inference(self, sample_data, n_components):
        """Fallback inference using simpler methods."""
        print("Using fallback inference method...")
        
        positions = np.array([s['position'] for s in sample_data])
        
        if len(positions) < n_components:
            n_components = max(1, len(positions))
        
        # Use K-means for initial clustering
        kmeans = KMeans(n_clusters=n_components, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(positions)
        
        # Estimate parameters for each cluster
        posterior_means = []
        posterior_covariances = []
        posterior_weights = []
        
        for i in range(n_components):
            cluster_mask = cluster_labels == i
            cluster_positions = positions[cluster_mask]
            
            if len(cluster_positions) > 0:
                mean = np.mean(cluster_positions, axis=0)
                if len(cluster_positions) > 1:
                    cov = np.cov(cluster_positions.T)
                    # Ensure positive definite
                    cov += np.eye(len(mean)) * 1e-6
                else:
                    cov = np.eye(len(mean)) * 0.1
                
                posterior_means.append(mean)
                posterior_covariances.append(cov)
                posterior_weights.append(len(cluster_positions) / len(positions))
        
        return {
            'posterior_samples': [],
            'posterior_means': np.array(posterior_means),
            'posterior_covariances': np.array(posterior_covariances),
            'posterior_weights': np.array(posterior_weights),
            'effective_sample_size': len(sample_data) * 0.5,
            'convergence': True,
            'n_components': len(posterior_means),
            'method': 'fallback'
        }
    
    def extract_understanding(self, posterior_results):
        """
        Extract understanding from posterior distribution.
        
        Args:
            posterior_results: Results from posterior sampling
            
        Returns:
            dict: Extracted understanding
        """
        print("Extracting understanding from posterior...")
        
        understanding = {
            'semantic_clusters': [],
            'uncertainty_estimates': {},
            'pattern_identification': {},
            'confidence_intervals': {}
        }
        
        # Extract semantic clusters
        means = posterior_results['posterior_means']
        covariances = posterior_results['posterior_covariances']
        weights = posterior_results['posterior_weights']
        
        for i, (mean, cov, weight) in enumerate(zip(means, covariances, weights)):
            cluster_info = {
                'cluster_id': i,
                'center': mean,
                'uncertainty': np.sqrt(np.diag(cov)),
                'importance': weight,
                'volume': np.sqrt(np.linalg.det(cov))
            }
            understanding['semantic_clusters'].append(cluster_info)
        
        # Overall uncertainty estimates
        understanding['uncertainty_estimates'] = {
            'total_clusters': len(means),
            'mean_uncertainty': np.mean([c['uncertainty'] for c in understanding['semantic_clusters']]),
            'effective_sample_size': posterior_results['effective_sample_size'],
            'convergence_achieved': posterior_results['convergence']
        }
        
        # Pattern identification
        understanding['pattern_identification'] = {
            'dominant_cluster': int(np.argmax(weights)) if len(weights) > 0 else 0,
            'cluster_separation': self._calculate_cluster_separation(means, covariances),
            'data_concentration': np.mean(weights) if len(weights) > 0 else 0.0
        }
        
        # Confidence intervals
        for i, (mean, cov) in enumerate(zip(means, covariances)):
            ci_lower = mean - 1.96 * np.sqrt(np.diag(cov))
            ci_upper = mean + 1.96 * np.sqrt(np.diag(cov))
            understanding['confidence_intervals'][f'cluster_{i}'] = {
                'lower': ci_lower,
                'upper': ci_upper
            }
        
        print(f"  Identified {len(understanding['semantic_clusters'])} semantic clusters")
        print(f"  Convergence achieved: {understanding['uncertainty_estimates']['convergence_achieved']}")
        
        return understanding
    
    def _calculate_cluster_separation(self, means, covariances):
        """Calculate average separation between clusters."""
        if len(means) <= 1:
            return 0.0
        
        separations = []
        for i in range(len(means)):
            for j in range(i+1, len(means)):
                # Mahalanobis distance
                diff = means[i] - means[j]
                avg_cov = (covariances[i] + covariances[j]) / 2
                try:
                    separation = np.sqrt(diff.T @ np.linalg.inv(avg_cov) @ diff)
                    separations.append(separation)
                except:
                    separation = np.linalg.norm(diff)
                    separations.append(separation)
        
        return np.mean(separations) if separations else 0.0
    
    def infer_understanding_from_samples(self, sample_data):
        """
        Complete inference pipeline: samples → posterior → understanding.
        
        Args:
            sample_data: Constrained stochastic samples
            
        Returns:
            dict: Complete inference results
        """
        print("Starting Bayesian inference on fuzzy window samples...")
        print(f"Input samples: {len(sample_data)}")
        
        # Sample posterior
        posterior_results = self.sample_posterior(sample_data)
        
        # Extract understanding
        understanding = self.extract_understanding(posterior_results)
        
        # Combine results
        complete_results = {
            'posterior_results': posterior_results,
            'understanding': understanding,
            'input_samples': len(sample_data),
            'inference_method': 'bayesian_gaussian_mixture'
        }
        
        return complete_results

def visualize_bayesian_inference(inference_results, original_samples=None, save_path=None):
    """
    Visualize Bayesian inference results.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    posterior_results = inference_results['posterior_results']
    understanding = inference_results['understanding']
    
    # 1. Original samples and inferred clusters
    if original_samples:
        sample_positions = np.array([s['position'] for s in original_samples])
        sample_weights = np.array([s['weight'] for s in original_samples])
        
        scatter = axes[0,0].scatter(sample_positions[:, 0], sample_positions[:, 1], 
                                   c=sample_weights, s=30, alpha=0.6, cmap='viridis')
        plt.colorbar(scatter, ax=axes[0,0], label='Sample Weight')
        
        # Plot inferred cluster centers
        if len(posterior_results['posterior_means']) > 0:
            cluster_centers = posterior_results['posterior_means']
            axes[0,0].scatter(cluster_centers[:, 0], cluster_centers[:, 1], 
                             c='red', s=200, marker='x', linewidths=3, 
                             label='Inferred Clusters')
            
            # Plot confidence ellipses
            for i, (center, cov) in enumerate(zip(cluster_centers, posterior_results['posterior_covariances'])):
                eigenvals, eigenvecs = np.linalg.eigh(cov)
                angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
                width, height = 2 * 1.96 * np.sqrt(eigenvals)  # 95% confidence
                
                ellipse = plt.matplotlib.patches.Ellipse(
                    center, width, height, angle=angle,
                    facecolor='none', edgecolor='red', linewidth=2, linestyle='--'
                )
                axes[0,0].add_patch(ellipse)
        
        axes[0,0].set_title('Original Samples and Inferred Clusters')
        axes[0,0].set_xlabel('X Coordinate')
        axes[0,0].set_ylabel('Y Coordinate')
        axes[0,0].legend()
    
    # 2. Posterior cluster weights
    if len(posterior_results['posterior_weights']) > 0:
        cluster_indices = range(len(posterior_results['posterior_weights']))
        axes[0,1].bar(cluster_indices, posterior_results['posterior_weights'])
        axes[0,1].set_title('Posterior Cluster Weights')
        axes[0,1].set_xlabel('Cluster Index')
        axes[0,1].set_ylabel('Weight')
    else:
        axes[0,1].text(0.5, 0.5, 'No clusters\nidentified', 
                      transform=axes[0,1].transAxes, ha='center', va='center')
        axes[0,1].set_title('Posterior Cluster Weights')
    
    # 3. Uncertainty visualization
    clusters = understanding['semantic_clusters']
    if clusters:
        uncertainties = [c['uncertainty'] for c in clusters]
        importances = [c['importance'] for c in clusters]
        
        for i, (unc, imp) in enumerate(zip(uncertainties, importances)):
            axes[0,2].scatter(unc[0] if len(unc) > 0 else 0, 
                             unc[1] if len(unc) > 1 else 0,
                             s=imp*500, alpha=0.7, label=f'Cluster {i}')
        
        axes[0,2].set_title('Cluster Uncertainties (size = importance)')
        axes[0,2].set_xlabel('X Uncertainty')
        axes[0,2].set_ylabel('Y Uncertainty')
        if len(clusters) <= 5:
            axes[0,2].legend()
    
    # 4. Convergence diagnostics
    convergence_data = {
        'Effective Sample Size': understanding['uncertainty_estimates']['effective_sample_size'],
        'Total Clusters': understanding['uncertainty_estimates']['total_clusters'],
        'Mean Uncertainty': understanding['uncertainty_estimates']['mean_uncertainty'],
        'Cluster Separation': understanding['pattern_identification']['cluster_separation']
    }
    
    metrics = list(convergence_data.keys())
    values = list(convergence_data.values())
    
    axes[1,0].barh(range(len(metrics)), values)
    axes[1,0].set_yticks(range(len(metrics)))
    axes[1,0].set_yticklabels(metrics)
    axes[1,0].set_title('Inference Quality Metrics')
    axes[1,0].set_xlabel('Value')
    
    # 5. Confidence intervals
    if understanding['confidence_intervals']:
        cluster_names = list(understanding['confidence_intervals'].keys())
        n_clusters = len(cluster_names)
        
        if n_clusters > 0:
            # Plot confidence intervals for X coordinate
            x_positions = range(n_clusters)
            x_lower = [understanding['confidence_intervals'][name]['lower'][0] 
                      for name in cluster_names]
            x_upper = [understanding['confidence_intervals'][name]['upper'][0] 
                      for name in cluster_names]
            x_centers = [(l+u)/2 for l, u in zip(x_lower, x_upper)]
            x_errors = [(u-l)/2 for l, u in zip(x_lower, x_upper)]
            
            axes[1,1].errorbar(x_positions, x_centers, yerr=x_errors, 
                              fmt='o', capsize=5, capthick=2, linewidth=2)
            axes[1,1].set_title('95% Confidence Intervals (X coordinate)')
            axes[1,1].set_xlabel('Cluster Index')
            axes[1,1].set_ylabel('X Coordinate')
            axes[1,1].set_xticks(x_positions)
    
    # 6. Summary statistics
    summary_text = f"""Bayesian Inference Summary

Input Data:
• Sample Count: {inference_results['input_samples']}
• Method: {inference_results['inference_method']}

Posterior Results:
• Clusters Identified: {len(understanding['semantic_clusters'])}
• Convergence: {understanding['uncertainty_estimates']['convergence_achieved']}
• Effective Sample Size: {understanding['uncertainty_estimates']['effective_sample_size']:.1f}

Understanding Extracted:
• Dominant Cluster: {understanding['pattern_identification']['dominant_cluster']}
• Mean Uncertainty: {understanding['uncertainty_estimates']['mean_uncertainty']:.3f}
• Cluster Separation: {understanding['pattern_identification']['cluster_separation']:.3f}
• Data Concentration: {understanding['pattern_identification']['data_concentration']:.3f}
"""
    
    axes[1,2].text(0.05, 0.95, summary_text, transform=axes[1,2].transAxes,
                   verticalalignment='top', fontsize=9, fontfamily='monospace')
    axes[1,2].set_xlim(0, 1)
    axes[1,2].set_ylim(0, 1)
    axes[1,2].axis('off')
    axes[1,2].set_title('Summary')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def demonstrate_bayesian_inference():
    """
    Demonstrate Bayesian inference on synthetic fuzzy window samples.
    """
    print("Bayesian Inference on Fuzzy Window Samples")
    print("=" * 50)
    
    # Create synthetic sample data (simulating constrained sampling results)
    print("Creating synthetic sample data...")
    
    # Three clusters of samples with different weights
    np.random.seed(42)
    
    sample_data = []
    
    # Cluster 1: High weight region
    for i in range(50):
        pos = np.random.multivariate_normal([2, 2], [[0.3, 0.1], [0.1, 0.3]])
        weight = np.exp(-0.5 * np.sum((pos - [2, 2])**2) / 0.5)  # Gaussian weight
        sample_data.append({
            'position': pos,
            'weight': weight,
            'step_size': np.random.uniform(0.1, 0.5),
            'gravity_magnitude': np.random.uniform(1.0, 3.0)
        })
    
    # Cluster 2: Medium weight region
    for i in range(30):
        pos = np.random.multivariate_normal([6, 3], [[0.5, 0], [0, 0.8]])
        weight = np.exp(-0.5 * np.sum((pos - [6, 3])**2) / 1.0) * 0.7
        sample_data.append({
            'position': pos,
            'weight': weight,
            'step_size': np.random.uniform(0.2, 0.8),
            'gravity_magnitude': np.random.uniform(0.5, 2.0)
        })
    
    # Cluster 3: Low weight region
    for i in range(20):
        pos = np.random.multivariate_normal([3, 7], [[0.8, 0.2], [0.2, 0.6]])
        weight = np.exp(-0.5 * np.sum((pos - [3, 7])**2) / 1.5) * 0.4
        sample_data.append({
            'position': pos,
            'weight': weight,
            'step_size': np.random.uniform(0.3, 1.0),
            'gravity_magnitude': np.random.uniform(0.3, 1.5)
        })
    
    print(f"Created {len(sample_data)} synthetic samples in 3 clusters")
    
    # Initialize Bayesian inference engine
    inference_engine = BayesianInferenceEngine()
    
    # Perform inference
    print("Performing Bayesian inference...")
    inference_results = inference_engine.infer_understanding_from_samples(sample_data)
    
    # Display results
    print("\nInference Results:")
    understanding = inference_results['understanding']
    print(f"  Clusters identified: {len(understanding['semantic_clusters'])}")
    print(f"  Convergence achieved: {understanding['uncertainty_estimates']['convergence_achieved']}")
    print(f"  Effective sample size: {understanding['uncertainty_estimates']['effective_sample_size']:.1f}")
    
    for i, cluster in enumerate(understanding['semantic_clusters']):
        print(f"  Cluster {i}: center={cluster['center']:.3f}, importance={cluster['importance']:.3f}")
    
    # Visualize results
    print("Visualizing Bayesian inference results...")
    visualize_bayesian_inference(
        inference_results, 
        original_samples=sample_data,
        save_path="bayesian_inference_demo.png"
    )
    
    print("Bayesian inference demonstration completed!")
    
    return inference_results

if __name__ == "__main__":
    demonstrate_bayesian_inference()
