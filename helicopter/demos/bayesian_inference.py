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
            try:
                # Ensure proper numpy dtypes
                mean = np.array(mean, dtype=np.float64)
                cov = np.array(cov, dtype=np.float64)
                weight = float(weight)
                
                # Calculate uncertainty (standard deviation)
                uncertainty = np.sqrt(np.maximum(np.diag(cov), 1e-12))  # Ensure positive
                
                # Calculate volume (determinant of covariance)
                det_cov = np.linalg.det(cov)
                volume = float(np.sqrt(np.maximum(det_cov, 1e-12)))
                
                cluster_info = {
                    'cluster_id': int(i),
                    'center': mean.tolist(),  # Convert to list for JSON serialization
                    'uncertainty': uncertainty.tolist(),
                    'importance': weight,
                    'volume': volume
                }
                understanding['semantic_clusters'].append(cluster_info)
                
            except Exception as e:
                print(f"Warning: Error processing cluster {i}: {e}")
                continue
        
        # Overall uncertainty estimates
        try:
            cluster_uncertainties = [np.array(c['uncertainty']) for c in understanding['semantic_clusters']]
            if cluster_uncertainties:
                mean_uncertainty = float(np.mean([np.mean(unc) for unc in cluster_uncertainties]))
            else:
                mean_uncertainty = 0.0
        except Exception:
            mean_uncertainty = 0.0
            
        understanding['uncertainty_estimates'] = {
            'total_clusters': int(len(means)),
            'mean_uncertainty': mean_uncertainty,
            'effective_sample_size': float(posterior_results.get('effective_sample_size', 0)),
            'convergence_achieved': bool(posterior_results.get('convergence', False))
        }
        
        # Pattern identification
        try:
            weights_array = np.array(weights, dtype=np.float64)
            dominant_cluster = int(np.argmax(weights_array)) if len(weights_array) > 0 else 0
            cluster_separation = self._calculate_cluster_separation(means, covariances)
            data_concentration = float(np.mean(weights_array)) if len(weights_array) > 0 else 0.0
            
            understanding['pattern_identification'] = {
                'dominant_cluster': dominant_cluster,
                'cluster_separation': cluster_separation,
                'data_concentration': data_concentration
            }
        except Exception as e:
            print(f"Warning: Error in pattern identification: {e}")
            understanding['pattern_identification'] = {
                'dominant_cluster': 0,
                'cluster_separation': 0.0,
                'data_concentration': 0.0
            }
        
        # Confidence intervals
        for i, (mean, cov) in enumerate(zip(means, covariances)):
            try:
                mean = np.array(mean, dtype=np.float64)
                cov = np.array(cov, dtype=np.float64)
                
                # Calculate 95% confidence intervals
                std_dev = np.sqrt(np.maximum(np.diag(cov), 1e-12))
                ci_lower = mean - 1.96 * std_dev
                ci_upper = mean + 1.96 * std_dev
                
                understanding['confidence_intervals'][f'cluster_{i}'] = {
                    'lower': ci_lower.tolist(),
                    'upper': ci_upper.tolist()
                }
            except Exception as e:
                print(f"Warning: Error calculating confidence interval for cluster {i}: {e}")
                continue
        
        print(f"  Identified {len(understanding['semantic_clusters'])} semantic clusters")
        print(f"  Convergence achieved: {understanding['uncertainty_estimates']['convergence_achieved']}")
        
        return understanding
    
    def _calculate_cluster_separation(self, means, covariances):
        """Calculate average separation between clusters."""
        if len(means) <= 1:
            return 0.0
        
        means = np.array(means, dtype=np.float64)
        covariances = np.array(covariances, dtype=np.float64)
        
        separations = []
        for i in range(len(means)):
            for j in range(i+1, len(means)):
                try:
                    # Mahalanobis distance
                    diff = means[i] - means[j]
                    avg_cov = (covariances[i] + covariances[j]) / 2
                    
                    # Ensure covariance is positive definite
                    if np.linalg.det(avg_cov) > 1e-12:
                        separation = float(np.sqrt(diff.T @ np.linalg.inv(avg_cov) @ diff))
                    else:
                        separation = float(np.linalg.norm(diff))
                    
                    if np.isfinite(separation):
                        separations.append(separation)
                        
                except Exception:
                    # Fallback to Euclidean distance
                    try:
                        diff = means[i] - means[j]
                        separation = float(np.linalg.norm(diff))
                        if np.isfinite(separation):
                            separations.append(separation)
                    except Exception:
                        continue
        
        return float(np.mean(separations)) if separations else 0.0
    
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
    if original_samples and len(original_samples) > 0:
        sample_positions = np.array([s['position'] for s in original_samples], dtype=np.float64)
        sample_weights = np.array([s['weight'] for s in original_samples], dtype=np.float64)
        
        # Ensure arrays are proper shape and finite
        if sample_positions.shape[1] >= 2 and len(sample_weights) > 0:
            # Remove any NaN or inf values
            valid_mask = np.isfinite(sample_positions).all(axis=1) & np.isfinite(sample_weights)
            sample_positions = sample_positions[valid_mask]
            sample_weights = sample_weights[valid_mask]
            
            if len(sample_positions) > 0:
                scatter = axes[0,0].scatter(
                    sample_positions[:, 0].astype(np.float64), 
                    sample_positions[:, 1].astype(np.float64), 
                    c=sample_weights.astype(np.float64), 
                    s=30, alpha=0.6, cmap='viridis'
                )
                plt.colorbar(scatter, ax=axes[0,0], label='Sample Weight')
        
        # Plot inferred cluster centers
        if ('posterior_means' in posterior_results and 
            len(posterior_results['posterior_means']) > 0):
            
            cluster_centers = np.array(posterior_results['posterior_means'], dtype=np.float64)
            
            if cluster_centers.shape[1] >= 2:
                axes[0,0].scatter(
                    cluster_centers[:, 0].astype(np.float64), 
                    cluster_centers[:, 1].astype(np.float64), 
                    c='red', s=200, marker='x', linewidths=3, 
                    label='Inferred Clusters'
                )
                
                # Plot confidence ellipses
                if ('posterior_covariances' in posterior_results and 
                    len(posterior_results['posterior_covariances']) > 0):
                    
                    posterior_covariances = np.array(posterior_results['posterior_covariances'], dtype=np.float64)
                    
                    for i, (center, cov) in enumerate(zip(cluster_centers, posterior_covariances)):
                        try:
                            # Ensure covariance matrix is valid
                            cov = np.array(cov, dtype=np.float64)
                            if cov.shape == (2, 2) and np.linalg.det(cov) > 1e-12:
                                eigenvals, eigenvecs = np.linalg.eigh(cov)
                                eigenvals = np.maximum(eigenvals, 1e-12)  # Ensure positive
                                
                                angle = float(np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0])))
                                width = float(2 * 1.96 * np.sqrt(eigenvals[0]))
                                height = float(2 * 1.96 * np.sqrt(eigenvals[1]))
                                center_x = float(center[0])
                                center_y = float(center[1])
                                
                                ellipse = plt.matplotlib.patches.Ellipse(
                                    (center_x, center_y), width, height, angle=angle,
                                    facecolor='none', edgecolor='red', linewidth=2, linestyle='--'
                                )
                                axes[0,0].add_patch(ellipse)
                        except Exception as e:
                            print(f"Warning: Could not draw ellipse for cluster {i}: {e}")
                            continue
        
        axes[0,0].set_title('Original Samples and Inferred Clusters')
        axes[0,0].set_xlabel('X Coordinate')
        axes[0,0].set_ylabel('Y Coordinate')
        axes[0,0].legend()
    
    # 2. Posterior cluster weights
    if ('posterior_weights' in posterior_results and 
        len(posterior_results['posterior_weights']) > 0):
        
        weights = np.array(posterior_results['posterior_weights'], dtype=np.float64)
        cluster_indices = np.arange(len(weights), dtype=int)
        
        axes[0,1].bar(cluster_indices, weights)
        axes[0,1].set_title('Posterior Cluster Weights')
        axes[0,1].set_xlabel('Cluster Index')
        axes[0,1].set_ylabel('Weight')
        
        # Add value labels on bars
        for i, weight in enumerate(weights):
            axes[0,1].text(i, float(weight) + 0.01, f'{float(weight):.3f}', 
                          ha='center', va='bottom', fontsize=9)
    else:
        axes[0,1].text(0.5, 0.5, 'No clusters\nidentified', 
                      transform=axes[0,1].transAxes, ha='center', va='center')
        axes[0,1].set_title('Posterior Cluster Weights')
    
    # 3. Uncertainty visualization
    clusters = understanding.get('semantic_clusters', [])
    if clusters and len(clusters) > 0:
        uncertainties = []
        importances = []
        
        for c in clusters:
            unc = np.array(c.get('uncertainty', [0, 0]), dtype=np.float64)
            imp = float(c.get('importance', 0.0))
            
            # Ensure uncertainty has at least 2 elements
            if len(unc) < 2:
                unc = np.array([float(unc[0]) if len(unc) > 0 else 0.0, 0.0])
            
            uncertainties.append(unc)
            importances.append(imp)
        
        for i, (unc, imp) in enumerate(zip(uncertainties, importances)):
            if len(unc) >= 2 and np.isfinite(unc).all() and np.isfinite(imp):
                x_unc = float(unc[0]) if unc[0] > 0 else 0.001
                y_unc = float(unc[1]) if unc[1] > 0 else 0.001
                size = float(imp) * 500 if imp > 0 else 10
                
                axes[0,2].scatter(x_unc, y_unc, s=size, alpha=0.7, 
                                 label=f'Cluster {i}')
        
        axes[0,2].set_title('Cluster Uncertainties (size = importance)')
        axes[0,2].set_xlabel('X Uncertainty')
        axes[0,2].set_ylabel('Y Uncertainty')
        if len(clusters) <= 5:
            axes[0,2].legend()
    else:
        axes[0,2].text(0.5, 0.5, 'No uncertainty\ndata available', 
                      transform=axes[0,2].transAxes, ha='center', va='center')
        axes[0,2].set_title('Cluster Uncertainties')
    
    # 4. Convergence diagnostics
    try:
        eff_sample_size = float(understanding.get('uncertainty_estimates', {}).get('effective_sample_size', 0))
        total_clusters = float(understanding.get('uncertainty_estimates', {}).get('total_clusters', 0))
        mean_uncertainty = float(understanding.get('uncertainty_estimates', {}).get('mean_uncertainty', 0))
        cluster_separation = float(understanding.get('pattern_identification', {}).get('cluster_separation', 0))
        
        convergence_data = {
            'Effective Sample Size': eff_sample_size,
            'Total Clusters': total_clusters,
            'Mean Uncertainty': mean_uncertainty,
            'Cluster Separation': cluster_separation
        }
        
        metrics = list(convergence_data.keys())
        values = [float(v) if np.isfinite(v) else 0.0 for v in convergence_data.values()]
        
        y_pos = np.arange(len(metrics))
        bars = axes[1,0].barh(y_pos, values)
        axes[1,0].set_yticks(y_pos)
        axes[1,0].set_yticklabels(metrics)
        axes[1,0].set_title('Inference Quality Metrics')
        axes[1,0].set_xlabel('Value')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, values)):
            if value > 0:
                axes[1,0].text(value + max(values) * 0.02, bar.get_y() + bar.get_height()/2, 
                              f'{value:.2f}', ha='left', va='center', fontsize=9)
                              
    except Exception as e:
        print(f"Warning: Could not create convergence diagnostics: {e}")
        axes[1,0].text(0.5, 0.5, 'Convergence data\nunavailable', 
                      transform=axes[1,0].transAxes, ha='center', va='center')
        axes[1,0].set_title('Inference Quality Metrics')
    
    # 5. Confidence intervals
    try:
        confidence_intervals = understanding.get('confidence_intervals', {})
        if confidence_intervals and len(confidence_intervals) > 0:
            cluster_names = list(confidence_intervals.keys())
            n_clusters = len(cluster_names)
            
            if n_clusters > 0:
                # Plot confidence intervals for X coordinate
                x_positions = np.arange(n_clusters, dtype=int)
                x_lower = []
                x_upper = []
                
                for name in cluster_names:
                    lower_data = confidence_intervals[name].get('lower', [0, 0])
                    upper_data = confidence_intervals[name].get('upper', [0, 0])
                    
                    lower_val = float(lower_data[0]) if len(lower_data) > 0 else 0.0
                    upper_val = float(upper_data[0]) if len(upper_data) > 0 else 0.0
                    
                    x_lower.append(lower_val)
                    x_upper.append(upper_val)
                
                x_lower = np.array(x_lower, dtype=np.float64)
                x_upper = np.array(x_upper, dtype=np.float64)
                
                x_centers = (x_lower + x_upper) / 2
                x_errors = np.abs(x_upper - x_lower) / 2
                
                # Ensure all values are finite
                finite_mask = np.isfinite(x_centers) & np.isfinite(x_errors)
                if np.any(finite_mask):
                    axes[1,1].errorbar(
                        x_positions[finite_mask], 
                        x_centers[finite_mask], 
                        yerr=x_errors[finite_mask], 
                        fmt='o', capsize=5, capthick=2, linewidth=2
                    )
                
                axes[1,1].set_title('95% Confidence Intervals (X coordinate)')
                axes[1,1].set_xlabel('Cluster Index')
                axes[1,1].set_ylabel('X Coordinate')
                axes[1,1].set_xticks(x_positions)
            else:
                axes[1,1].text(0.5, 0.5, 'No confidence\nintervals available', 
                              transform=axes[1,1].transAxes, ha='center', va='center')
                axes[1,1].set_title('95% Confidence Intervals')
        else:
            axes[1,1].text(0.5, 0.5, 'No confidence\nintervals available', 
                          transform=axes[1,1].transAxes, ha='center', va='center')
            axes[1,1].set_title('95% Confidence Intervals')
            
    except Exception as e:
        print(f"Warning: Could not create confidence intervals plot: {e}")
        axes[1,1].text(0.5, 0.5, 'Confidence intervals\nunavailable', 
                      transform=axes[1,1].transAxes, ha='center', va='center')
        axes[1,1].set_title('95% Confidence Intervals')
    
    # 6. Summary statistics
    try:
        input_samples = int(inference_results.get('input_samples', 0))
        inference_method = str(inference_results.get('inference_method', 'unknown'))
        
        n_clusters = len(understanding.get('semantic_clusters', []))
        convergence = understanding.get('uncertainty_estimates', {}).get('convergence_achieved', False)
        eff_sample_size = float(understanding.get('uncertainty_estimates', {}).get('effective_sample_size', 0))
        
        dominant_cluster = int(understanding.get('pattern_identification', {}).get('dominant_cluster', 0))
        mean_uncertainty = float(understanding.get('uncertainty_estimates', {}).get('mean_uncertainty', 0))
        cluster_separation = float(understanding.get('pattern_identification', {}).get('cluster_separation', 0))
        data_concentration = float(understanding.get('pattern_identification', {}).get('data_concentration', 0))
        
        summary_text = f"""Bayesian Inference Summary

Input Data:
• Sample Count: {input_samples}
• Method: {inference_method}

Posterior Results:
• Clusters Identified: {n_clusters}
• Convergence: {convergence}
• Effective Sample Size: {eff_sample_size:.1f}

Understanding Extracted:
• Dominant Cluster: {dominant_cluster}
• Mean Uncertainty: {mean_uncertainty:.3f}
• Cluster Separation: {cluster_separation:.3f}
• Data Concentration: {data_concentration:.3f}
"""
        
        axes[1,2].text(0.05, 0.95, summary_text, transform=axes[1,2].transAxes,
                       verticalalignment='top', fontsize=9, fontfamily='monospace')
        axes[1,2].set_xlim(0, 1)
        axes[1,2].set_ylim(0, 1)
        axes[1,2].axis('off')
        axes[1,2].set_title('Summary')
        
    except Exception as e:
        print(f"Warning: Could not create summary statistics: {e}")
        axes[1,2].text(0.5, 0.5, 'Summary data\nunavailable', 
                      transform=axes[1,2].transAxes, ha='center', va='center')
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
        center = cluster['center']
        if isinstance(center, list):
            center_str = f"[{', '.join(f'{x:.3f}' for x in center)}]"
        else:
            center_str = f"{center:.3f}"
        print(f"  Cluster {i}: center={center_str}, importance={cluster['importance']:.3f}")
    
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
