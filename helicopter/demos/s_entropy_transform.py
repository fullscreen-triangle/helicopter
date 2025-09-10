#!/usr/bin/env python3
"""
S-Entropy Coordinate Transformation for Images
==============================================

Transforms visual data to S-entropy coordinates using semantic cardinal directions.
Based on the mathematical framework from st-stellas-dictionary.tex.
"""

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.image import extract_patches_2d
import seaborn as sns

class SEntropyCoordinateTransformer:
    """
    Transforms images to 4D S-entropy coordinate space using semantic cardinal directions.
    
    Coordinate System:
    - Technical/Precision (North): (1, 0, 0, 0)  
    - Emotional/Expression (South): (-1, 0, 0, 0)
    - Action/Process (East): (0, 1, 0, 0)
    - Descriptive/Attribute (West): (0, -1, 0, 0) 
    - Abstract/Conceptual (Up): (0, 0, 1, 0)
    - Concrete/Physical (Down): (0, 0, -1, 0)
    - Positive/Affirmation (Forward): (0, 0, 0, 1)
    - Negative/Negation (Backward): (0, 0, 0, -1)
    """
    
    def __init__(self):
        # Cardinal direction basis vectors
        self.cardinal_directions = {
            'technical': np.array([1.0, 0.0, 0.0, 0.0]),      # North
            'emotional': np.array([-1.0, 0.0, 0.0, 0.0]),     # South  
            'action': np.array([0.0, 1.0, 0.0, 0.0]),         # East
            'descriptive': np.array([0.0, -1.0, 0.0, 0.0]),   # West
            'abstract': np.array([0.0, 0.0, 1.0, 0.0]),       # Up
            'concrete': np.array([0.0, 0.0, -1.0, 0.0]),      # Down
            'positive': np.array([0.0, 0.0, 0.0, 1.0]),       # Forward
            'negative': np.array([0.0, 0.0, 0.0, -1.0])       # Backward
        }
    
    def analyze_image_semantics(self, image):
        """
        Analyze image to extract semantic properties for coordinate mapping.
        
        Args:
            image: Input image as numpy array (H, W, 3)
            
        Returns:
            dict: Semantic analysis results
        """
        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) if len(image.shape) == 3 else np.zeros((image.shape[0], image.shape[1], 3))
        
        # Technical/Precision analysis (edges, structures)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
        
        # Detect lines (technical precision indicator)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=5)
        line_count = len(lines) if lines is not None else 0
        technical_score = min(1.0, (edge_density * 10 + line_count / 100))
        
        # Emotional analysis (color warmth, saturation)
        if len(image.shape) == 3:
            # Warm colors (reds, oranges, yellows) vs cool colors (blues, greens)
            warm_mask = (hsv[:,:,0] < 30) | (hsv[:,:,0] > 150)  # Red-orange-yellow hues
            warm_ratio = np.sum(warm_mask) / (hsv.shape[0] * hsv.shape[1])
            
            # High saturation indicates emotional content
            saturation_mean = np.mean(hsv[:,:,1])
            emotional_score = (warm_ratio + saturation_mean / 255.0) / 2
        else:
            emotional_score = 0.1  # Low emotion for grayscale
        
        # Action analysis (motion blur, dynamic patterns)
        # Detect motion blur using variance of Laplacian
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Detect diagonal edges (movement indicators)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
        action_score = min(1.0, np.mean(sobel_combined) / 100)
        
        # Descriptive analysis (texture complexity, detail density)
        # Local Binary Pattern-like analysis
        patches = extract_patches_2d(gray, (8, 8), max_patches=100, random_state=42)
        patch_variances = [np.var(patch) for patch in patches]
        texture_complexity = np.mean(patch_variances)
        descriptive_score = min(1.0, texture_complexity / 1000)
        
        # Abstract analysis (symmetry, geometric patterns)
        # Detect symmetry
        height, width = gray.shape
        left_half = gray[:, :width//2]
        right_half = np.fliplr(gray[:, width//2:])
        min_width = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_width]
        right_half = right_half[:, :min_width]
        
        if left_half.shape == right_half.shape:
            symmetry_score = 1.0 - np.mean(np.abs(left_half.astype(float) - right_half.astype(float))) / 255.0
        else:
            symmetry_score = 0.0
        
        # Detect circular/geometric patterns
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=10, maxRadius=100)
        circle_count = len(circles[0]) if circles is not None else 0
        abstract_score = min(1.0, (symmetry_score + circle_count / 10) / 2)
        
        # Concrete analysis (recognizable objects, natural textures)
        # High local contrast indicates concrete objects
        local_contrast = np.std(gray)
        
        # Object-like blob detection
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 100
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(gray)
        blob_count = len(keypoints)
        
        concrete_score = min(1.0, (local_contrast / 100 + blob_count / 20) / 2)
        
        # Positive/Negative analysis (brightness, contrast)
        brightness = np.mean(gray)
        brightness_normalized = brightness / 255.0
        
        # High contrast and brightness = positive
        contrast = np.std(gray)
        positive_score = (brightness_normalized + contrast / 255.0) / 2
        negative_score = 1.0 - positive_score
        
        return {
            'technical': technical_score,
            'emotional': emotional_score, 
            'action': action_score,
            'descriptive': descriptive_score,
            'abstract': abstract_score,
            'concrete': concrete_score,
            'positive': positive_score,
            'negative': negative_score
        }
    
    def transform_to_coordinates(self, image):
        """
        Transform image to S-entropy coordinates.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            numpy.ndarray: 4D S-entropy coordinates
        """
        # Analyze semantic properties
        semantic_scores = self.analyze_image_semantics(image)
        
        # Calculate weighted sum of cardinal directions
        s_coordinate = np.zeros(4)
        
        for direction_name, direction_vector in self.cardinal_directions.items():
            weight = semantic_scores[direction_name]
            s_coordinate += weight * direction_vector
        
        # Normalize to unit vector to maintain coordinate space properties
        magnitude = np.linalg.norm(s_coordinate)
        if magnitude > 0:
            s_coordinate = s_coordinate / magnitude
        
        return s_coordinate, semantic_scores
    
    def visualize_coordinate_transformation(self, image, s_coordinate, semantic_scores, save_path=None):
        """
        Visualize the coordinate transformation results.
        
        Args:
            image: Original image
            s_coordinate: Computed S-entropy coordinate
            semantic_scores: Semantic analysis scores
            save_path: Optional path to save visualization
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0,0].imshow(image)
        axes[0,0].set_title('Original Image')
        axes[0,0].axis('off')
        
        # Semantic scores radar chart
        categories = list(semantic_scores.keys())
        values = list(semantic_scores.values())
        
        # Create radar chart
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # Complete the circle
        angles += angles[:1]
        
        ax_radar = plt.subplot(2, 3, 2, projection='polar')
        ax_radar.plot(angles, values, 'o-', linewidth=2, label='Semantic Scores')
        ax_radar.fill(angles, values, alpha=0.25)
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(categories)
        ax_radar.set_ylim(0, 1)
        ax_radar.set_title('Semantic Analysis Scores')
        
        # S-entropy coordinates visualization
        axes[0,2].bar(range(4), s_coordinate)
        axes[0,2].set_xticks(range(4))
        axes[0,2].set_xticklabels(['North-South\n(Tech-Emotion)', 'East-West\n(Action-Desc)', 
                                   'Up-Down\n(Abstract-Concrete)', 'Forward-Back\n(Pos-Neg)'])
        axes[0,2].set_title('S-Entropy Coordinates')
        axes[0,2].set_ylabel('Coordinate Value')
        
        # 3D projection of coordinates (first 3 dimensions)
        ax_3d = fig.add_subplot(2, 3, 4, projection='3d')
        ax_3d.scatter([s_coordinate[0]], [s_coordinate[1]], [s_coordinate[2]], 
                     s=100, c='red', label='Image Coordinate')
        ax_3d.set_xlabel('North-South (Technical-Emotional)')
        ax_3d.set_ylabel('East-West (Action-Descriptive)')
        ax_3d.set_zlabel('Up-Down (Abstract-Concrete)')
        ax_3d.set_title('3D S-Entropy Projection')
        
        # Coordinate magnitude and direction
        magnitude = np.linalg.norm(s_coordinate)
        axes[1,1].pie([magnitude, 1-magnitude], labels=['Coordinate Magnitude', 'Remaining'], 
                     autopct='%1.3f', startangle=90)
        axes[1,1].set_title(f'Coordinate Magnitude: {magnitude:.4f}')
        
        # Distance from cardinal directions
        distances = {}
        for name, direction in self.cardinal_directions.items():
            distance = np.linalg.norm(s_coordinate - direction)
            distances[name] = distance
        
        axes[1,2].bar(range(len(distances)), list(distances.values()))
        axes[1,2].set_xticks(range(len(distances)))
        axes[1,2].set_xticklabels(list(distances.keys()), rotation=45)
        axes[1,2].set_title('Distance from Cardinal Directions')
        axes[1,2].set_ylabel('Euclidean Distance')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig

def demonstrate_s_entropy_transformation():
    """
    Demonstrate S-entropy coordinate transformation on sample images.
    """
    # Initialize transformer
    transformer = SEntropyCoordinateTransformer()
    
    print("S-Entropy Coordinate Transformation Demonstration")
    print("=" * 50)
    
    # Test with different types of images
    test_cases = [
        ("Technical Image", "technical_circuit.jpg"),
        ("Natural Image", "nature_scene.jpg"), 
        ("Emotional Image", "emotional_face.jpg")
    ]
    
    for case_name, filename in test_cases:
        print(f"\nProcessing: {case_name}")
        
        try:
            # For demo, create synthetic test images if files don't exist
            if case_name == "Technical Image":
                # Create a technical-looking image with lines and geometric shapes
                img = np.zeros((200, 200, 3), dtype=np.uint8)
                cv2.rectangle(img, (50, 50), (150, 150), (255, 255, 255), 2)
                cv2.line(img, (0, 100), (200, 100), (255, 255, 255), 2)
                cv2.line(img, (100, 0), (100, 200), (255, 255, 255), 2)
                cv2.circle(img, (100, 100), 30, (255, 255, 255), 2)
                
            elif case_name == "Natural Image":
                # Create a natural-looking image with organic shapes
                img = np.random.randint(50, 200, (200, 200, 3), dtype=np.uint8)
                # Add some circular "organic" patterns
                cv2.circle(img, (70, 70), 40, (100, 200, 100), -1)
                cv2.circle(img, (130, 130), 35, (150, 180, 120), -1)
                cv2.circle(img, (60, 140), 25, (120, 190, 140), -1)
                
            else:  # Emotional Image
                # Create a warm, high-saturation image
                img = np.full((200, 200, 3), [200, 100, 50], dtype=np.uint8)
                # Add some warm gradients
                for i in range(200):
                    for j in range(200):
                        dist = np.sqrt((i-100)**2 + (j-100)**2)
                        intensity = max(0, 1 - dist/100)
                        img[i, j] = [int(200 * intensity + 50), 
                                   int(150 * intensity + 50), 
                                   int(100 * intensity + 50)]
            
            # Transform to S-entropy coordinates
            s_coordinate, semantic_scores = transformer.transform_to_coordinates(img)
            
            print(f"S-Entropy Coordinate: {s_coordinate}")
            print("Semantic Scores:")
            for key, value in semantic_scores.items():
                print(f"  {key.capitalize()}: {value:.4f}")
            
            # Visualize results
            transformer.visualize_coordinate_transformation(
                img, s_coordinate, semantic_scores,
                save_path=f"s_entropy_demo_{case_name.lower().replace(' ', '_')}.png"
            )
            
        except Exception as e:
            print(f"Error processing {case_name}: {e}")
    
    print("\nS-Entropy transformation demonstration completed!")

if __name__ == "__main__":
    demonstrate_s_entropy_transformation()
