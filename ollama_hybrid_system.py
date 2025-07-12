#!/usr/bin/env python3
"""
Ollama + Custom Parser Hybrid System for Image RAG
Combines local AI models with custom analysis for cost-effective, high-quality results
"""

import json
import time
import hashlib
import requests
import numpy as np
from PIL import Image, ImageStat, ImageFilter
from PIL.ExifTags import TAGS
import cv2
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
import os
from io import BytesIO
import colorsys
from sklearn.cluster import KMeans
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomImageParser:
    """
    Advanced custom parser for technical image analysis
    Provides detailed analysis without external API costs
    """
    
    def __init__(self):
        self.quality_weights = {
            'sharpness': 0.25,
            'contrast': 0.20,
            'brightness': 0.15,
            'noise': 0.15,
            'composition': 0.15,
            'color_harmony': 0.10
        }
    
    def analyze_image(self, image_data: bytes, filename: str = None) -> Dict[str, Any]:
        """Complete technical analysis of the image"""
        
        try:
            # Convert to PIL and OpenCV formats
            pil_image = Image.open(BytesIO(image_data))
            cv_image = self._pil_to_cv2(pil_image)
            
            # Basic image information
            basic_info = self._extract_basic_info(pil_image, len(image_data))
            
            # Technical quality metrics
            quality_metrics = self._analyze_quality(pil_image, cv_image)
            
            # Color analysis
            color_analysis = self._analyze_colors(pil_image, cv_image)
            
            # Composition analysis
            composition = self._analyze_composition(cv_image)
            
            # Content classification
            content_type = self._classify_content(cv_image, basic_info)
            
            # Feature extraction
            features = self._extract_features(cv_image)
            
            # Generate technical tags
            technical_tags = self._generate_technical_tags(
                quality_metrics, color_analysis, composition, content_type
            )
            
            # Calculate overall quality score
            overall_score = self._calculate_overall_quality(quality_metrics, composition, color_analysis)
            
            return {
                'basic_info': basic_info,
                'quality_metrics': quality_metrics,
                'color_analysis': color_analysis,
                'composition': composition,
                'content_type': content_type,
                'features': features,
                'technical_tags': technical_tags,
                'overall_quality_score': overall_score,
                'processing_time': time.time(),
                'parser_version': '1.0.0'
            }
            
        except Exception as e:
            logger.error(f"Parser analysis failed: {str(e)}")
            return self._get_fallback_analysis()
    
    def _pil_to_cv2(self, pil_image: Image.Image) -> np.ndarray:
        """Convert PIL image to OpenCV format"""
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    def _extract_basic_info(self, image: Image.Image, file_size: int) -> Dict[str, Any]:
        """Extract basic image information"""
        
        width, height = image.size
        
        # EXIF data
        exif_data = {}
        try:
            exif = image._getexif()
            if exif:
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    exif_data[tag] = str(value)
        except:
            pass
        
        return {
            'width': width,
            'height': height,
            'aspect_ratio': round(width / height, 2),
            'megapixels': round((width * height) / 1_000_000, 1),
            'file_size_mb': round(file_size / (1024 * 1024), 2),
            'format': image.format or 'Unknown',
            'mode': image.mode,
            'has_transparency': image.mode in ('RGBA', 'LA') or 'transparency' in image.info,
            'exif_data': exif_data
        }
    
    def _analyze_quality(self, pil_image: Image.Image, cv_image: np.ndarray) -> Dict[str, float]:
        """Analyze technical quality metrics"""
        
        # Sharpness (Laplacian variance)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(100, (sharpness / 1000) * 100)
        
        # Contrast (RMS contrast)
        contrast = gray.std()
        contrast_score = min(100, (contrast / 64) * 100)
        
        # Brightness analysis
        brightness = gray.mean()
        # Optimal brightness is around 128 (middle gray)
        brightness_score = 100 - abs(brightness - 128) / 128 * 100
        
        # Noise estimation (using high-frequency content)
        noise = self._estimate_noise(gray)
        noise_score = max(0, 100 - noise * 10)
        
        # Dynamic range
        dynamic_range = gray.max() - gray.min()
        dynamic_range_score = (dynamic_range / 255) * 100
        
        return {
            'sharpness': round(sharpness_score, 1),
            'contrast': round(contrast_score, 1),
            'brightness': round(brightness_score, 1),
            'noise_level': round(100 - noise_score, 1),
            'dynamic_range': round(dynamic_range_score, 1),
            'raw_sharpness': round(sharpness, 2),
            'raw_contrast': round(contrast, 2),
            'raw_brightness': round(brightness, 2)
        }
    
    def _estimate_noise(self, gray_image: np.ndarray) -> float:
        """Estimate noise level using high-frequency analysis"""
        # Apply Gaussian blur and subtract from original
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        noise_map = cv2.absdiff(gray_image, blurred)
        return noise_map.mean() / 255.0
    
    def _analyze_colors(self, pil_image: Image.Image, cv_image: np.ndarray) -> Dict[str, Any]:
        """Comprehensive color analysis"""
        
        # Convert to RGB for analysis
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Dominant colors using K-means
        dominant_colors = self._extract_dominant_colors(rgb_image, n_colors=5)
        
        # Color temperature estimation
        color_temp = self._estimate_color_temperature(rgb_image)
        
        # Saturation analysis
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        avg_saturation = hsv_image[:, :, 1].mean()
        
        # Color harmony analysis
        harmony_score = self._analyze_color_harmony(dominant_colors)
        
        # Color distribution
        color_distribution = self._analyze_color_distribution(rgb_image)
        
        return {
            'dominant_colors': dominant_colors,
            'color_temperature': color_temp,
            'avg_saturation': round(avg_saturation / 255 * 100, 1),
            'color_harmony_score': round(harmony_score, 1),
            'color_distribution': color_distribution,
            'total_unique_colors': len(np.unique(rgb_image.reshape(-1, 3), axis=0))
        }
    
    def _extract_dominant_colors(self, rgb_image: np.ndarray, n_colors: int = 5) -> List[Dict[str, Any]]:
        """Extract dominant colors using K-means clustering"""
        
        # Reshape image to list of pixels
        pixels = rgb_image.reshape(-1, 3)
        
        # Sample pixels for performance (max 10000 pixels)
        if len(pixels) > 10000:
            indices = np.random.choice(len(pixels), 10000, replace=False)
            pixels = pixels[indices]
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        colors = []
        labels = kmeans.labels_
        
        for i, color in enumerate(kmeans.cluster_centers_):
            percentage = (labels == i).sum() / len(labels) * 100
            
            r, g, b = color.astype(int)
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            
            # Convert to HSV for additional info
            h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
            
            colors.append({
                'rgb': [int(r), int(g), int(b)],
                'hex': hex_color,
                'percentage': round(percentage, 1),
                'hue': round(h * 360, 1),
                'saturation': round(s * 100, 1),
                'value': round(v * 100, 1),
                'color_name': self._get_color_name(r, g, b)
            })
        
        # Sort by percentage
        colors.sort(key=lambda x: x['percentage'], reverse=True)
        return colors
    
    def _get_color_name(self, r: int, g: int, b: int) -> str:
        """Get approximate color name"""
        
        # Simple color naming based on RGB values
        if r > 200 and g > 200 and b > 200:
            return "Light"
        elif r < 50 and g < 50 and b < 50:
            return "Dark"
        elif r > g and r > b:
            if g > 100:
                return "Orange" if g > b else "Red"
            else:
                return "Red"
        elif g > r and g > b:
            return "Green"
        elif b > r and b > g:
            return "Blue"
        elif r > 150 and g > 150 and b < 100:
            return "Yellow"
        elif r > 150 and g < 100 and b > 150:
            return "Purple"
        elif r < 100 and g > 150 and b > 150:
            return "Cyan"
        else:
            return "Mixed"
    
    def _estimate_color_temperature(self, rgb_image: np.ndarray) -> Dict[str, Any]:
        """Estimate color temperature of the image"""
        
        # Calculate average RGB values
        avg_r = rgb_image[:, :, 0].mean()
        avg_g = rgb_image[:, :, 1].mean()
        avg_b = rgb_image[:, :, 2].mean()
        
        # Simple color temperature estimation
        if avg_b > avg_r:
            temp_category = "Cool"
            temp_kelvin = 6500 + (avg_b - avg_r) * 10
        else:
            temp_category = "Warm"
            temp_kelvin = 3200 + (avg_r - avg_b) * 10
        
        temp_kelvin = max(2000, min(10000, temp_kelvin))
        
        return {
            'category': temp_category,
            'estimated_kelvin': round(temp_kelvin),
            'warmth_score': round((avg_r - avg_b) / 255 * 100, 1)
        }
    
    def _analyze_color_harmony(self, dominant_colors: List[Dict]) -> float:
        """Analyze color harmony using color theory"""
        
        if len(dominant_colors) < 2:
            return 50.0
        
        harmony_score = 0
        total_comparisons = 0
        
        for i, color1 in enumerate(dominant_colors[:3]):
            for color2 in dominant_colors[i+1:4]:
                # Calculate hue difference
                hue_diff = abs(color1['hue'] - color2['hue'])
                hue_diff = min(hue_diff, 360 - hue_diff)  # Circular distance
                
                # Check for harmonious relationships
                if hue_diff < 30:  # Analogous
                    harmony_score += 80
                elif 150 < hue_diff < 210:  # Complementary
                    harmony_score += 90
                elif 110 < hue_diff < 130 or 230 < hue_diff < 250:  # Triadic
                    harmony_score += 85
                else:
                    harmony_score += 40
                
                total_comparisons += 1
        
        return harmony_score / total_comparisons if total_comparisons > 0 else 50.0
    
    def _analyze_color_distribution(self, rgb_image: np.ndarray) -> Dict[str, float]:
        """Analyze color distribution across the image"""
        
        # Convert to HSV for better analysis
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        
        # Analyze hue distribution
        hue_hist = cv2.calcHist([hsv_image], [0], None, [180], [0, 180])
        hue_entropy = self._calculate_entropy(hue_hist)
        
        # Analyze saturation distribution
        sat_hist = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])
        sat_entropy = self._calculate_entropy(sat_hist)
        
        return {
            'hue_diversity': round(hue_entropy / 7.5 * 100, 1),  # Normalized to 0-100
            'saturation_diversity': round(sat_entropy / 8 * 100, 1),
            'color_richness': round((hue_entropy + sat_entropy) / 15.5 * 100, 1)
        }
    
    def _calculate_entropy(self, histogram: np.ndarray) -> float:
        """Calculate entropy of a histogram"""
        histogram = histogram.flatten()
        histogram = histogram[histogram > 0]  # Remove zeros
        probabilities = histogram / histogram.sum()
        return -np.sum(probabilities * np.log2(probabilities))
    
    def _analyze_composition(self, cv_image: np.ndarray) -> Dict[str, Any]:
        """Analyze image composition"""
        
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Rule of thirds analysis
        thirds_score = self._analyze_rule_of_thirds(gray)
        
        # Edge density and distribution
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width)
        
        # Symmetry analysis
        symmetry_score = self._analyze_symmetry(gray)
        
        # Leading lines detection
        lines_score = self._detect_leading_lines(edges)
        
        # Balance analysis
        balance_score = self._analyze_balance(gray)
        
        return {
            'rule_of_thirds_score': round(thirds_score, 1),
            'edge_density': round(edge_density * 100, 2),
            'symmetry_score': round(symmetry_score, 1),
            'leading_lines_score': round(lines_score, 1),
            'balance_score': round(balance_score, 1),
            'composition_score': round((thirds_score + symmetry_score + lines_score + balance_score) / 4, 1)
        }
    
    def _analyze_rule_of_thirds(self, gray_image: np.ndarray) -> float:
        """Analyze adherence to rule of thirds"""
        
        height, width = gray_image.shape
        
        # Define thirds lines
        h_third1, h_third2 = height // 3, 2 * height // 3
        v_third1, v_third2 = width // 3, 2 * width // 3
        
        # Calculate interest points (high gradient areas)
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Find intersection points of thirds
        intersections = [
            (h_third1, v_third1), (h_third1, v_third2),
            (h_third2, v_third1), (h_third2, v_third2)
        ]
        
        # Calculate score based on interest near intersection points
        score = 0
        for h, w in intersections:
            region = gradient_magnitude[max(0, h-20):min(height, h+20), 
                                      max(0, w-20):min(width, w+20)]
            score += region.mean()
        
        # Normalize score
        return min(100, score / 4)
    
    def _analyze_symmetry(self, gray_image: np.ndarray) -> float:
        """Analyze image symmetry"""
        
        height, width = gray_image.shape
        
        # Vertical symmetry
        left_half = gray_image[:, :width//2]
        right_half = np.fliplr(gray_image[:, width//2:])
        
        # Resize to match if needed
        min_width = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_width]
        right_half = right_half[:, :min_width]
        
        vertical_symmetry = 100 - np.mean(np.abs(left_half - right_half)) / 255 * 100
        
        # Horizontal symmetry
        top_half = gray_image[:height//2, :]
        bottom_half = np.flipud(gray_image[height//2:, :])
        
        min_height = min(top_half.shape[0], bottom_half.shape[0])
        top_half = top_half[:min_height, :]
        bottom_half = bottom_half[:min_height, :]
        
        horizontal_symmetry = 100 - np.mean(np.abs(top_half - bottom_half)) / 255 * 100
        
        return max(vertical_symmetry, horizontal_symmetry)
    
    def _detect_leading_lines(self, edges: np.ndarray) -> float:
        """Detect leading lines in the image"""
        
        # Use Hough line transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is None:
            return 0
        
        # Analyze line directions and strength
        line_strength = len(lines)
        
        # Prefer diagonal lines (more dynamic)
        diagonal_lines = 0
        for line in lines:
            rho, theta = line[0]
            angle = theta * 180 / np.pi
            if 30 < angle < 60 or 120 < angle < 150:
                diagonal_lines += 1
        
        diagonal_ratio = diagonal_lines / len(lines) if len(lines) > 0 else 0
        
        # Score based on line presence and diagonal preference
        score = min(100, line_strength * 2) * (0.7 + 0.3 * diagonal_ratio)
        return score
    
    def _analyze_balance(self, gray_image: np.ndarray) -> float:
        """Analyze visual balance of the image"""
        
        height, width = gray_image.shape
        center_x, center_y = width // 2, height // 2
        
        # Calculate moments
        moments = cv2.moments(gray_image)
        
        if moments['m00'] == 0:
            return 50  # Neutral score for empty image
        
        # Center of mass
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        
        # Distance from geometric center
        distance = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        # Balance score (closer to center = better balance)
        balance_score = 100 - (distance / max_distance) * 100
        
        return max(0, balance_score)
    
    def _classify_content(self, cv_image: np.ndarray, basic_info: Dict) -> Dict[str, Any]:
        """Classify image content type"""
        
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        aspect_ratio = basic_info['aspect_ratio']
        
        # Initialize scores
        scores = {
            'portrait': 0,
            'landscape': 0,
            'architecture': 0,
            'food': 0,
            'product': 0,
            'nature': 0,
            'abstract': 0
        }
        
        # Aspect ratio hints
        if aspect_ratio < 0.8:
            scores['portrait'] += 30
        elif aspect_ratio > 1.5:
            scores['landscape'] += 30
            scores['architecture'] += 20
        else:
            scores['food'] += 20
            scores['product'] += 20
        
        # Edge analysis
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width)
        
        if edge_density > 0.1:
            scores['architecture'] += 25
            scores['product'] += 15
        elif edge_density < 0.05:
            scores['portrait'] += 20
            scores['nature'] += 15
        
        # Texture analysis
        texture_score = self._analyze_texture(gray)
        if texture_score > 50:
            scores['nature'] += 20
            scores['abstract'] += 15
        
        # Color complexity
        color_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        unique_colors = len(np.unique(color_image.reshape(-1, 3), axis=0))
        color_complexity = min(100, unique_colors / 1000 * 100)
        
        if color_complexity > 70:
            scores['nature'] += 15
            scores['abstract'] += 10
        elif color_complexity < 30:
            scores['product'] += 15
            scores['portrait'] += 10
        
        # Determine primary category
        primary_category = max(scores, key=scores.get)
        confidence = scores[primary_category] / 100
        
        return {
            'primary_category': primary_category,
            'confidence': round(confidence, 2),
            'scores': scores,
            'characteristics': self._get_content_characteristics(primary_category)
        }
    
    def _analyze_texture(self, gray_image: np.ndarray) -> float:
        """Analyze image texture complexity"""
        
        # Local Binary Pattern approximation
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        texture_response = cv2.filter2D(gray_image, cv2.CV_64F, kernel)
        texture_variance = np.var(texture_response)
        
        # Normalize to 0-100 scale
        return min(100, texture_variance / 1000 * 100)
    
    def _get_content_characteristics(self, category: str) -> List[str]:
        """Get characteristics for content category"""
        
        characteristics_map = {
            'portrait': ['human_subject', 'facial_features', 'shallow_depth'],
            'landscape': ['natural_scenery', 'horizon_line', 'wide_view'],
            'architecture': ['geometric_shapes', 'structural_elements', 'urban'],
            'food': ['organic_shapes', 'close_up', 'appetizing'],
            'product': ['clean_background', 'commercial', 'isolated_subject'],
            'nature': ['organic_textures', 'natural_colors', 'outdoor'],
            'abstract': ['artistic', 'non_representational', 'creative']
        }
        
        return characteristics_map.get(category, ['general'])
    
    def _extract_features(self, cv_image: np.ndarray) -> Dict[str, Any]:
        """Extract various image features"""
        
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # SIFT keypoints (simplified)
        try:
            sift = cv2.SIFT_create()
            keypoints = sift.detect(gray, None)
            sift_count = len(keypoints)
        except:
            sift_count = 0
        
        # Corner detection
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
        corner_count = len(corners) if corners is not None else 0
        
        # Contour analysis
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_count = len(contours)
        
        return {
            'sift_keypoints': sift_count,
            'corner_features': corner_count,
            'contour_count': contour_count,
            'feature_density': round((sift_count + corner_count) / (gray.shape[0] * gray.shape[1]) * 10000, 2)
        }
    
    def _generate_technical_tags(self, quality: Dict, colors: Dict, composition: Dict, content: Dict) -> List[str]:
        """Generate technical tags based on analysis"""
        
        tags = []
        
        # Quality-based tags
        if quality['sharpness'] > 80:
            tags.append('sharp')
        if quality['contrast'] > 70:
            tags.append('high_contrast')
        if quality['brightness'] > 60 and quality['brightness'] < 80:
            tags.append('well_exposed')
        
        # Color-based tags
        if colors['avg_saturation'] > 70:
            tags.append('vibrant')
        elif colors['avg_saturation'] < 30:
            tags.append('muted')
        
        if colors['color_temperature']['category'] == 'Warm':
            tags.append('warm_tones')
        else:
            tags.append('cool_tones')
        
        # Composition-based tags
        if composition['rule_of_thirds_score'] > 60:
            tags.append('well_composed')
        if composition['symmetry_score'] > 70:
            tags.append('symmetric')
        if composition['leading_lines_score'] > 50:
            tags.append('dynamic')
        
        # Content-based tags
        tags.extend(content['characteristics'])
        tags.append(content['primary_category'])
        
        return list(set(tags))  # Remove duplicates
    
    def _calculate_overall_quality(self, quality: Dict, composition: Dict, colors: Dict) -> float:
        """Calculate overall quality score"""
        
        # Weighted average of different aspects
        quality_score = (
            quality['sharpness'] * self.quality_weights['sharpness'] +
            quality['contrast'] * self.quality_weights['contrast'] +
            quality['brightness'] * self.quality_weights['brightness'] +
            (100 - quality['noise_level']) * self.quality_weights['noise'] +
            composition['composition_score'] * self.quality_weights['composition'] +
            colors['color_harmony_score'] * self.quality_weights['color_harmony']
        )
        
        return round(quality_score, 1)
    
    def _get_fallback_analysis(self) -> Dict[str, Any]:
        """Fallback analysis for error cases"""
        return {
            'basic_info': {'width': 0, 'height': 0, 'error': 'Analysis failed'},
            'quality_metrics': {'sharpness': 50, 'contrast': 50, 'brightness': 50, 'noise_level': 50},
            'color_analysis': {'dominant_colors': [], 'color_temperature': {'category': 'Unknown'}},
            'composition': {'composition_score': 50},
            'content_type': {'primary_category': 'unknown', 'confidence': 0},
            'features': {'sift_keypoints': 0},
            'technical_tags': ['unknown'],
            'overall_quality_score': 50,
            'error': True
        }


class OllamaDescriptionGenerator:
    """
    Ollama-based description generator for cost-effective AI descriptions
    """
    
    def __init__(self, model_name: str = "llava:7b", ollama_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.available = self._check_ollama_availability()
    
    def _check_ollama_availability(self) -> bool:
        """Check if Ollama is available and model is installed"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return any(model['name'].startswith(self.model_name.split(':')[0]) for model in models)
            return False
        except:
            return False
    
    def generate_description(self, image_data: bytes, analysis_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AI description using Ollama"""
        
        if not self.available:
            return self._get_fallback_description(analysis_context)
        
        try:
            # Convert image to base64
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            # Create context-aware prompt
            prompt = self._create_context_prompt(analysis_context)
            
            # Call Ollama API
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 200
                }
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                description = result.get('response', '').strip()
                
                # Post-process description
                description = self._post_process_description(description, analysis_context)
                
                return {
                    'description': description,
                    'model_used': self.model_name,
                    'processing_time': result.get('total_duration', 0) / 1e9,  # Convert to seconds
                    'context_enhanced': True,
                    'source': 'ollama'
                }
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return self._get_fallback_description(analysis_context)
                
        except Exception as e:
            logger.error(f"Ollama description generation failed: {str(e)}")
            return self._get_fallback_description(analysis_context)
    
    def _create_context_prompt(self, analysis: Dict[str, Any]) -> str:
        """Create context-aware prompt based on technical analysis"""
        
        content_type = analysis.get('content_type', {}).get('primary_category', 'general')
        quality_score = analysis.get('overall_quality_score', 50)
        colors = analysis.get('color_analysis', {})
        composition = analysis.get('composition', {})
        
        # Base prompt
        prompt = "Describe this image in detail, focusing on:"
        
        # Add context based on analysis
        if content_type == 'portrait':
            prompt += " the subject's expression, lighting, and professional quality."
        elif content_type == 'landscape':
            prompt += " the natural scenery, lighting conditions, and atmospheric mood."
        elif content_type == 'architecture':
            prompt += " the architectural style, structural elements, and design features."
        elif content_type == 'food':
            prompt += " the food presentation, colors, textures, and appetizing qualities."
        elif content_type == 'product':
            prompt += " the product features, presentation, and commercial appeal."
        else:
            prompt += " the main subjects, composition, and visual impact."
        
        # Add quality context
        if quality_score > 80:
            prompt += " Emphasize the high technical quality and professional appearance."
        elif quality_score < 60:
            prompt += " Note any technical aspects while focusing on the content."
        
        # Add color context
        if colors.get('avg_saturation', 0) > 70:
            prompt += " Mention the vibrant and rich colors."
        
        prompt += " Write in a professional, engaging style suitable for marketing or portfolio use."
        
        return prompt
    
    def _post_process_description(self, description: str, analysis: Dict[str, Any]) -> str:
        """Post-process and enhance the generated description"""
        
        # Remove common AI artifacts
        description = description.replace("This image shows", "").strip()
        description = description.replace("The image depicts", "").strip()
        description = description.replace("I can see", "").strip()
        
        # Ensure it starts with a capital letter
        if description and not description[0].isupper():
            description = description[0].upper() + description[1:]
        
        # Add technical quality note if high quality
        quality_score = analysis.get('overall_quality_score', 50)
        if quality_score > 85 and 'professional' not in description.lower():
            description += " The image demonstrates professional quality with excellent technical execution."
        
        return description
    
    def _get_fallback_description(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback description when Ollama is not available"""
        
        content_type = analysis.get('content_type', {}).get('primary_category', 'general')
        quality_score = analysis.get('overall_quality_score', 50)
        
        # Template-based descriptions
        templates = {
            'portrait': "A professional portrait featuring excellent composition and lighting. The image demonstrates strong technical quality with careful attention to detail and professional presentation.",
            'landscape': "A captivating landscape image showcasing natural beauty with excellent composition. The scene captures the essence of the environment with professional quality and artistic vision.",
            'architecture': "An architectural photograph highlighting structural design and geometric elements. The image demonstrates professional composition with attention to lines, forms, and spatial relationships.",
            'food': "A beautifully presented food photograph with appealing composition and lighting. The image showcases culinary artistry with professional styling and appetizing presentation.",
            'product': "A clean, professional product photograph with excellent presentation. The image demonstrates commercial quality with careful attention to lighting and composition.",
            'nature': "A nature photograph capturing organic beauty with excellent technical quality. The image showcases natural elements with professional composition and artistic vision.",
            'abstract': "An artistic image with creative composition and visual impact. The image demonstrates artistic vision with professional execution and engaging visual elements."
        }
        
        base_description = templates.get(content_type, templates['portrait'])
        
        # Customize based on quality
        if quality_score > 85:
            base_description = base_description.replace("excellent", "exceptional")
            base_description += " The superior technical quality makes this image suitable for professional use and high-end applications."
        elif quality_score < 60:
            base_description = base_description.replace("excellent", "good")
        
        return {
            'description': base_description,
            'model_used': 'template_fallback',
            'processing_time': 0.1,
            'context_enhanced': True,
            'source': 'fallback'
        }


class HybridImageAnalyzer:
    """
    Main hybrid analyzer combining custom parser with Ollama descriptions
    """
    
    def __init__(self, ollama_model: str = "llava:7b"):
        self.parser = CustomImageParser()
        self.ollama = OllamaDescriptionGenerator(ollama_model)
        self.processing_stats = {
            'total_processed': 0,
            'ollama_used': 0,
            'fallback_used': 0,
            'avg_processing_time': 0
        }
    
    def analyze_image_complete(self, image_data: bytes, filename: str = None) -> Dict[str, Any]:
        """Complete image analysis using hybrid approach"""
        
        start_time = time.time()
        
        # Step 1: Technical analysis with custom parser
        logger.info("Starting technical analysis...")
        technical_analysis = self.parser.analyze_image(image_data, filename)
        
        # Step 2: AI description with Ollama
        logger.info("Generating AI description...")
        ai_description = self.ollama.generate_description(image_data, technical_analysis)
        
        # Step 3: Generate smart tags
        smart_tags = self._generate_smart_tags(technical_analysis, ai_description)
        
        # Step 4: Create use case suggestions
        use_cases = self._suggest_use_cases(technical_analysis, ai_description)
        
        # Step 5: Calculate final scores
        final_scores = self._calculate_final_scores(technical_analysis)
        
        processing_time = time.time() - start_time
        
        # Update stats
        self._update_stats(processing_time, ai_description['source'])
        
        # Combine all results
        result = {
            'image_info': technical_analysis['basic_info'],
            'quality_analysis': {
                'overall_score': technical_analysis['overall_quality_score'],
                'metrics': technical_analysis['quality_metrics'],
                'breakdown': final_scores
            },
            'ai_description': ai_description['description'],
            'smart_tags': smart_tags,
            'color_analysis': technical_analysis['color_analysis'],
            'composition_analysis': technical_analysis['composition'],
            'content_classification': technical_analysis['content_type'],
            'features': technical_analysis['features'],
            'use_cases': use_cases,
            'processing_metadata': {
                'total_time': round(processing_time, 2),
                'parser_version': technical_analysis.get('parser_version', '1.0.0'),
                'ai_model': ai_description['model_used'],
                'ai_source': ai_description['source'],
                'timestamp': datetime.now().isoformat()
            },
            'cost_analysis': self._calculate_cost_analysis(ai_description['source'])
        }
        
        return result
    
    def _generate_smart_tags(self, technical: Dict[str, Any], ai_desc: Dict[str, Any]) -> List[str]:
        """Generate smart tags combining technical and AI analysis"""
        
        tags = set()
        
        # Add technical tags
        tags.update(technical.get('technical_tags', []))
        
        # Add content-based tags
        content_type = technical.get('content_type', {})
        tags.add(content_type.get('primary_category', 'general'))
        
        # Add quality-based tags
        quality_score = technical.get('overall_quality_score', 50)
        if quality_score > 85:
            tags.update(['high_quality', 'professional'])
        elif quality_score > 70:
            tags.add('good_quality')
        
        # Add color-based tags
        colors = technical.get('color_analysis', {})
        if colors.get('avg_saturation', 0) > 70:
            tags.add('vibrant')
        if colors.get('color_temperature', {}).get('category') == 'Warm':
            tags.add('warm')
        else:
            tags.add('cool')
        
        # Add composition tags
        composition = technical.get('composition', {})
        if composition.get('rule_of_thirds_score', 0) > 60:
            tags.add('well_composed')
        if composition.get('symmetry_score', 0) > 70:
            tags.add('balanced')
        
        # Extract keywords from AI description
        if ai_desc.get('description'):
            description_words = ai_desc['description'].lower().split()
            # Add relevant descriptive words
            relevant_words = [
                'beautiful', 'stunning', 'dramatic', 'peaceful', 'vibrant',
                'professional', 'artistic', 'creative', 'modern', 'classic',
                'natural', 'urban', 'colorful', 'minimalist', 'detailed'
            ]
            for word in relevant_words:
                if word in description_words:
                    tags.add(word)
        
        return sorted(list(tags))
    
    def _suggest_use_cases(self, technical: Dict[str, Any], ai_desc: Dict[str, Any]) -> List[str]:
        """Suggest use cases based on analysis"""
        
        use_cases = []
        
        content_type = technical.get('content_type', {}).get('primary_category', 'general')
        quality_score = technical.get('overall_quality_score', 50)
        
        # Base use cases by content type
        use_case_map = {
            'portrait': ['LinkedIn profile', 'Business website', 'Professional headshot', 'About page'],
            'landscape': ['Website header', 'Travel blog', 'Desktop wallpaper', 'Print artwork'],
            'architecture': ['Real estate listing', 'Architecture portfolio', 'Urban planning', 'Design inspiration'],
            'food': ['Restaurant menu', 'Food blog', 'Social media', 'Recipe website'],
            'product': ['E-commerce listing', 'Product catalog', 'Marketing material', 'Advertisement'],
            'nature': ['Environmental blog', 'Nature website', 'Educational material', 'Wellness brand'],
            'abstract': ['Art gallery', 'Creative portfolio', 'Brand identity', 'Artistic expression']
        }
        
        use_cases.extend(use_case_map.get(content_type, ['General purpose', 'Web content']))
        
        # Add quality-specific use cases
        if quality_score > 85:
            use_cases.extend(['Print publication', 'Professional portfolio', 'High-end marketing'])
        elif quality_score > 70:
            use_cases.extend(['Web publication', 'Social media', 'Digital marketing'])
        else:
            use_cases.extend(['Web thumbnail', 'Internal use', 'Draft material'])
        
        # Add composition-specific use cases
        composition = technical.get('composition', {})
        if composition.get('rule_of_thirds_score', 0) > 70:
            use_cases.append('Featured image')
        if composition.get('balance_score', 0) > 80:
            use_cases.append('Brand imagery')
        
        return list(set(use_cases))  # Remove duplicates
    
    def _calculate_final_scores(self, technical: Dict[str, Any]) -> Dict[str, float]:
        """Calculate detailed score breakdown"""
        
        quality = technical.get('quality_metrics', {})
        composition = technical.get('composition', {})
        colors = technical.get('color_analysis', {})
        
        return {
            'technical_quality': round((
                quality.get('sharpness', 50) * 0.3 +
                quality.get('contrast', 50) * 0.25 +
                quality.get('brightness', 50) * 0.2 +
                (100 - quality.get('noise_level', 50)) * 0.25
            ), 1),
            'artistic_composition': composition.get('composition_score', 50),
            'color_harmony': colors.get('color_harmony_score', 50),
            'visual_impact': round((
                composition.get('leading_lines_score', 50) * 0.4 +
                colors.get('color_richness', 50) * 0.3 +
                composition.get('balance_score', 50) * 0.3
            ), 1)
        }
    
    def _calculate_cost_analysis(self, ai_source: str) -> Dict[str, Any]:
        """Calculate cost analysis for the processing"""
        
        if ai_source == 'ollama':
            cost_per_image = 0.001  # Minimal electricity cost
            cost_type = 'Local processing'
        elif ai_source == 'fallback':
            cost_per_image = 0.0
            cost_type = 'Template-based'
        else:
            cost_per_image = 0.02  # Estimated external API cost
            cost_type = 'External API'
        
        return {
            'cost_per_image': cost_per_image,
            'cost_type': cost_type,
            'monthly_cost_1000_images': round(cost_per_image * 1000, 2),
            'savings_vs_openai': round(0.02 - cost_per_image, 3)
        }
    
    def _update_stats(self, processing_time: float, ai_source: str):
        """Update processing statistics"""
        
        self.processing_stats['total_processed'] += 1
        
        if ai_source == 'ollama':
            self.processing_stats['ollama_used'] += 1
        else:
            self.processing_stats['fallback_used'] += 1
        
        # Update average processing time
        total = self.processing_stats['total_processed']
        current_avg = self.processing_stats['avg_processing_time']
        self.processing_stats['avg_processing_time'] = (
            (current_avg * (total - 1) + processing_time) / total
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and statistics"""
        
        return {
            'ollama_available': self.ollama.available,
            'ollama_model': self.ollama.model_name,
            'processing_stats': self.processing_stats,
            'estimated_monthly_cost': round(
                self.processing_stats['total_processed'] * 0.001 * 30, 2
            ),
            'system_health': 'Healthy' if self.ollama.available else 'Degraded (Fallback mode)'
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize hybrid analyzer
    analyzer = HybridImageAnalyzer()
    
    # Test with a sample image (you would load actual image data)
    print("Hybrid Image Analyzer Test")
    print("=" * 50)
    
    # System status
    status = analyzer.get_system_status()
    print(f"System Status: {status['system_health']}")
    print(f"Ollama Available: {status['ollama_available']}")
    print(f"Model: {status['ollama_model']}")
    print()
    
    # For testing, create dummy image data
    # In real usage, you would load actual image bytes
    dummy_image = Image.new('RGB', (800, 600), color='red')
    img_byte_arr = BytesIO()
    dummy_image.save(img_byte_arr, format='JPEG')
    dummy_image_data = img_byte_arr.getvalue()
    
    print("Processing test image...")
    result = analyzer.analyze_image_complete(dummy_image_data, "test_image.jpg")
    
    print(f"Processing completed in {result['processing_metadata']['total_time']} seconds")
    print(f"Overall Quality Score: {result['quality_analysis']['overall_score']}")
    print(f"AI Description: {result['ai_description'][:100]}...")
    print(f"Smart Tags: {', '.join(result['smart_tags'][:10])}")
    print(f"Cost per image: ${result['cost_analysis']['cost_per_image']}")
    print(f"AI Source: {result['processing_metadata']['ai_source']}")

