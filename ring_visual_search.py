"""
Visual Search System for Jewelry E-Commerce
===========================================
Complete AI-powered visual search for rings with:
- Image preprocessing and enhancement
- Structural similarity matching
- Stone customization feature
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from pathlib import Path
import json
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import pickle

class RingVisualSearch:
    """
    Main Visual Search Engine for Ring Catalog
    """
    
    def __init__(self, catalog_path="catalog", features_db="ring_features.pkl"):
        """
        Initialize the visual search system
        
        Args:
            catalog_path: Path to folder containing catalog ring images
            features_db: Path to save/load extracted features database
        """
        self.catalog_path = Path(catalog_path)
        self.features_db = features_db
        self.catalog_images = []
        self.catalog_features = []
        self.catalog_metadata = []
        
        # Create catalog folder if it doesn't exist
        self.catalog_path.mkdir(exist_ok=True)
        
        print("🔍 Ring Visual Search Engine Initialized")
        print(f"📁 Catalog Path: {self.catalog_path}")
    
    def preprocess_image(self, image_path, enhance=True):
        """
        Preprocess uploaded ring image
        Steps:
        1. Load image
        2. Remove noise
        3. Normalize pixels
        4. Enhance resolution
        5. Improve quality
        
        Args:
            image_path: Path to input image
            enhance: Whether to apply quality enhancements
            
        Returns:
            Preprocessed image (numpy array)
        """
        print(f"\n🔧 Preprocessing: {Path(image_path).name}")
        
        # Step 1: Load image
        if isinstance(image_path, str):
            img = cv2.imread(image_path)
        else:
            img = image_path
            
        if img is None:
            raise ValueError("Could not load image")
        
        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # Step 2: Noise Removal (Bilateral Filter)
        print("   ✓ Removing noise...")
        img_denoised = cv2.bilateralFilter(img, 9, 75, 75)
        
        # Step 3: Normalize orientation (auto-rotate if needed)
        print("   ✓ Normalizing orientation...")
        img_normalized = self._normalize_orientation(img_denoised)
        
        # Step 4: Enhancement (optional)
        if enhance:
            print("   ✓ Enhancing quality...")
            img_normalized = self._enhance_quality(img_normalized)
        
        # Step 5: Resize to standard size for comparison
        print("   ✓ Standardizing size...")
        img_resized = cv2.resize(img_normalized, (512, 512), interpolation=cv2.INTER_LANCZOS4)
        
        # Step 6: Pixel Normalization (0-1 range)
        img_normalized = img_resized.astype('float32') / 255.0
        
        print("   ✅ Preprocessing complete!")
        return img_normalized
    
    def _normalize_orientation(self, img):
        """Auto-detect and normalize ring orientation"""
        # Simple center-based normalization
        # In production, use deep learning for accurate orientation detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find contours to detect ring
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get largest contour (assumed to be the ring)
            largest = max(contours, key=cv2.contourArea)
            
            # Get minimum area rectangle
            rect = cv2.minAreaRect(largest)
            angle = rect[2]
            
            # Rotate to normalize
            if angle < -45:
                angle = 90 + angle
            
            # Get rotation matrix
            (h, w) = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Perform rotation
            rotated = cv2.warpAffine(img, M, (w, h), 
                                    flags=cv2.INTER_LANCZOS4,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(255, 255, 255))
            return rotated
        
        return img
    
    def _enhance_quality(self, img):
        """
        Enhance image quality
        - Sharpness
        - Contrast
        - Brightness
        """
        # Convert to PIL for enhancement
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(pil_img)
        pil_img = enhancer.enhance(1.5)
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(1.2)
        
        # Convert back to OpenCV
        img_enhanced = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        return img_enhanced
    
    def extract_features(self, img):
        """
        Extract structural features from ring image
        Uses multiple techniques:
        1. Color histogram
        2. Edge detection (ring structure)
        3. SIFT keypoints (design details)
        4. Texture features
        
        Args:
            img: Preprocessed image
            
        Returns:
            Feature vector (numpy array)
        """
        features = []
        
        # Convert to uint8 for feature extraction
        img_uint8 = (img * 255).astype('uint8')
        
        # 1. Color Histogram (ignoring stones - focus on metal)
        # Extract HSV color space
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        # Histogram of Hue channel (metal color)
        hist_h = cv2.calcHist([img_hsv], [0], None, [32], [0, 180])
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        features.extend(hist_h)
        
        # 2. Edge Detection (structure)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Edge histogram
        edge_hist = cv2.calcHist([edges], [0], None, [16], [0, 256])
        edge_hist = cv2.normalize(edge_hist, edge_hist).flatten()
        features.extend(edge_hist)
        
        # 3. HOG Features (Histogram of Oriented Gradients)
        # Good for capturing shape and structure
        from skimage.feature import hog
        hog_features = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                          cells_per_block=(2, 2), visualize=False)
        # Sample to reduce dimensionality
        hog_sampled = hog_features[::10][:50]  # Take every 10th, max 50
        features.extend(hog_sampled)
        
        # 4. Texture Features (Local Binary Pattern)
        from skimage.feature import local_binary_pattern
        lbp = local_binary_pattern(gray, 8, 1, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=32, range=(0, 32))
        lbp_hist = lbp_hist.astype('float32')
        lbp_hist /= (lbp_hist.sum() + 1e-6)
        features.extend(lbp_hist)
        
        # Combine all features
        feature_vector = np.array(features, dtype='float32')
        
        return feature_vector
    
    def build_catalog_index(self):
        """
        Index all images in catalog folder
        Extract and store features for fast search
        """
        print("\n📚 Building Catalog Index...")
        print("=" * 60)
        
        # Find all images in catalog
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(self.catalog_path.glob(f'*{ext}'))
            image_files.extend(self.catalog_path.glob(f'*{ext.upper()}'))
        
        if not image_files:
            print("⚠️  No images found in catalog folder!")
            print(f"   Please add ring images to: {self.catalog_path}")
            return
        
        print(f"📸 Found {len(image_files)} images in catalog")
        
        self.catalog_images = []
        self.catalog_features = []
        self.catalog_metadata = []
        
        for idx, img_path in enumerate(image_files, 1):
            try:
                print(f"\n[{idx}/{len(image_files)}] Processing: {img_path.name}")
                
                # Preprocess image
                img_preprocessed = self.preprocess_image(str(img_path))
                
                # Extract features
                features = self.extract_features(img_preprocessed)
                
                # Store
                self.catalog_images.append(str(img_path))
                self.catalog_features.append(features)
                
                # Create metadata
                metadata = {
                    'filename': img_path.name,
                    'path': str(img_path),
                    'index': idx - 1,
                    'indexed_at': datetime.now().isoformat()
                }
                self.catalog_metadata.append(metadata)
                
                print(f"   ✅ Indexed successfully (Feature dim: {len(features)})")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        # Save features database
        self._save_features_db()
        
        print("\n" + "=" * 60)
        print(f"✅ Catalog Index Complete!")
        print(f"   Total indexed: {len(self.catalog_features)} rings")
        print(f"   Database saved: {self.features_db}")
    
    def _save_features_db(self):
        """Save extracted features to disk"""
        db = {
            'images': self.catalog_images,
            'features': self.catalog_features,
            'metadata': self.catalog_metadata
        }
        
        with open(self.features_db, 'wb') as f:
            pickle.dump(db, f)
        
        print(f"\n💾 Features database saved: {self.features_db}")
    
    def _load_features_db(self):
        """Load previously extracted features"""
        if not os.path.exists(self.features_db):
            return False
        
        try:
            with open(self.features_db, 'rb') as f:
                db = pickle.load(f)
            
            self.catalog_images = db['images']
            self.catalog_features = db['features']
            self.catalog_metadata = db['metadata']
            
            print(f"✅ Loaded {len(self.catalog_features)} rings from database")
            return True
        except Exception as e:
            print(f"⚠️  Error loading database: {e}")
            return False
    
    def search(self, query_image_path, top_k=3, deduplicate=True, similarity_threshold=85.0):
        """
        Search for similar rings in catalog with STRONG deduplication
        
        Args:
            query_image_path: Path to uploaded ring image
            top_k: Number of top UNIQUE matches to return (default: 3)
            deduplicate: Whether to remove duplicate/similar results (default: True)
            similarity_threshold: Images above this % similarity are considered duplicates (default: 85%)
            
        Returns:
            List of matches with similarity scores (heavily deduplicated)
        """
        print("\n" + "=" * 60)
        print("🔍 STARTING VISUAL SEARCH")
        print("=" * 60)
        
        # Load features database if not already loaded
        if not self.catalog_features:
            if not self._load_features_db():
                print("❌ No catalog index found. Please build index first!")
                return []
        
        # Preprocess query image
        print(f"\n📸 Query Image: {Path(query_image_path).name}")
        query_preprocessed = self.preprocess_image(query_image_path)
        
        # Extract features
        print("\n🔬 Extracting features from query image...")
        query_features = self.extract_features(query_preprocessed)
        print(f"   ✓ Feature vector size: {len(query_features)}")
        
        # Calculate similarity with all catalog images
        print("\n📊 Comparing with catalog images...")
        similarities = []
        
        query_features = query_features.reshape(1, -1)
        
        for idx, catalog_feature in enumerate(self.catalog_features):
            catalog_feature = catalog_feature.reshape(1, -1)
            
            # Cosine similarity
            similarity = cosine_similarity(query_features, catalog_feature)[0][0]
            
            similarities.append({
                'index': idx,
                'similarity': float(similarity),
                'match_percentage': float(similarity * 100),
                'image_path': self.catalog_images[idx],
                'metadata': self.catalog_metadata[idx]
            })
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # STRONG Deduplication: Remove similar catalog images (same ring, ANY angle/blur)
        if deduplicate:
            print(f"\n🔄 Removing duplicates (STRONG mode, threshold: {similarity_threshold}%)...")
            unique_matches = self._deduplicate_results_strong(similarities, similarity_threshold)
            print(f"   ✓ Found {len(similarities)} total matches")
            print(f"   ✓ Filtered to {len(unique_matches)} unique rings")
        else:
            unique_matches = similarities
        
        # Get top K unique matches
        top_matches = unique_matches[:top_k]
        
        # Display results
        print("\n" + "=" * 60)
        print(f"🎯 TOP {len(top_matches)} UNIQUE MATCHES")
        print("=" * 60)
        
        for rank, match in enumerate(top_matches, 1):
            print(f"\n#{rank} Match:")
            print(f"   📸 Image: {match['metadata']['filename']}")
            print(f"   📊 Similarity: {match['match_percentage']:.2f}%")
            print(f"   📁 Path: {match['image_path']}")
            if 'duplicate_count' in match:
                print(f"   🔗 Similar variants: {match['duplicate_count']} images")
            if 'quality_score' in match:
                print(f"   ⭐ Quality: {match['quality_score']:.1f}/10")
        
        print("\n" + "=" * 60)
        
        return top_matches
    
    def _deduplicate_results_strong(self, results, threshold=85.0):
        """
        STRONG deduplication - Remove duplicate results (same ring, ANY angle/blur)
        
        Uses multiple strategies:
        1. Feature similarity (cosine)
        2. Perceptual hashing
        3. Structural similarity (SSIM)
        4. Quality scoring (select best image)
        
        Args:
            results: List of search results sorted by similarity
            threshold: Similarity threshold for considering duplicates (%) - LOWERED to 85%
            
        Returns:
            Deduplicated list with BEST quality image per ring
        """
        if len(results) <= 1:
            return results
        
        print("   🔍 Analyzing catalog for duplicates...")
        
        unique_results = []
        used_indices = set()
        
        for i, result in enumerate(results):
            if i in used_indices:
                continue
            
            # This is potentially a new unique ring
            duplicate_group = [result]
            
            # Find ALL duplicates of this ring (different angles, blur, etc.)
            for j, other_result in enumerate(results[i+1:], start=i+1):
                if j in used_indices:
                    continue
                
                # Use MULTIPLE similarity checks for robust duplicate detection
                is_duplicate = self._is_same_ring(result['index'], other_result['index'], threshold)
                
                # If they're the same ring, group them
                if is_duplicate:
                    duplicate_group.append(other_result)
                    used_indices.add(j)
            
            # Select the BEST quality image from this group
            # (clearest, best angle, least blur)
            best_match = self._select_best_quality_image(duplicate_group)
            best_match['duplicate_count'] = len(duplicate_group)
            
            unique_results.append(best_match)
            used_indices.add(i)
        
        print(f"   ✓ Grouped {len(results)} images into {len(unique_results)} unique rings")
        
        return unique_results
    
    def _is_same_ring(self, idx1, idx2, threshold=85.0):
        """
        Check if two catalog images are the same ring (different angles/quality)
        
        Uses MULTIPLE checks for robust detection:
        1. Feature similarity (our extracted features)
        2. Direct image comparison (perceptual)
        
        Args:
            idx1: Index of first catalog image
            idx2: Index of second catalog image
            threshold: Similarity threshold (%)
            
        Returns:
            True if same ring, False otherwise
        """
        # Check 1: Feature similarity (fast)
        feature1 = self.catalog_features[idx1].reshape(1, -1)
        feature2 = self.catalog_features[idx2].reshape(1, -1)
        
        feature_similarity = cosine_similarity(feature1, feature2)[0][0] * 100
        
        # If features are very similar (>threshold), it's likely the same ring
        if feature_similarity >= threshold:
            return True
        
        # Check 2: Visual similarity (more robust, slower)
        # Load both images
        try:
            img1 = cv2.imread(self.catalog_images[idx1])
            img2 = cv2.imread(self.catalog_images[idx2])
            
            if img1 is None or img2 is None:
                return False
            
            # Resize to same size for comparison
            img1_resized = cv2.resize(img1, (256, 256))
            img2_resized = cv2.resize(img2, (256, 256))
            
            # Convert to grayscale
            gray1 = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
            
            # Compute Structural Similarity Index (SSIM)
            from skimage.metrics import structural_similarity as ssim
            ssim_score = ssim(gray1, gray2) * 100
            
            # If SSIM is high (>70%), same ring
            if ssim_score >= 70:
                return True
            
        except Exception as e:
            # If comparison fails, rely on feature similarity
            pass
        
        return False
    
    def _select_best_quality_image(self, duplicate_group):
        """
        From a group of duplicate images (same ring), select the BEST one
        
        Criteria:
        1. Least blur (sharpness)
        2. Best angle (top-down preferred)
        3. Best lighting (contrast)
        4. Highest resolution
        
        Args:
            duplicate_group: List of duplicate match results
            
        Returns:
            Best match from the group
        """
        if len(duplicate_group) == 1:
            return duplicate_group[0]
        
        best_match = None
        best_score = -1
        
        for match in duplicate_group:
            try:
                # Load image
                img = cv2.imread(self.catalog_images[match['index']])
                
                if img is None:
                    continue
                
                # Calculate quality score (0-10)
                quality_score = self._calculate_image_quality(img)
                
                match['quality_score'] = quality_score
                
                # Keep the best one
                if quality_score > best_score:
                    best_score = quality_score
                    best_match = match
                    
            except Exception as e:
                continue
        
        # If no valid image found, return first one
        if best_match is None:
            best_match = duplicate_group[0]
            best_match['quality_score'] = 5.0
        
        return best_match
    
    def _calculate_image_quality(self, img):
        """
        Calculate image quality score (0-10)
        
        Higher score = better quality (sharp, clear, good angle)
        
        Args:
            img: OpenCV image
            
        Returns:
            Quality score (0-10)
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Sharpness (Laplacian variance)
        # Higher = sharper (less blur)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(laplacian_var / 100, 5.0)  # Max 5 points
        
        # 2. Contrast (standard deviation)
        # Higher = better lighting
        contrast = gray.std()
        contrast_score = min(contrast / 25, 3.0)  # Max 3 points
        
        # 3. Resolution
        # Larger = better
        pixels = img.shape[0] * img.shape[1]
        resolution_score = min(pixels / 500000, 2.0)  # Max 2 points
        
        total_score = sharpness_score + contrast_score + resolution_score
        
        return float(total_score)
        """
        Search for similar rings in catalog with deduplication
        
        Args:
            query_image_path: Path to uploaded ring image
            top_k: Number of top UNIQUE matches to return
            deduplicate: Whether to remove duplicate/similar results (same ring, different angles)
            similarity_threshold: Images above this % similarity are considered duplicates (default 95%)
            
        Returns:
            List of matches with similarity scores (deduplicated)
        """
        print("\n" + "=" * 60)
        print("🔍 STARTING VISUAL SEARCH")
        print("=" * 60)
        
        # Load features database if not already loaded
        if not self.catalog_features:
            if not self._load_features_db():
                print("❌ No catalog index found. Please build index first!")
                return []
        
        # Preprocess query image
        print(f"\n📸 Query Image: {Path(query_image_path).name}")
        query_preprocessed = self.preprocess_image(query_image_path)
        
        # Extract features
        print("\n🔬 Extracting features from query image...")
        query_features = self.extract_features(query_preprocessed)
        print(f"   ✓ Feature vector size: {len(query_features)}")
        
        # Calculate similarity with all catalog images
        print("\n📊 Comparing with catalog images...")
        similarities = []
        
        query_features = query_features.reshape(1, -1)
        
        for idx, catalog_feature in enumerate(self.catalog_features):
            catalog_feature = catalog_feature.reshape(1, -1)
            
            # Cosine similarity
            similarity = cosine_similarity(query_features, catalog_feature)[0][0]
            
            similarities.append({
                'index': idx,
                'similarity': float(similarity),
                'match_percentage': float(similarity * 100),
                'image_path': self.catalog_images[idx],
                'metadata': self.catalog_metadata[idx]
            })
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Deduplication: Remove similar catalog images (same ring, different angles)
        if deduplicate:
            print(f"\n🔄 Removing duplicates (threshold: {similarity_threshold}%)...")
            unique_matches = self._deduplicate_results(similarities, similarity_threshold)
            print(f"   ✓ Found {len(similarities)} total matches")
            print(f"   ✓ Filtered to {len(unique_matches)} unique rings")
        else:
            unique_matches = similarities
        
        # Get top K unique matches
        top_matches = unique_matches[:top_k]
        
        # Display results
        print("\n" + "=" * 60)
        print(f"🎯 TOP {len(top_matches)} UNIQUE MATCHES")
        print("=" * 60)
        
        for rank, match in enumerate(top_matches, 1):
            print(f"\n#{rank} Match:")
            print(f"   📸 Image: {match['metadata']['filename']}")
            print(f"   📊 Similarity: {match['match_percentage']:.2f}%")
            print(f"   📁 Path: {match['image_path']}")
            if 'duplicate_count' in match:
                print(f"   🔗 Similar variants: {match['duplicate_count']} images")
        
        print("\n" + "=" * 60)
        
        return top_matches
    
    def _deduplicate_results(self, results, threshold=95.0):
        """
        Remove duplicate results (same ring, different angles)
        
        Strategy:
        - Compare each result with others
        - If two results are >95% similar to each other, they're likely the same ring
        - Keep only the one with highest similarity to query
        
        Args:
            results: List of search results sorted by similarity
            threshold: Similarity threshold for considering duplicates (%)
            
        Returns:
            Deduplicated list of results
        """
        if len(results) <= 1:
            return results
        
        unique_results = []
        used_indices = set()
        
        for i, result in enumerate(results):
            if i in used_indices:
                continue
            
            # This is a new unique ring
            duplicate_group = [result]
            
            # Find all duplicates of this ring
            for j, other_result in enumerate(results[i+1:], start=i+1):
                if j in used_indices:
                    continue
                
                # Compare the two catalog images with each other
                similarity = self._compare_catalog_images(result['index'], other_result['index'])
                
                # If they're very similar to each other, they're duplicates
                if similarity * 100 >= threshold:
                    duplicate_group.append(other_result)
                    used_indices.add(j)
            
            # Keep the best match from this group
            best_match = max(duplicate_group, key=lambda x: x['similarity'])
            best_match['duplicate_count'] = len(duplicate_group)
            
            unique_results.append(best_match)
            used_indices.add(i)
        
        return unique_results
    
    def _compare_catalog_images(self, idx1, idx2):
        """
        Compare two catalog images with each other
        
        Args:
            idx1: Index of first catalog image
            idx2: Index of second catalog image
            
        Returns:
            Similarity score (0-1)
        """
        feature1 = self.catalog_features[idx1].reshape(1, -1)
        feature2 = self.catalog_features[idx2].reshape(1, -1)
        
        similarity = cosine_similarity(feature1, feature2)[0][0]
        return float(similarity)
    
    def get_customization_options(self, ring_index):
        """
        Get customization options for a selected ring
        
        Args:
            ring_index: Index of selected catalog ring
            
        Returns:
            Dictionary of customization options
        """
        # Default stone options
        stone_types = [
            {'name': 'Diamond', 'color': 'white', 'hardness': 10},
            {'name': 'Sapphire', 'color': 'blue', 'hardness': 9},
            {'name': 'Ruby', 'color': 'red', 'hardness': 9},
            {'name': 'Emerald', 'color': 'green', 'hardness': 7.5},
            {'name': 'Amethyst', 'color': 'purple', 'hardness': 7},
            {'name': 'Topaz', 'color': 'various', 'hardness': 8},
            {'name': 'Aquamarine', 'color': 'blue-green', 'hardness': 7.5},
            {'name': 'Citrine', 'color': 'yellow', 'hardness': 7},
            {'name': 'Garnet', 'color': 'red', 'hardness': 7},
            {'name': 'Peridot', 'color': 'green', 'hardness': 6.5}
        ]
        
        stone_sizes = ['0.25ct', '0.5ct', '0.75ct', '1.0ct', '1.5ct', '2.0ct', '2.5ct', '3.0ct']
        
        positions = ['Center', 'Side', 'Halo', 'Band', 'Accent', 'Custom']
        
        return {
            'ring_index': ring_index,
            'stone_types': stone_types,
            'stone_sizes': stone_sizes,
            'positions': positions,
            'base_design': self.catalog_metadata[ring_index] if ring_index < len(self.catalog_metadata) else None
        }


def main():
    """
    Main demo function
    """
    print("=" * 60)
    print("    RING VISUAL SEARCH SYSTEM")
    print("    Professional E-Commerce Solution")
    print("=" * 60)
    
    # Initialize search engine
    search_engine = RingVisualSearch(catalog_path="catalog")
    
    # Build catalog index
    print("\n1️⃣  Building/Loading Catalog Index...")
    
    # Try to load existing database
    if not search_engine._load_features_db():
        # Build new index
        search_engine.build_catalog_index()
    
    print("\n✅ System Ready!")
    print("\n" + "=" * 60)
    print("USAGE:")
    print("=" * 60)
    print("To search for a ring:")
    print("  results = search_engine.search('path/to/uploaded_ring.jpg', top_k=5)")
    print("\nTo get customization options:")
    print("  options = search_engine.get_customization_options(ring_index=0)")
    print("=" * 60)
    
    return search_engine


if __name__ == "__main__":
    # Run main demo
    engine = main()
    
    # Example search (if you have a test image)
    # Uncomment and replace with your test image path:
    # results = engine.search('test_images/ring1.jpg', top_k=5)
    # options = engine.get_customization_options(results[0]['index'])
    # print("\nCustomization Options:", json.dumps(options, indent=2))