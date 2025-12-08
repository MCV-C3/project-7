import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
from typing import Tuple, Optional, List

class BOVW:
    def __init__(self, detector_type="AKAZE", codebook_size=50, dense_sift=False, 
                 sift_step=10, sift_scales=1, use_scaler=False, 
                 detector_kwargs=None, codebook_kwargs=None, pca_components=None):
        self.detector_type = detector_type
        self.codebook_size = codebook_size
        self.dense_sift = dense_sift
        self.sift_step = sift_step
        self.sift_scales = sift_scales
        self.use_scaler = use_scaler
        self.pca_components = pca_components
        self.pca = PCA(n_components=pca_components) if pca_components is not None else None
        
        detector_kwargs = detector_kwargs or {}
        codebook_kwargs = codebook_kwargs or {}
        
        if detector_type == 'SIFT':
            self.detector = cv2.SIFT_create(**detector_kwargs)
        elif detector_type == 'AKAZE':
            self.detector = cv2.AKAZE_create(**detector_kwargs)
        elif detector_type == 'ORB':
            self.detector = cv2.ORB_create(**detector_kwargs)
        else:
            raise ValueError("Detector type must be 'SIFT', 'SURF', or 'ORB'")
        
        self.codebook_algo = MiniBatchKMeans(n_clusters=self.codebook_size, **codebook_kwargs)
        self.scaler = StandardScaler() if use_scaler else None
    
    def _extract_dense_sift(self, image: np.ndarray) -> Tuple[List, Optional[np.ndarray]]:
        if self.detector_type != 'SIFT':
            raise ValueError("Dense SIFT only works with SIFT detector")
        # Ensure we operate on a grayscale image
        if image.ndim == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        all_descriptors = []
        all_keypoints = []

        for scale_idx in range(self.sift_scales):
            scale_factor = 2 ** scale_idx
            # Downscale the image for this pyramid level
            scaled_img = cv2.resize(gray, None, fx=1/scale_factor, fy=1/scale_factor, interpolation=cv2.INTER_LINEAR)

            h, w = scaled_img.shape[:2]
            # keypoints for this scale (in coordinates of scaled_img)
            kp_scale = []
            for y in range(0, h, self.sift_step):
                for x in range(0, w, self.sift_step):
                    kp = cv2.KeyPoint(float(x), float(y), float(self.sift_step))
                    kp_scale.append(kp)

            if not kp_scale:
                continue

            _, descriptors = self.detector.compute(scaled_img, kp_scale)
            if descriptors is not None and len(descriptors) > 0:
                all_descriptors.append(descriptors)
                # store keypoints mapped back to original image coordinates
                for kp in kp_scale:
                    orig_x = kp.pt[0] * scale_factor
                    orig_y = kp.pt[1] * scale_factor
                    orig_size = kp.size * scale_factor
                    all_keypoints.append(cv2.KeyPoint(float(orig_x), float(orig_y), float(orig_size)))

        if all_descriptors:
            return all_keypoints, np.vstack(all_descriptors)
        return [], None
    
    def _extract_features(self, image: np.ndarray) -> Tuple[List, Optional[np.ndarray]]:
        if self.dense_sift:
            return self._extract_dense_sift(image)
        return self.detector.detectAndCompute(image, None)
    
    def _update_fit_codebook(self, descriptors: List[np.ndarray]) -> Tuple[MiniBatchKMeans, np.ndarray]:
        all_descriptors = np.vstack(descriptors)
        
        if self.scaler is not None:
            all_descriptors = self.scaler.fit_transform(all_descriptors)
        
        self.codebook_algo.fit(all_descriptors)
        return self.codebook_algo, self.codebook_algo.cluster_centers_
    
    def _compute_codebook_descriptor(self, descriptors: np.ndarray, kmeans=None) -> np.ndarray:
        if kmeans is None:
            kmeans = self.codebook_algo
        
        if self.scaler is not None:
            descriptors = self.scaler.transform(descriptors)
        
        visual_words = kmeans.predict(descriptors)
        histogram = np.zeros(kmeans.n_clusters)
        
        for label in visual_words:
            histogram[label] += 1
        
        histogram = histogram / (np.linalg.norm(histogram) + 1e-6)
        return histogram
    
    def _fit_transform_pca(self, histograms):
        if self.pca is None:
            return histograms
        self.pca.fit(histograms)
        return self.pca.transform(histograms)
    
    def transform_pca(self, histograms):
        if self.pca is None:
            return histograms
        return self.pca.transform(histograms)

def visualize_bow_histogram(histogram, image_index, output_folder="./histograms"):
    os.makedirs(output_folder, exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(histogram)), histogram)
    plt.title(f"BoVW Histogram for Image {image_index}")
    plt.xlabel("Visual Word Index")
    plt.ylabel("Frequency")
    plt.xticks(range(len(histogram)))
    
    plot_path = os.path.join(output_folder, f"bovw_histogram_image_{image_index}.png")
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Plot saved to: {plot_path}")

