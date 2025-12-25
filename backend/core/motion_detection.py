"""
Motion Detection Module
Detects motion between consecutive frames and returns coordinates and visualization
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class MotionRegion:
    """Represents a detected motion region"""
    x: int
    y: int
    width: int
    height: int
    area: int
    centroid: Tuple[int, int]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "area": self.area,
            "centroid": {
                "x": self.centroid[0],
                "y": self.centroid[1]
            }
        }
    
    def to_bbox(self) -> Tuple[int, int, int, int]:
        """Get bounding box coordinates (x1, y1, x2, y2)"""
        return (self.x, self.y, self.x + self.width, self.y + self.height)


@dataclass
class MotionDetectionResult:
    """Complete motion detection result"""
    motion_detected: bool
    motion_regions: List[MotionRegion]
    motion_mask: np.ndarray
    confidence: float
    total_motion_area: int
    frame_dimensions: Tuple[int, int]  # (height, width)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format for API response"""
        return {
            "motion_detected": self.motion_detected,
            "confidence": float(self.confidence),
            "total_motion_area": int(self.total_motion_area),
            "frame_dimensions": {
                "height": self.frame_dimensions[0],
                "width": self.frame_dimensions[1]
            },
            "motion_regions": [region.to_dict() for region in self.motion_regions],
            "num_regions": len(self.motion_regions)
        }


class MotionDetector:
    """
    Advanced motion detection agent for security camera analysis.
    
    Detects motion between consecutive frames using frame differencing,
    morphological operations, and contour detection.
    
    Attributes:
        threshold: Pixel difference threshold for motion detection
        min_area: Minimum contour area to be considered as motion
        blur_kernel: Gaussian blur kernel size
        morph_kernel_size: Morphological operations kernel size
    """
    
    def __init__(
        self,
        threshold: int = 25,
        min_area: int = 500,
        blur_kernel: Tuple[int, int] = (21, 21),
        morph_kernel_size: int = 5
    ):
        """
        Initialize motion detector with configuration parameters.
        
        Args:
            threshold: Pixel difference threshold (0-255)
            min_area: Minimum motion area in pixels
            blur_kernel: Gaussian blur kernel size (must be odd)
            morph_kernel_size: Kernel size for morphological operations
        """
        self.threshold = threshold
        self.min_area = min_area
        self.blur_kernel = blur_kernel
        self.morph_kernel_size = morph_kernel_size
        
        # Create morphological kernel
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (morph_kernel_size, morph_kernel_size)
        )
        
        logger.info(
            f"MotionDetector initialized: threshold={threshold}, "
            f"min_area={min_area}, blur_kernel={blur_kernel}"
        )
    
    def load_images(self, image_paths: List[str]) -> List[np.ndarray]:
        """
        Load and validate images from file paths.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of loaded images as numpy arrays
            
        Raises:
            ValueError: If images cannot be loaded or have different dimensions
        """
        if len(image_paths) < 2:
            raise ValueError("At least 2 images are required for motion detection")
        
        images = []
        reference_shape = None
        
        for i, path in enumerate(image_paths):
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(f"Failed to load image: {path}")
            
            # Validate dimensions match
            if reference_shape is None:
                reference_shape = img.shape
            elif img.shape != reference_shape:
                raise ValueError(
                    f"Image {i} has different dimensions: "
                    f"{img.shape} vs {reference_shape}"
                )
            
            images.append(img)
        
        logger.debug(f"Loaded {len(images)} images with shape {reference_shape}")
        return images
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame: convert to grayscale and apply Gaussian blur.
        
        Args:
            frame: Input BGR image
            
        Returns:
            Preprocessed grayscale blurred image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, self.blur_kernel, 0)
        
        return blurred
    
    def compute_frame_difference(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray
    ) -> np.ndarray:
        """
        Compute absolute difference between two frames.
        
        Args:
            frame1: First preprocessed frame
            frame2: Second preprocessed frame
            
        Returns:
            Binary motion mask
        """
        # Compute absolute difference
        frame_diff = cv2.absdiff(frame1, frame2)
        
        # Apply threshold to create binary mask
        _, thresh = cv2.threshold(
            frame_diff,
            self.threshold,
            255,
            cv2.THRESH_BINARY
        )
        
        # Apply morphological operations to reduce noise
        # Dilation fills small holes
        dilated = cv2.dilate(thresh, self.morph_kernel, iterations=2)
        
        # Erosion removes small noise
        eroded = cv2.erode(dilated, self.morph_kernel, iterations=1)
        
        return eroded
    
    def find_motion_regions(self, motion_mask: np.ndarray) -> List[MotionRegion]:
        """
        Find contours in motion mask and extract motion regions.
        
        Args:
            motion_mask: Binary motion mask
            
        Returns:
            List of detected motion regions
        """
        # Find contours
        contours, _ = cv2.findContours(
            motion_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        motion_regions = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by minimum area
            if area < self.min_area:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate centroid
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2
            
            region = MotionRegion(
                x=int(x),
                y=int(y),
                width=int(w),
                height=int(h),
                area=int(area),
                centroid=(cx, cy)
            )
            
            motion_regions.append(region)
        
        # Sort by area (largest first)
        motion_regions.sort(key=lambda r: r.area, reverse=True)
        
        return motion_regions
    
    def detect_motion(
        self,
        image_paths: List[str]
    ) -> MotionDetectionResult:
        """
        Main method to detect motion from image paths.
        
        Args:
            image_paths: List of 2-3 image file paths
            
        Returns:
            MotionDetectionResult with all detection information
        """
        # Load images
        images = self.load_images(image_paths)
        
        # Preprocess all frames
        preprocessed = [self.preprocess_frame(img) for img in images]
        
        # Compute motion between consecutive frames
        motion_masks = []
        for i in range(len(preprocessed) - 1):
            mask = self.compute_frame_difference(
                preprocessed[i],
                preprocessed[i + 1]
            )
            motion_masks.append(mask)
        
        # Combine all motion masks (OR operation)
        combined_mask = np.zeros_like(motion_masks[0])
        for mask in motion_masks:
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Find motion regions
        motion_regions = self.find_motion_regions(combined_mask)
        
        # Calculate confidence based on motion area
        total_motion_area = np.sum(combined_mask > 0)
        frame_area = combined_mask.shape[0] * combined_mask.shape[1]
        confidence = min(total_motion_area / (frame_area * 0.1), 1.0)
        
        motion_detected = len(motion_regions) > 0
        
        result = MotionDetectionResult(
            motion_detected=motion_detected,
            motion_regions=motion_regions,
            motion_mask=combined_mask,
            confidence=confidence,
            total_motion_area=int(total_motion_area),
            frame_dimensions=(combined_mask.shape[0], combined_mask.shape[1])
        )
        
        logger.info(
            f"Motion detection complete: detected={motion_detected}, "
            f"regions={len(motion_regions)}, area={total_motion_area}px"
        )
        
        return result
    
    def visualize_motion(
        self,
        image_path: str,
        result: MotionDetectionResult,
        output_path: Optional[str] = None,
        show_mask: bool = True,
        show_boxes: bool = True,
        show_centroids: bool = True
    ) -> np.ndarray:
        """
        Create visualization with motion regions highlighted.
        
        Args:
            image_path: Path to base image for visualization
            result: MotionDetectionResult from detect_motion()
            output_path: Optional path to save visualization
            show_mask: Whether to overlay motion mask
            show_boxes: Whether to draw bounding boxes
            show_centroids: Whether to mark centroids
            
        Returns:
            Annotated image as numpy array
        """
        # Load base image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Create overlay copy for mask
        overlay = img.copy()
        
        # Show motion mask as green overlay
        if show_mask and result.motion_detected:
            # Create colored mask (green)
            mask_colored = np.zeros_like(img)
            mask_colored[:, :, 1] = result.motion_mask  # Green channel
            
            # Blend with original image
            overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
        
        # Draw bounding boxes and labels
        if show_boxes:
            for i, region in enumerate(result.motion_regions, 1):
                x1, y1, x2, y2 = region.to_bbox()
                
                # Draw rectangle
                cv2.rectangle(
                    overlay,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),  # Green
                    2
                )
                
                # Add label with region number and area
                label = f"#{i} ({region.area}px)"
                label_size, _ = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    1
                )
                
                # Background for text
                cv2.rectangle(
                    overlay,
                    (x1, y1 - label_size[1] - 10),
                    (x1 + label_size[0], y1),
                    (0, 255, 0),
                    -1
                )
                
                # Text
                cv2.putText(
                    overlay,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1
                )
        
        # Draw centroids
        if show_centroids:
            for region in result.motion_regions:
                cx, cy = region.centroid
                cv2.circle(overlay, (cx, cy), 5, (255, 0, 0), -1)  # Blue
                cv2.circle(overlay, (cx, cy), 6, (255, 255, 255), 1)  # White border
        
        # Add summary text
        summary = f"Motion Detected: {len(result.motion_regions)} regions"
        cv2.putText(
            overlay,
            summary,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # Save if output path provided
        if output_path:
            cv2.imwrite(str(output_path), overlay)
            logger.info(f"Visualization saved to {output_path}")
        
        return overlay


def detect_motion_from_paths(
    image_paths: List[str],
    threshold: int = 25,
    min_area: int = 500
) -> MotionDetectionResult:
    """
    Convenience function to detect motion from image paths.
    
    Args:
        image_paths: List of 2-3 image file paths
        threshold: Motion detection threshold
        min_area: Minimum motion area in pixels
        
    Returns:
        MotionDetectionResult
    """
    detector = MotionDetector(threshold=threshold, min_area=min_area)
    return detector.detect_motion(image_paths)
