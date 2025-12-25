from ultralytics import YOLO
import numpy as np
from typing import List, Dict, Any, Union
import cv2


class YoloProcessor:
    def __init__(self, model_name: str = "yolov8n.pt"):
        """
        Initialize the YOLO processor.
        Args:
            model_name: Name of the YOLO model file (default: "yolov8n.pt").
        """
        self.model = YOLO(model_name)


    def detect_objects(self, image: Union[str, np.ndarray], conf_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Detect objects in a cropped image (suspicious region).
        Args:
            image: Path to image crop or numpy array (the crop).
            conf_threshold: Confidence threshold.
        Returns:
            List of detected objects relative to the crop.
        """
        results = self.model(image, conf=conf_threshold)
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                detections.append({
                    "class": cls_name,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2]
                })
        return detections


    def classify_crop(self, image: Union[str, np.ndarray], conf_threshold: float = 0.5) -> str:
        """
        Classify the dominant object in a given image crop.
        Args:
            image: Path to image crop or numpy array.
            conf_threshold: Confidence threshold.
        Returns:
            The class name of the detected object with the highest confidence.
            Returns "unknown" if no objects are detected.
        """
        detections = self.detect_objects(image, conf_threshold)
        if not detections:
            return "unknown"
        best_detection = sorted(detections, key=lambda x: x['confidence'], reverse=True)[0]
        return best_detection['class']


def process_motion_and_alert(image, motion_bbox, yolo, alert_classes=("person", "car", "truck", "bus", "motorcycle")):
    """
    חותך crop לפי תנועה, מריץ YOLO, מסמן ומחזיר רק אם זוהה אדם/רכב.
    Args:
        image: np.ndarray (BGR) - התמונה המקורית
        motion_bbox: (x1, y1, x2, y2) - תיבת התנועה
        yolo: מופע YoloProcessor
        alert_classes: tuple - אילו קלאסים יגרמו להתרעה
    Returns:
        (image_with_box, detection_dict) אם זוהה רכב/אדם, אחרת None
    """
    x1, y1, x2, y2 = motion_bbox
    crop = image[y1:y2, x1:x2]
    detections = yolo.detect_objects(crop)
    import time
    for det in detections:
        if det["class"] in alert_classes:
            bx1, by1, bx2, by2 = [int(v) for v in det["bbox"]]
            # תיאום ליחס התמונה המקורית
            abs_x1, abs_y1 = x1 + bx1, y1 + by1
            abs_x2, abs_y2 = x1 + bx2, y1 + by2
            label = f'{det["class"]} ({det["confidence"]:.2f})'
            # מסגרת אדומה
            cv2.rectangle(image, (abs_x1, abs_y1), (abs_x2, abs_y2), (0,0,255), 2)
            cv2.putText(image, label, (abs_x1, abs_y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            # שמירה עם סיומת ייחודית
            timestamp = int(time.time()*1000)
            save_path = fr"C:\Users\User\Downloads\alert_{det['class']}_{int(det['confidence']*100)}_{timestamp}.jpg"
            cv2.imwrite(save_path, image)
            return image, det, save_path
    return None


# דוגמת שימוש - יש למקם אחרי הגדרת המחלקה
if __name__ == "__main__":
    image_path = r"C:\Users\User\Downloads\הורדה (2).jpg"
    yolo = YoloProcessor()
    image = cv2.imread(image_path)
    # דוגמה: תיבת תנועה (יש להחליף בזיהוי אמיתי)
    motion_bbox = (50, 50, image.shape[1]-50, image.shape[0]-50)
    result = process_motion_and_alert(image, motion_bbox, yolo)
    if result:
        image_with_box, det, save_path = result
        print(f"Alert: {det['class']} (confidence: {det['confidence']:.2f})")
        print(f"Image saved to: {save_path}")
    else:
        print("No alert-worthy object detected.")