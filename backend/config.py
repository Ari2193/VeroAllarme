"""
VeroAllarme Configuration
Central configuration file for the smart motion alert filtering system
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings and configuration"""
    
    # Application
    APP_NAME: str = "VeroAllarme"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_PREFIX: str = "/api"
    
    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://veroallarme:password@localhost:5432/veroallarme"
    )
    
    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    REDIS_CACHE_TTL: int = 3600  # 1 hour
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    IMAGES_DIR: Path = DATA_DIR / "images"
    MASKS_DIR: Path = DATA_DIR / "masks"
    HEATMAPS_DIR: Path = DATA_DIR / "heatmaps"
    MODELS_DIR: Path = BASE_DIR / "models"
    
    # Motion Detection Settings
    MOTION_THRESHOLD: int = 25  # Pixel difference threshold
    MOTION_MIN_AREA: int = 500  # Minimum motion area in pixels
    GAUSSIAN_BLUR_KERNEL: tuple = (21, 21)
    MORPHOLOGY_KERNEL_SIZE: int = 5
    
    # Heat Map Settings
    HEATMAP_HISTORY_DAYS: int = 30
    HEATMAP_UPDATE_INTERVAL: int = 3600  # Update every hour
    HEATMAP_ANOMALY_THRESHOLD: float = 0.3  # Low probability zones
    
    # Anomaly Detection Settings
    ANOMALY_Z_SCORE_THRESHOLD: float = 2.5
    ANOMALY_MIN_SAMPLES: int = 50  # Minimum historical samples
    
    # YOLO Settings
    YOLO_MODEL_PATH: str = str(MODELS_DIR / "yolov8n.pt")
    YOLO_CONFIDENCE_THRESHOLD: float = 0.6
    YOLO_IOU_THRESHOLD: float = 0.45
    YOLO_DEVICE: str = os.getenv("YOLO_DEVICE", "cpu")  # 'cpu' or 'cuda'
    YOLO_IMAGE_SIZE: int = 640
    
    # YOLO Trigger Conditions
    YOLO_TRIGGER_ON_ANOMALY: bool = True
    YOLO_TRIGGER_ON_CRITICAL_ZONE: bool = True
    
    # Object Classes of Interest
    YOLO_CLASSES: list = [
        "person", "car", "truck", "bus", "motorcycle",
        "bicycle", "dog", "cat", "bird", "backpack"
    ]
    
    # Visualization Settings
    VISUALIZATION_MOTION_COLOR: tuple = (0, 255, 0)  # Green
    VISUALIZATION_HEATMAP_COLORMAP: str = "hot"  # OpenCV colormap
    VISUALIZATION_MASK_COLOR: tuple = (128, 128, 128)  # Gray
    VISUALIZATION_BBOX_COLOR: tuple = (255, 0, 0)  # Red
    VISUALIZATION_TEXT_COLOR: tuple = (255, 255, 255)  # White
    
    # Processing Settings
    MAX_IMAGES_PER_ALERT: int = 3
    IMAGE_PROCESSING_TIMEOUT: int = 30  # seconds
    
    # Celery Settings
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", REDIS_URL)
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    ALLOWED_ORIGINS: list = [
        "http://localhost:3000",
        "http://localhost:8000",
    ]
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()


# Create necessary directories on startup
def initialize_directories():
    """Create all required directories if they don't exist"""
    directories = [
        settings.DATA_DIR,
        settings.IMAGES_DIR,
        settings.MASKS_DIR,
        settings.HEATMAPS_DIR,
        settings.MODELS_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✓ Directory ensured: {directory}")


if __name__ == "__main__":
    # Test configuration
    initialize_directories()
    print(f"\n{settings.APP_NAME} v{settings.APP_VERSION}")
    print(f"Debug mode: {settings.DEBUG}")
    print(f"YOLO device: {settings.YOLO_DEVICE}")
    print(f"Database: {settings.DATABASE_URL}")
