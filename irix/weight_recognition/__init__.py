from .qr_reader import PlateQRReader, PLATE_REGISTRY
from .vision_classifier import VisionPlateClassifier
from .vlm_backend import VLMBackend, LocalVLMBackend, GeminiVLMBackend, FakeVLMBackend
from .confirmation import ExtractionConfirmer, ConfirmedReading, validate_weight_kg

__all__ = [
    "PlateQRReader", "PLATE_REGISTRY",
    "VisionPlateClassifier", "VLMBackend", "LocalVLMBackend", "GeminiVLMBackend", "FakeVLMBackend",
    "ExtractionConfirmer", "ConfirmedReading", "validate_weight_kg",
]
