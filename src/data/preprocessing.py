import cv2
import numpy as np
from src.config import PreprocessingConfig


def _load_dicom_gray(path: str) -> np.ndarray:
    """Load a DICOM file and return a grayscale uint8 numpy array."""
    try:
        import pydicom
    except ImportError as e:
        raise ImportError(
            "pydicom is required to read .dcm files. Install with `pip install pydicom`."
        ) from e
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        lo, hi = np.percentile(arr, [1, 99])
        if hi <= lo:
            hi = lo + 1.0
        arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
        arr = (arr * 255.0).astype(np.uint8)
    return arr


def _load_image_gray(path: str) -> np.ndarray:
    """Load an image as grayscale uint8, supporting DICOM and standard formats."""
    if path.lower().endswith(".dcm") or path.lower().endswith(".dicom"):
        return _load_dicom_gray(path)
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def apply_otsu_threshold(image: np.ndarray) -> np.ndarray:
    """Apply Otsu thresholding to isolate breast tissue."""
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def crop_to_breast(image: np.ndarray, margin: int = 50) -> np.ndarray:
    """Crop the image to the breast region using Otsu thresholding."""
    binary = apply_otsu_threshold(image)
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) == 0:
        return image
    y_min, x_min = coords.min(axis=0) - margin
    y_max, x_max = coords.max(axis=0) + margin
    y_min, y_max = max(0, y_min), min(image.shape[0], y_max)
    x_min, x_max = max(0, x_min), min(image.shape[1], x_max)
    return image[y_min:y_max, x_min:x_max]


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, grid_size: tuple = (8, 8)) -> np.ndarray:
    """Apply CLAHE for contrast enhancement."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(image)


def preprocess_image(image_path: str, config: PreprocessingConfig, image_size: int) -> np.ndarray:
    """Full preprocessing pipeline."""
    image = _load_image_gray(image_path)
    image = crop_to_breast(image, margin=config.crop_margin)
    image = cv2.resize(image, (image_size, image_size))
    image = apply_clahe(image, config.clahe_clip_limit, tuple(config.clahe_grid_size))
    image = image.astype(np.float32) / 255.0
    return image
