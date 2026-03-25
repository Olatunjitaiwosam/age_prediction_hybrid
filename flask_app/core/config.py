import os

class Config:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(BASE_DIR, "models"))
    UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
    SECRET_KEY = os.environ.get("SECRET_KEY", "change-me-in-production")
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500 MB

    IMAGE_SIZE = 224
    AGE_GROUPS = {"child": (0, 12), "teen": (13, 17), "adult": (18, 116)}
    GROUP_NAMES = ["child", "teen", "adult"]
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    DEFAULT_BACKBONE = os.environ.get("MODEL_BACKBONE", "densenet")
    CNN_FILENAME     = os.environ.get("CNN_FILENAME",  "densenet_best.pth")
    YOLO_FILENAME    = os.environ.get("YOLO_FILENAME", "yolo_face_best.pt")

    BACKBONE_FILES = {
        "densenet":        "densenet_best.pth",
        "resnet50":        "resnet50_best.pth",
        "efficientnet_v2": "efficientnet_v2_best.pth",
        "convnext":        "convnext_best.pth",
        "vit-base":        "vit_base_best.pth",
        "vit-tiny":        "vit_base_best.pth",
    }

    _GH_BASE = "https://github.com/Mystique1337/age_prediction_hybrid-/releases/download/v1.0-models"
    MODEL_URLS = {
        "densenet_best.pth":        f"{_GH_BASE}/densenet_best.pth",
        "resnet50_best.pth":        f"{_GH_BASE}/resnet50_best.pth",
        "efficientnet_v2_best.pth": f"{_GH_BASE}/efficientnet_v2_best.pth",
        "convnext_best.pth":        f"{_GH_BASE}/convnext_best.pth",
        "vit_base_best.pth":        f"{_GH_BASE}/vit_base_best.pth",
        "yolo_face_best.pt":        f"{_GH_BASE}/yolo_face_best.pt",
    }

    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
