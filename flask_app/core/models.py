import os
import logging
import requests
import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False


class AgePredictionCNN(nn.Module):
    def __init__(self, backbone: str = "densenet", pretrained: bool = False):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "resnet50":
            base = models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
            self.features = nn.Sequential(*list(base.children())[:-1])
            feat_dim = 2048
        elif backbone == "efficientnet_v2":
            base = models.efficientnet_v2_m(weights="IMAGENET1K_V1" if pretrained else None)
            self.features = base.features
            feat_dim = 1280
        elif backbone == "convnext":
            base = models.convnext_base(weights="IMAGENET1K_V1" if pretrained else None)
            self.features = base.features
            feat_dim = 1024
        elif backbone == "densenet":
            base = models.densenet201(weights="IMAGENET1K_V1" if pretrained else None)
            self.features = base.features
            feat_dim = 1920
        else:
            raise ValueError(f"Unsupported CNN backbone: {backbone}")

        self.age_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Dropout(0.4), nn.Linear(feat_dim, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.group_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Dropout(0.4), nn.Linear(feat_dim, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        feats = self.features(x)
        if isinstance(feats, torch.Tensor) and feats.dim() == 2:
            feats = feats.unsqueeze(-1).unsqueeze(-1)
        return self.age_head(feats), self.group_head(feats)


class ViTAgePrediction(nn.Module):
    def __init__(self, model_name: str = "vit_base_patch16_224"):
        super().__init__()
        if not HAS_TIMM:
            raise ImportError("timm is required for ViT models.")
        self.vit = timm.create_model(model_name, pretrained=False, num_classes=0)
        vit_dim = self.vit.num_features

        self.age_head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(vit_dim, 512), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, 1),
        )
        self.group_head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(vit_dim, 256), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        feats = self.vit(x)
        return self.age_head(feats), self.group_head(feats)


def _is_valid_model_file(path: str, min_bytes: int = 1 << 20) -> bool:
    try:
        if os.path.getsize(path) < min_bytes:
            return False
        with open(path, "rb") as fh:
            return fh.read(4) == b"PK\x03\x04"
    except Exception:
        return False


def gh_download(url: str, dest_path: str, label: str) -> bool:
    if os.path.exists(dest_path) and _is_valid_model_file(dest_path):
        return True
    if os.path.exists(dest_path):
        logger.warning(f"Corrupt file at {dest_path}; re-downloading.")
        os.remove(dest_path)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    try:
        with requests.get(url, stream=True, timeout=300, allow_redirects=True) as r:
            r.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
        return os.path.exists(dest_path)
    except Exception as e:
        logger.warning(f"Download failed for {label}: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def load_cnn_model(backbone: str, model_path: str, device: torch.device):
    backbone_lower = backbone.lower()
    if "vit-base" in backbone_lower:
        model = ViTAgePrediction("vit_base_patch16_224")
    elif "vit-tiny" in backbone_lower:
        model = ViTAgePrediction("vit_tiny_patch16_224")
    else:
        model = AgePredictionCNN(backbone=backbone_lower)

    if os.path.exists(model_path):
        try:
            state = torch.load(model_path, map_location=device, weights_only=True)
            model.load_state_dict(state)
            logger.info(f"CNN loaded from {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load CNN weights: {e}. Using random weights.")
    else:
        logger.warning(f"CNN weights not found at {model_path}. Using random weights.")

    model.to(device).eval()
    return model


def load_yolo_model(yolo_path: str):
    if not HAS_YOLO:
        return None
    try:
        if os.path.exists(yolo_path):
            return YOLO(yolo_path)
        else:
            logger.warning("Custom YOLO weights not found. Loading pretrained YOLOv8n.")
            return YOLO("yolov8n.pt")
    except Exception as e:
        logger.warning(f"YOLO load failed: {e}")
        return None
