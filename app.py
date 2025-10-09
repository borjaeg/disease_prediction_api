from __future__ import annotations

import base64
import io
import os
import logging
from typing import Dict, List, Tuple

from flask import Flask, request, jsonify
from PIL import Image, ImageOps
import torch
from torch import nn
import timm

# --------------------------- Config ---------------------------
CLASS_NAMES: List[str] = ['Botrytis', 'Healthy', 'Oidium']
BACKBONE_NAME = os.getenv('BACKBONE_NAME', 'ft-models/eva02_small_patch14_336.mim_in22k_ft_in1k')
BACKBONE_PATH = os.getenv('BACKBONE_PATH', 'ft-models/eva02_small_patch14_336.mim_in22k_ft_in1k.0-backbone-NextGenBioPestSimp_2_100-False-True.pth')
HEAD_PATH     = os.getenv('HEAD_PATH',     'ft-models/eva02_small_patch14_336.mim_in22k_ft_in1k.0-head-NextGenBioPestSimp_2_100-False-True.pth')
TOP_K = int(os.getenv('TOP_K', '3'))
MAX_CONTENT_MB = float(os.getenv('MAX_CONTENT_MB', '15'))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Optional: keep PyTorch CPU threads in check for container usage
if DEVICE == 'cpu':
    torch.set_num_threads(max(1, os.cpu_count() or 1))

# --------------------------- Logging ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("inference-app")

# --------------------------- Flask ---------------------------
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = int(MAX_CONTENT_MB * 1024 * 1024)

# --------------------------- Model ---------------------------
class LinearClassifier(nn.Module):
    def __init__(self, latent_dim: int, num_classes: int = 3):
        super().__init__()
        self.mlp_head = nn.Linear(latent_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp_head(x)

class Predictor:
    def __init__(self, backbone_name: str, backbone_path: str, head_path: str | None):
        logger.info(f"Loading backbone '{backbone_name}' on {DEVICE}...")
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        state = torch.load(backbone_path, map_location='cpu')
        self.backbone.load_state_dict(state, strict=True)
        self.backbone.to(DEVICE).eval()

        self.head = LinearClassifier(self.backbone.num_features, num_classes=len(CLASS_NAMES))
        if head_path:
            logger.info("Loading classification head...")
            self.head.load_state_dict(torch.load(head_path, map_location='cpu'), strict=True)
        self.head.to(DEVICE).eval()

        # Build eval transform from model's own config
        data_cfg = timm.data.resolve_model_data_config(self.backbone)
        self.preprocess = timm.data.create_transform(**data_cfg, is_training=False)
        self.input_size = data_cfg.get("input_size", (3, 224, 224))

        # Warm-up (optional, helps first-request latency on GPU)
        with torch.inference_mode(), torch.autocast(device_type=DEVICE, enabled=(DEVICE == 'cuda')):
            dummy = torch.zeros(1, *self.input_size, device=DEVICE)
            _ = self.head(self.backbone(dummy))
        logger.info("Model ready.")

    def predict(self, img_pil: Image.Image, top_k: int = 3) -> Tuple[int, float, List[Tuple[int, float]]]:
        img = ImageOps.exif_transpose(img_pil).convert('RGB')
        tensor = self.preprocess(img).unsqueeze(0).to(DEVICE)

        with torch.inference_mode(), torch.autocast(device_type=DEVICE, enabled=(DEVICE == 'cuda')):
            feats = self.backbone(tensor)
            logits = self.head(feats)
            probs = logits.softmax(dim=1).squeeze(0)

        conf, idx = torch.max(probs, dim=0)
        topk_conf, topk_idx = torch.topk(probs, k=min(top_k, probs.numel()))
        topk = [(i.item(), c.item()) for i, c in zip(topk_idx, topk_conf)]
        return idx.item(), conf.item(), topk

# Single global predictor (thread-safe for inference)
PREDICTOR = Predictor(BACKBONE_NAME, BACKBONE_PATH, HEAD_PATH)

# --------------------------- Helpers ---------------------------
def read_image_from_request() -> Image.Image:
    """
    Accepts:
      - multipart/form-data with key 'image' (recommended), or
      - application/json with base64 field 'image_b64'
    """
    if request.files and 'image' in request.files:
        file = request.files['image']
        img_bytes = file.read()
        return Image.open(io.BytesIO(img_bytes))

    if request.is_json:
        data = request.get_json(silent=True) or {}
        b64 = data.get('image_b64')
        if b64:
            try:
                img_bytes = base64.b64decode(b64, validate=True)
                return Image.open(io.BytesIO(img_bytes))
            except Exception as e:
                raise ValueError(f"Invalid base64 image: {e}")

    raise ValueError("No image provided. Use multipart field 'image' or JSON field 'image_b64'.")

def format_topk(topk: List[Tuple[int, float]]) -> List[Dict]:
    return [
        {
            "class_index": idx,
            "class_name": CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else "unknown",
            "confidence": float(conf)
        } for idx, conf in topk
    ]

# --------------------------- Routes ---------------------------
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

@app.route('/metadata', methods=['GET'])
def metadata():
    return jsonify({
        "model": BACKBONE_NAME,
        "device": DEVICE,
        "classes": CLASS_NAMES,
        "input_size": PREDICTOR.input_size,
        "top_k_default": TOP_K
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        img = read_image_from_request()
        idx, conf, topk = PREDICTOR.predict(img, top_k=TOP_K)
        resp = {
            "class_index": idx,
            "class_name": CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else "unknown",
            "confidence": float(conf),
            #"topk": format_topk(topk)
        }
        return jsonify(resp), 200
    except ValueError as ve:
        logger.warning(f"Bad request: {ve}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.exception("Prediction failed")
        return jsonify({"error": "Internal server error", "detail": str(e)}), 500

# --------------------------- Main ---------------------------
#if __name__ == '__main__':
    # For local dev only. In production run with gunicorn/uvicorn, e.g.:
    # gunicorn -w 2 -k gthread -t 120 -b 0.0.0.0:5001 app:app
#    app.run(host='0.0.0.0', port=5001, debug=True, threaded=True)
