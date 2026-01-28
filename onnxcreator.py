# convert_to_onnx.py
# Converts your existing model to ONNX format

import torch
import torch.nn as nn
import timm
import os

# ------------------------
# CONFIG
# ------------------------
IMG_SIZE = 224
MODEL_PATH = "/Users/pranavjaishankar/Desktop/nevinet-inference/backend/mobilenetv3_minimal_lr_bump.pth"
OUTPUT_PATH = "/Users/pranavjaishankar/Desktop/nevinet-inference/nevinet_mobile.onnx"

# ------------------------
# MODEL (same as your app.py)
# ------------------------
class CleanEnhancedMobileNetV3(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        self.backbone = timm.create_model(
            "mobilenetv3_large_100",
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )

        with torch.no_grad():
            dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
            feature_dim = self.backbone(dummy).shape[1]

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x).squeeze(1)

# ------------------------
# LOAD MODEL
# ------------------------
print("📦 Loading model...")
model = CleanEnhancedMobileNetV3(dropout=0.2)
checkpoint = torch.load(MODEL_PATH, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✅ Model loaded from {MODEL_PATH}")
print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
print(f"   Val Acc: {checkpoint.get('val_acc', 'N/A')}")

# ------------------------
# CONVERT TO ONNX
# ------------------------
print(f"\n🔄 Converting to ONNX...")

dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    OUTPUT_PATH,
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
print(f"✅ ONNX model saved: {OUTPUT_PATH}")
print(f"📊 Size: {size_mb:.1f} MB")

# ------------------------
# TEST (optional but recommended)
# ------------------------
print("\n🧪 Testing ONNX model...")

try:
    import onnxruntime as ort
    import numpy as np
    
    ort_session = ort.InferenceSession(OUTPUT_PATH)
    dummy_np = dummy_input.numpy()
    outputs = ort_session.run(None, {'input': dummy_np})
    
    # Apply sigmoid
    prob = 1 / (1 + np.exp(-outputs[0][0]))
    
    print(f"✅ ONNX model works!")
    print(f"   Test output: {outputs[0][0]:.4f}")
    print(f"   Test probability: {prob:.4f}")
except ImportError:
    print("⚠️  onnxruntime not installed (optional for testing)")
    print("   Install with: pip install onnxruntime")

print("\n" + "="*60)
print("🎉 CONVERSION COMPLETE!")
print("="*60)
print(f"📁 ONNX file: {os.path.abspath(OUTPUT_PATH)}")
print("\nReady for mobile app development!")