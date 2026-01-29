import os
import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torch.nn as nn
import timm
import torchvision.transforms as transforms

# ------------------------
# CONFIG
# ------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "mobilenetv3_minimal_lr_bump.pth")
FRONTEND_DIST = os.path.abspath(os.path.join(BASE_DIR, "../frontend/dist"))

IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------
# APP INIT
# ------------------------
app = FastAPI(title="Nevinet Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# MODEL
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


def load_model():
    model = CleanEnhancedMobileNetV3().to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


model = load_model()

# ------------------------
# TRANSFORMS
# ------------------------
transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

# ------------------------
# PREDICTION
# ------------------------
def run_inference(image: Image.Image):
    x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits1 = model(x)
        logits2 = model(torch.flip(x, dims=[3]))
        prob = torch.sigmoid((logits1 + logits2) / 2).item()

    return prob


# ------------------------
# ROUTES (DEFINED FIRST — CRITICAL)
# ------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        prob = run_inference(image)
        label = "malignant" if prob >= 0.5 else "benign"

        return {
            "label": label,
            "probability": prob,
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# ------------------------
# STATIC FILES (MUST BE LAST)
# ------------------------
if os.path.exists(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
else:
    print(f"⚠️ Frontend dist not found at {FRONTEND_DIST}")