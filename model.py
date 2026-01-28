# backend/model.py
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image

# -------- Device --------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- Model Class --------
class CleanEnhancedMobileNetV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "mobilenetv3_large_100",
            pretrained=True,
            num_classes=0,
            global_pool="avg"
        )
        feat_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(768, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(384, 1)
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x).squeeze(1)

# -------- Preprocessing --------
preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# -------- Load Model --------
def load_model(path="mobilenetv3_minimal_lr_bump.pth"):
    checkpoint = torch.load(path, map_location=DEVICE)
    model = CleanEnhancedMobileNetV3().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

# -------- Predict --------
def predict(model, image: Image.Image):
    x = preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logit = model(x)
        prob = torch.sigmoid(logit).item()  # probability of malignant
    pred_class = "Malignant" if prob > 0.5 else "Benign"
    return {"prediction": pred_class, "confidence": round(prob*100, 2)}