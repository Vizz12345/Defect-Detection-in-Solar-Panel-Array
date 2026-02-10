import io
import time
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import googlenet, GoogLeNet_Weights
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# --- 1. CONFIGURATION & CLASS MAPPINGS ---
RGB_CLASSES = [
    "Bird-drop", "Clean", "Dusty",
    "Electrical-damage", "Physical-Damage", "Snow-Covered"
]

EL_CLASSES = [
    "crack", "finger", "black_core", "thick_line", "star_crack",
    "corner", "fragment", "scratch", "horizontal_dislocation",
    "vertical_dislocation", "printing_error", "short_circuit",
    "good", "material"
]

THERMAL_CLASSES = [
    "MultiByPassed", "MultiDiode", "MultiHotSpot",
    "SingleByPassed", "SingleDiode", "SingleHotSpot",
    "StringOpenCircuit", "StringReversedPolarity"
]

models_registry = {}
class_registry = {
    "RGB": RGB_CLASSES,
    "EL": EL_CLASSES,
    "Thermal": THERMAL_CLASSES
}

# --- 2. MODEL LOADERS ---

def get_googlenet_model(path, num_classes):
    model = googlenet(weights=GoogLeNet_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Handle Aux classifiers
    # model.aux1.fc2 = nn.Linear(model.aux1.fc2.in_features, num_classes)
    # model.aux2.fc2 = nn.Linear(model.aux2.fc2.in_features, num_classes)

    try:
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded GoogLeNet from {path}")
    except Exception as e:
        print(f"Error loading GoogLeNet ({path}): {e}")

    model.eval()
    return model

def get_efficientnet_v2_s_model(path, num_classes):
    """ Used for EL images """
    # model = models.efficientnet_v2_s(weights=None)
    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    try:
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        print(f"Loaded EfficientNetV2-S from {path}")
    except Exception as e:
        print(f"Error loading EfficientNetV2-S ({path}): {e}")

    model.eval()
    return model

def get_efficientnet_v2_m_model(path, num_classes):
    """
    Used for Thermal images.
    Based on logs: Stem=24, Head=1280, Depth > Small.
    Matches EfficientNetV2-M.
    """
    model = models.efficientnet_v2_m(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    try:
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        print(f"Loaded EfficientNetV2-M from {path}")
    except Exception as e:
        print(f"Error loading EfficientNetV2-M ({path}): {e}")

    model.eval()
    return model

# --- 3. LIFESPAN (STARTUP) ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    # RGB -> GoogLeNet
    models_registry["RGB"] = get_googlenet_model("../Models/googlenet_rgb.pth", len(RGB_CLASSES))

    # EL -> EfficientNetV2-S
    models_registry["EL"] = get_efficientnet_v2_s_model(
        "../Models/efficientnet_el_full.pth",
        len(EL_CLASSES)
    )

    # Thermal -> EfficientNetV2-M (Updated)
    models_registry["Thermal"] = get_efficientnet_v2_m_model(
        "../Models/efficientnet_best_thermal_model.pth",
        len(THERMAL_CLASSES)
    )

    yield
    models_registry.clear()

app = FastAPI(lifespan=lifespan)

# --- 4. MIDDLEWARE & PREPROCESSING ---

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 5. PREDICTION ENDPOINT ---

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    image_type: str = Form(...)
):
    if image_type not in models_registry:
        return {"error": "Invalid image type or model not loaded"}

    start_time = time.time()

    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        input_tensor = transform_pipeline(image).unsqueeze(0)

        selected_model = models_registry[image_type]
        class_names = class_registry[image_type]

        with torch.no_grad():
            outputs = selected_model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            print(probabilities)
            confidence, predicted_idx = torch.max(probabilities, 0)

        inference_time = time.time() - start_time

        predicted_label = class_names[predicted_idx.item()]

        # Determine architecture name for the UI
        if image_type == "RGB":
            arch_name = "GoogLeNet"
        elif image_type == "Thermal":
            arch_name = "EfficientNetV2-M"
        else:
            arch_name = "EfficientNetV2-S"

        return {
            "label": predicted_label,
            "confidence": f"{confidence.item():.2f}",
            "inference": f"{inference_time:.3f}s",
            "modelUsed": f"{arch_name} ({image_type})"
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
