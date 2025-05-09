from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import joblib
import io

# Charger les modèles
feature_extractor = load_model("efficientnet_feature_extractor.h5")
classifier = joblib.load("classifier_MLP.pkl")

# Liste des noms de labels
label_names = [
    "Saine",
    "Mildiou",
    "Mineuse des feuilles",
    "Carence en magnésium",
    "Virus de la maladie tachetée"
]

# Configuration de l'application FastAPI
app = FastAPI()

# Autoriser les requêtes CORS (utile pour Flutter)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prétraitement de l'image
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))  # Adapter à EfficientNet
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Endpoint de prédiction
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = preprocess_image(image_bytes)

        features = feature_extractor.predict(image)
        prediction = classifier.predict(features).toarray()[0]

        predicted_labels = [label_names[i] for i, val in enumerate(prediction) if val == 1]

        return JSONResponse(content={
            "binary_prediction": prediction.tolist(),
            "predicted_labels": predicted_labels
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
