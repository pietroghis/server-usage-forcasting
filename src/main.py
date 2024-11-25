# main.py
from fastapi import FastAPI
import tensorflow as tf

from dataset import DatasetCreator
from models.multi_step_model import MultiStep
from models.residual_wrapper import ResidualWrapper
from models.single_step_model import SingleStep
from prediction.prediction import PredictModel
from window_generator import WindowGenerator

app = FastAPI()

# Load models (latest version)
multi_step_model = tf.keras.models.load_model("models/multi_step_model")
residual_model = tf.keras.models.load_model("models/residual_model")
single_step_model = tf.keras.models.load_model("models/single_step_model")

@app.get("/health")
async def health_check():
    return {"status": "API is healthy and models are loaded"}

@app.post("/train/{model_name}")
async def train_model(model_name: str):
    folder_path = "/content/drive/MyDrive/server_dataset"
    
    dataset_creator = DatasetCreator(folder_path)

    # Caricare e preparare il dataset completo
    full_df = dataset_creator.create_combined_dataframe()

    # Normalizzare il dataset completo
    full_df, _ = dataset_creator.normalize(full_df, full_df)

    train_df, val_df, test_df = dataset_creator.split_dataset()

    # Ottieni il numero di caratteristiche del dataset
    num_features = train_df.shape[1]  # Numero di caratteristiche di output
    
    OUT_STEPS = 24

    # Inizializza il WindowGenerator
    window = WindowGenerator(input_width=24, label_width=OUT_STEPS, shift=1, 
                             train_df=train_df, val_df=val_df, test_df=test_df)

    if model_name == "multi_step":
        model = MultiStep(multi_step_model, window)

    elif model_name == "residual":
        model = ResidualWrapper(residual_model, window)

    elif model_name == "single_step":
        model = SingleStep(single_step_model, window)
    else:
        return {"error": "Invalid model name"}

    window.compile_and_fit(model)

    val_performance = model.evaluate(window.make_dataset(window.val_df))
    test_performance = model.evaluate(window.make_dataset(window.test_df))

    return { 
        "val_performance": f"Prestazioni sul dataset di validazione: {val_performance}",
        "test_performance":f"Prestazioni sul dataset di test: {test_performance}"
        }

@app.post("/predict/{model_name}")
async def predict(model_name: str, features: dict):
    data = features["data"]

    dataset_creator = DatasetCreator(json_data = data)
    
    # Caricare e preparare il dataset completo
    full_df = dataset_creator.create_combined_dataframe()

    # Normalizzare il dataset completo
    full_df, _ = dataset_creator.normalize(full_df, full_df)

    train_df, val_df, test_df = dataset_creator.split_dataset()
    
    OUT_STEPS = 24

    # Inizializza il WindowGenerator
    window = WindowGenerator(input_width=24, label_width=OUT_STEPS, shift=1, 
                             train_df=train_df, val_df=val_df, test_df=test_df)

    if model_name == "multi_step":
        predictor = PredictModel(multi_step_model, window)

    elif model_name == "residual":
        predictor = PredictModel(residual_model, window)

    elif model_name == "single_step":
        predictor = PredictModel(single_step_model, window)
    else:
        return {"error": "Invalid model name"}
   
    # Prevedere l'ora successiva per ogni colonna
    next_hour_predictions = predictor.predict_next_hour(full_df)

    return {"predictions": next_hour_predictions.tolist()}
