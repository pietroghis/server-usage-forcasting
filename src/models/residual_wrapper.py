import os
import pandas as pd
import tensorflow as tf
from src.prediction.prediction import PredictModel
from src.window_generator import WindowGenerator
from dataset import DatasetCreator  # Assicurati che la classe DatasetCreator sia implementata correttamente

class ResidualWrapper(tf.keras.Model):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def call(self, inputs, *args, **kwargs):
        delta = self.model(inputs, *args, **kwargs)
        # La previsione per ogni passo temporale è l'input dal passo precedente più la delta calcolata dal modello
        return inputs + delta


# Esempio di utilizzo
if __name__ == "__main__":
    # Sostituisci con il percorso della tua cartella
    folder_path = "/content/drive/MyDrive/server_dataset"

    # Utilizza la classe DatasetCreator per caricare e preparare il dataset
    dataset_creator = DatasetCreator(folder_path)
    
    # Suddividi il dataset in training, validation e test
    train_df, val_df, test_df = dataset_creator.split_dataset()

    # Ottieni il numero di caratteristiche del dataset
    num_features = train_df.shape[1]  # Numero di caratteristiche di output

    # Inizializza il WindowGenerator
    window = WindowGenerator(input_width=24, label_width=1, shift=1, 
                             train_df=train_df, val_df=val_df, test_df=test_df)

    # Crea il modello ResidualWrapper
    residual_model = ResidualWrapper(
        tf.keras.Sequential([
            tf.keras.layers.LSTM(32, return_sequences=True),
            tf.keras.layers.Dense(
                num_features,
                # Le delte predette dovrebbero iniziare piccole, quindi inizializziamo lo strato di output con zeri.
                kernel_initializer=tf.initializers.zeros())
        ])
    )

    # Addestramento del modello
    history = window.compile_and_fit(residual_model)

    # Valutazione del modello
    val_performance = residual_model.evaluate(window.make_dataset(window.val_df))
    test_performance = residual_model.evaluate(window.make_dataset(window.test_df))

    print(f"Prestazioni sul dataset di validazione: {val_performance}")
    print(f"Prestazioni sul dataset di test: {test_performance}")

    # Crea un'istanza della classe PredictModel
    predictor = PredictModel(residual_model, window)

    # Effettua previsioni sul dataset di test
    test_dataset = window.make_dataset(test_df)
    predictions = predictor.make_prediction(test_dataset)

    # Traccia le previsioni con la funzione plot predefinita del window_generator
    predictor.plot_predictions(plot_col='feature1')  # Sostituisci 'feature1' con il nome della tua colonna
