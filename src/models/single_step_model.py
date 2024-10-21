import os
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from dataset import DatasetCreator
from src.prediction.prediction import PredictModel
from src.window_generator import WindowGenerator

class SingleStep(tf.keras.Model):
    def __init__(self, units, num_features):
        super().__init__()
        self.units = units
        self.lstm = tf.keras.layers.LSTM(units, return_sequences=True),
        self.dense = tf.keras.layers.Dense(units=num_features)

    def call(self, inputs):
        """
        Definisce la logica del modello per la previsione a singolo passo.

        Args:
            inputs (tf.Tensor): I dati di input per la previsione.

        Returns:
            tf.Tensor: La previsione finale.
        """
        x = self.lstm(inputs)
        return self.dense(x)


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

    # Crea il modello SingleStep
    single_step_model = SingleStep(units=32, num_features=num_features)

    # Addestramento del modello
    history = window.compile_and_fit(single_step_model)

    # Valutazione del modello
    val_performance = single_step_model.evaluate(window.make_dataset(window.val_df))
    test_performance = single_step_model.evaluate(window.make_dataset(window.test_df))

    print(f"Prestazioni sul dataset di validazione: {val_performance}")
    print(f"Prestazioni sul dataset di test: {test_performance}")
    
    # Crea un'istanza della classe PredictModel
    predictor = PredictModel(single_step_model, window)

    # Effettua previsioni sul dataset di test
    test_dataset = window.make_dataset(test_df)
    predictions = predictor.make_prediction(test_dataset)

    # Traccia le previsioni con la funzione plot predefinita del window_generator
    predictor.calculate_deltas(test_dataset, plot_col='feature1')  # Sostituisci 'feature1' con il nome della tua colonna