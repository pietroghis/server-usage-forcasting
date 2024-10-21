import tensorflow as tf
import numpy as np
import pandas as pd
from dataset import DatasetCreator
from src.window_generator import WindowGenerator
from src.models.multi_step_model import MultiStepModel

class PredictModel:
    def __init__(self, model, window_generator):
        """
        Inizializza la classe per le previsioni.

        Args:
            model (tf.keras.Model): Il modello addestrato.
            window_generator (WindowGenerator): Generatore di finestre temporali per pre-elaborare i dati.
        """
        self.model = model
        self.window_generator = window_generator

    def make_prediction(self, dataset):
        """
        Effettua le previsioni sull'intero dataset fornito.

        Args:
            dataset (pd.DataFrame): Dataset di input.
        
        Returns:
            dict: Previsioni del modello per ogni colonna di interesse.
        """
        # Converti il dataframe in dataset compatibile con il modello
        data = self.window_generator.make_dataset(dataset, shuffle=False)

        # Previsioni del modello
        predictions = self.model.predict(data)

        # Creazione di un dizionario per le previsioni per ogni colonna
        prediction_dict = {}
        columns = ['cpu_usage', 'ram_usage', 'bandwidth_usage', 'power_usage', 'cpu_temperature']

        for i, col in enumerate(columns):
            prediction_dict[col] = predictions[:, -1, i]  # Previsioni per l'ultima ora

        return prediction_dict

    def predict_next_hour(self, full_dataset):
        """
        Prevede l'ora successiva basata sull'ultimo dato disponibile del dataset per tutte le colonne.

        Args:
            full_dataset (pd.DataFrame): Il dataset completo.

        Returns:
            dict: Previsioni per l'ora successiva per ogni colonna indicata.
        """
        # Estrarre l'ultimo blocco di dati (finestra temporale) dal dataset completo
        last_data = full_dataset.iloc[-self.window_generator.total_window_size:]

        # Effettua la predizione usando l'ultimo blocco di dati
        predictions = self.make_prediction(last_data)

        return predictions

    def plot_predictions(self, plot_col='cpu_usage', max_subplots=3):
        """
        Funzione per plottare le finestre e i risultati delle previsioni del modello.

        Args:
            plot_col (str): La colonna da plottare.
            max_subplots (int): Numero massimo di subplot da plottare.
        """
        self.window_generator.plot(model=self.model, plot_col=plot_col, max_subplots=max_subplots)


# Esempio di utilizzo
if __name__ == "__main__":
    folder_path = "/content/drive/MyDrive/server_dataset"  # Percorso al dataset
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

    model = tf.keras.models.load_model('path')

    # Creare il modello PredictModel con la finestra temporale configurata
    predictor = PredictModel(model, window)

    # Prevedere l'ora successiva per ogni colonna
    next_hour_predictions = predictor.predict_next_hour(full_df)

    # Stampa le previsioni
    print(f"Previsioni per l'ora successiva: {next_hour_predictions}")
