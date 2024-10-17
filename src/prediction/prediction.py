import tensorflow as tf
import numpy as np
from src.window_generator import WindowGenerator

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
        Effettua le previsioni sul dataset fornito.

        Args:
            dataset (tf.data.Dataset): Dataset di input (test o validazione).
        
        Returns:
            np.array: Previsioni del modello.
        """
        predictions = self.model.predict(dataset)
        return predictions

    def plot_predictions(self, plot_col='feature1', max_subplots=3):
        """
        Funzione per plottare le finestre e i risultati delle previsioni del modello.

        Args:
            plot_col (str): La colonna da plottare.
            max_subplots (int): Numero massimo di subplot da plottare.
        """
        self.window_generator.plot(model=self.model, plot_col=plot_col, max_subplots=max_subplots)