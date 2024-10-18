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
         # Verifica che il dataset non sia vuoto
        try:
         inputs, _ = next(iter(dataset))
        except StopIteration:
         raise ValueError("Il dataset è vuoto o non contiene dati sufficienti per le previsioni.")
    
        predictions = self.model.predict(dataset)
        return predictions

    def calculate_deltas(self, dataset, plot_col='feature1'):
        """
        Calcola il delta (differenza) tra l'ultimo valore del dataset e le previsioni del modello,
        restituisce inoltre l'ultimo valore del dataset e la previsione finale (last_val + delta).

        Args:
            dataset (tf.data.Dataset): Dataset di input (test o validazione).
            plot_col (str): Colonna da utilizzare per il calcolo delle previsioni.

        Returns:
            dict: Un dizionario con i campi:
                - 'delta': Differenza tra predizione e ultimo valore del dataset.
                - 'last_val': L'ultimo valore nel dataset.
                - 'prediction': La previsione finale (last_val + delta).
        """
        try:
         inputs, labels  = next(iter(dataset))
        except StopIteration:
         raise ValueError("Il dataset è vuoto o non contiene dati sufficienti per le previsioni.")
        
        # Ottieni indice della colonna da plottare
        plot_col_index = self.window_generator.df.columns.get_loc(plot_col)

        # Previsioni del modello sugli input
        predictions = self.make_prediction(dataset)

        # Estrai l'ultimo valore reale dal dataset
        last_val = inputs[-1, -1, plot_col_index].numpy()  # Ultimo valore nel dataset

        # Estrai la previsione corrispondente
        predicted_delta = predictions[-1, -1, plot_col_index]  # Previsione del modello (delta)

        # Calcola la previsione finale (last_val + delta)
        prediction = last_val + predicted_delta

        # Ritorna il delta, l'ultimo valore e la previsione finale
        return {
            'delta': predicted_delta,
            'last_val': last_val,
            'prediction': prediction
        }

    def plot_predictions(self, plot_col='feature1', max_subplots=3):
        """
        Funzione per plottare le finestre e i risultati delle previsioni del modello.

        Args:
            plot_col (str): La colonna da plottare.
            max_subplots (int): Numero massimo di subplot da plottare.
        """
        self.window_generator.plot(model=self.model, plot_col=plot_col, max_subplots=max_subplots)

