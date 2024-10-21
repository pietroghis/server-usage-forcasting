import os
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from src.prediction.prediction import PredictModel
from src.window_generator import WindowGenerator
from dataset import DatasetCreator  # Assicurati che la classe DatasetCreator sia implementata correttamente

class FeedBack(tf.keras.Model):
    def __init__(self, units, out_steps, num_features):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        # Wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        self.dense = tf.keras.layers.Dense(num_features)

    def warmup(self, inputs):
        """
        Metodo per l'inizializzazione delle previsioni.

        Args:
            inputs (tf.Tensor): Dati di input per la previsione.

        Returns:
            tuple: La previsione e lo stato del LSTM.
        """
        # inputs.shape => (batch, time, features)
        x, *state = self.lstm_rnn(inputs)

        # predictions.shape => (batch, features)
        prediction = self.dense(x)
        return prediction, state

    def call(self, inputs, training=None):
        """
        Definisce la logica del modello per la previsione.

        Args:
            inputs (tf.Tensor): I dati di input per la previsione.
            training (bool): Se il modello è in fase di addestramento.

        Returns:
            tf.Tensor: Le previsioni finali.
        """
        predictions = []
        prediction, state = self.warmup(inputs)

        # Insert the first prediction.
        predictions.append(prediction)

        # Run the rest of the prediction steps.
        for n in range(1, self.out_steps):
            # Use the last prediction as input.
            x = prediction
            # Execute one LSTM step.
            x, state = self.lstm_cell(x, states=state, training=training)
            # Convert the LSTM output to a prediction.
            prediction = self.dense(x)
            # Add the prediction to the output.
            predictions.append(prediction)

        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions


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
    
    OUT_STEPS = 24

    # Inizializza il WindowGenerator
    window = WindowGenerator(input_width=24, label_width=OUT_STEPS, shift=1, 
                             train_df=train_df, val_df=val_df, test_df=test_df)

    # Crea il modello FeedBack
    feedback_model = FeedBack(units=32, out_steps=OUT_STEPS, num_features=num_features)

    # Addestramento del modello
    history = window.compile_and_fit(feedback_model)

    # Valutazione del modello
    val_performance = feedback_model.evaluate(window.make_dataset(window.val_df))
    test_performance = feedback_model.evaluate(window.make_dataset(window.test_df))

    print(f"Prestazioni sul dataset di validazione: {val_performance}")
    print(f"Prestazioni sul dataset di test: {test_performance}")

    # Crea un'istanza della classe PredictModel
    predictor = PredictModel(feedback_model, window)

    # Effettua previsioni sul dataset di test
    test_dataset = window.make_dataset(test_df)
    predictions = predictor.make_prediction(test_dataset)

    # Traccia le previsioni con la funzione plot predefinita del window_generator
    predictor.plot_predictions(plot_col='feature1')  # Sostituisci 'feature1' con il nome della tua colonna
