import os
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from src.window_generator import WindowGenerator


import tensorflow as tf
from src.window_generator import WindowGenerator

class FeedBack(tf.keras.Model):
    def __init__(self, units, out_steps, num_features):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
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
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        # Initialize the LSTM state.
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


def create_combined_dataframe(folder_path):
    all_data = pd.DataFrame()

    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path, sep=';')  # Assumiamo che il separatore sia ';'
            all_data = pd.concat([all_data, df], ignore_index=True)

    return all_data


# Esempio di utilizzo
if __name__ == "__main__":
    # Sostituisci con il percorso della tua cartella
    folder_path = "/content/drive/MyDrive/server_dataset"

    # Crea il DataFrame combinato
    df = create_combined_dataframe(folder_path)

    # Droppare colonne non necessarie
    columns_to_drop = ['\tCPU cores', '\tCPU capacity provisioned [MHZ]', 
                       '\tMemory capacity provisioned [KB]', 
                       '\tDisk write throughput [KB/s]', 
                       '\tDisk read throughput [KB/s]', 
                       '\tNetwork received throughput [KB/s]', 
                       '\tNetwork transmitted throughput [KB/s]', 
                       '\tCPU usage [MHZ]', '\tMemory usage [KB]']

    df.drop(columns_to_drop, axis=1, inplace=True)

    # Convertire il timestamp e preparare i dati
    date_time = pd.to_datetime(df.pop('Timestamp [ms]'), unit='ms')
    timestamp_s = date_time.map(pd.Timestamp.timestamp)
    column_indices = {name: i for i, name in enumerate(df.columns)}

    n = len(df)
    train_df = df[0:int(n*0.7)]
    val_df = df[int(n*0.7):int(n*0.9)]
    test_df = df[int(n*0.9):]

    num_features = df.shape[1]  # Numero di caratteristiche di output
    print(len(train_df))
    
    window = WindowGenerator(input_width=24, label_width=3, shift=1, train_df=train_df, val_df=val_df, test_df=test_df)

    OUT_STEPS = 3  # Numero di passi futuri da prevedere

    # Crea il modello FeedBack
    feedback_model = FeedBack(units=32, out_steps=OUT_STEPS, num_features=num_features)

    # Addestramento del modello
    history = window.compile_and_fit(feedback_model)

    # Valutazione del modello
    val_performance = feedback_model.evaluate(window.make_dataset(window.val_df))
    test_performance = feedback_model.evaluate(window.make_dataset(window.test_df))

    print(f"Prestazioni sul dataset di validazione: {val_performance}")
    print(f"Prestazioni sul dataset di test: {test_performance}")
