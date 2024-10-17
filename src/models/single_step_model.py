import os
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
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