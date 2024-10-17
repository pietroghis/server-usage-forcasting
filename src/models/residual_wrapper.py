import os
import pandas as pd
import tensorflow as tf
from src.prediction.prediction import PredictModel
from src.window_generator import WindowGenerator
from src.dataset import Dataset  # Assicurati che la classe Dataset sia implementata correttamente

class ResidualWrapper(tf.keras.Model):
  def __init__(self, model):
    super().__init__()
    self.model = model

  def call(self, inputs, *args, **kwargs):
    delta = self.model(inputs, *args, **kwargs)

    # The prediction for each time step is the input
    # from the previous time step plus the delta
    # calculated by the model.
    return inputs + delta


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

    # Inizializza il WindowGenerator
    window = WindowGenerator(input_width=24, label_width=1, shift=1, 
                             train_df=train_df, val_df=val_df, test_df=test_df)

    # Crea il modello ResidualWrapper
    residual_model = ResidualWrapper(
    tf.keras.Sequential([
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.Dense(
        num_features,
        # The predicted deltas should start small.
        # Therefore, initialize the output layer with zeros.
        kernel_initializer=tf.initializers.zeros())
    ]))


    # Addestramento del modello
    history = window.compile_and_fit(residual_model)

    # Valutazione del modello
    val_performance = residual_model.evaluate(window.make_dataset(window.val_df))
    test_performance = residual_model.evaluate(window.make_dataset(window.test_df))

    print(f"Prestazioni sul dataset di validazione: {val_performance}")
    print(f"Prestazioni sul dataset di test: {test_performance}")
    
    column_names = df.columns.to_list()

    # Crea un'istanza della classe PredictModel
    predictor = PredictModel(residual_model, window)

    # Effettua previsioni sul dataset di test
    test_dataset = window.make_dataset(window.test_df)
    predictions = predictor.make_prediction(test_dataset)

    # Traccia le previsioni con la funzione plot predefinita del window_generator
    predictor.plot_predictions(plot_col=column_names.pop)  # Sostituisci 'feature1' con il nome della tua colonna
