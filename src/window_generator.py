import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

class WindowGenerator:
    def __init__(self, input_width, label_width, shift, train_df, val_df, test_df, label_columns=None):
        """
        Inizializza un oggetto WindowGenerator per la gestione delle finestre di input.

        Args:
            input_width (int): La larghezza della finestra di input.
            label_width (int): La larghezza della finestra delle etichette (valori target).
            shift (int): Lo spostamento tra l'input e l'etichetta.
            train_df (pd.DataFrame): Il dataset di addestramento.
            val_df (pd.DataFrame): Il dataset di validazione.
            test_df (pd.DataFrame): Il dataset di test.
            label_columns (list): Una lista delle colonne che contengono le etichette (target).
        """
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.df = pd.concat([self.train_df, self.val_df, self.test_df], axis=0).reset_index(drop=True)
        
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.label_columns = label_columns

        self.total_window_size = input_width + shift
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_window(self, features):
        """
        Divide le finestre in input e label.

        Args:
            features (np.ndarray): I dati delle caratteristiche su cui costruire la finestra.

        Returns:
            tuple: input_window, label_window
        """
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.train_df.columns.get_loc(name)] for name in self.label_columns],
                axis=-1
            )

        return inputs, labels

    def make_dataset(self, data, shuffle=True, batch_size=32):
        """
        Converte un dataframe in un dataset TensorFlow.

        Args:
            data (pd.DataFrame): Il dataframe da convertire.
            shuffle (bool): Se si vuole eseguire lo shuffle sui dati.
            batch_size (int): La dimensione del batch per l'addestramento.

        Returns:
            tf.data.Dataset: Dataset pronto per l'addestramento o la validazione.
        """
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=shuffle,
            batch_size=batch_size
        )

        ds = ds.map(self.split_window)
        return ds

    def plot(self, model=None, plot_col='feature1', max_subplots=3):
        """
        Funzione per plottare le finestre e i risultati del modello.

        Args:
            model (tf.keras.Model): Il modello da utilizzare per fare previsioni.
            plot_col (str): La colonna da plottare.
            max_subplots (int): Il numero massimo di subplot da plottare.
        """
        import matplotlib.pyplot as plt

        inputs, labels = next(iter(self.make_dataset(self.train_df)))
        plt.figure(figsize=(12, 8))

        plot_col_index = self.train_df.columns.get_loc(plot_col)
        max_n = min(max_subplots, len(inputs))

        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel(f'{plot_col} [norm]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index], label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns.index(plot_col)
            else:
                label_col_index = plot_col_index

            plt.scatter(self.label_indices, labels[n, :, label_col_index], edgecolors='k', label='Labels', c='#2ca02c', s=64)

            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index], marker='X', edgecolors='k', label='Predictions', c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')
        plt.show()

    def compile_and_fit(self, model, patience=2, max_epochs=20):
        """
        Compila e addestra il modello con i dati di training e di validazione.

        Args:
            model (tf.keras.Model): Il modello da addestrare.
            patience (int): Il numero di epoche da aspettare prima di interrompere in caso di stallo.
            max_epochs (int): Il numero massimo di epoche da eseguire.

        Returns:
            tf.keras.callbacks.History: La storia dell'addestramento.
        """
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min')

        model.compile(loss='mse',
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=['mae'])

        history = model.fit(
            self.make_dataset(self.train_df),
            validation_data=self.make_dataset(self.val_df),
            epochs=max_epochs,
            callbacks=[early_stopping]
        )

        return history

# Esempio di utilizzo
if __name__ == "__main__":
    # Creazione di dati di esempio
    train_data = pd.DataFrame(np.random.randn(100, 3), columns=['feature1', 'feature2', 'feature3'])
    val_data = pd.DataFrame(np.random.randn(50, 3), columns=['feature1', 'feature2', 'feature3'])
    test_data = pd.DataFrame(np.random.randn(50, 3), columns=['feature1', 'feature2', 'feature3'])

    # Creazione di un oggetto WindowGenerator
    window = WindowGenerator(input_width=24, label_width=1, shift=1, train_df=train_data, val_df=val_data, test_df=test_data)

    # Esempio di utilizzo della funzione plot
    window.plot(plot_col='feature1')

    # Esempio di modello semplice
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(32, return_sequences=False),
        tf.keras.layers.Dense(1)
    ])

    # Compilazione e addestramento del modello
    history = window.compile_and_fit(model)
