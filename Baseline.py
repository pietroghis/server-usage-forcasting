import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from window_generator import WindowGenerator

class Baseline(tf.keras.Model):
    def __init__(self, label_index=None):
        """
        Inizializza il modello Baseline che prevede l'ultimo valore noto.

        Args:
            label_index (int): L'indice della colonna da prevedere come output.
        """
        super().__init__()
        self.label_index = label_index

    def call(self, inputs):
        """
        Definisce la logica di previsione del modello Baseline.
        Il modello predice l'ultimo valore dell'input come output.

        Args:
            inputs (tf.Tensor): I dati di input per la previsione.

        Returns:
            tf.Tensor: L'output previsto dal modello.
        """
        if self.label_index is None:
            return inputs[:, -1, :]
        return inputs[:, -1, self.label_index][:, tf.newaxis]

# Esempio di utilizzo
if __name__ == "__main__":
    # Dati di esempio
    train_data = pd.DataFrame(np.random.randn(100, 3), columns=['feature1', 'feature2', 'feature3'])
    val_data = pd.DataFrame(np.random.randn(50, 3), columns=['feature1', 'feature2', 'feature3'])
    test_data = pd.DataFrame(np.random.randn(50, 3), columns=['feature1', 'feature2', 'feature3'])

    # Inizializza il WindowGenerator
    window = WindowGenerator(input_width=24, label_width=1, shift=1, train_df=train_data, val_df=val_data, test_df=test_data)

    # Inizializza il modello Baseline
    baseline_model = Baseline(label_index=0)

    # Compila e addestra il modello Baseline
    history = window.compile_and_fit(baseline_model)

    # Valutazione delle prestazioni del modello
    val_performance = baseline_model.evaluate(window.make_dataset(window.val_df))
    test_performance = baseline_model.evaluate(window.make_dataset(window.test_df))

    print(f"Prestazioni sul dataset di validazione: {val_performance}")
    print(f"Prestazioni sul dataset di test: {test_performance}")
