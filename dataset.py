import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

class DatasetCreator:
    def __init__(self, train_df, test_df):
        """
        Inizializza il creatore del dataset con i dataframe di train e test.
        
        Args:
            train_df (pd.DataFrame): Il dataset di addestramento.
            test_df (pd.DataFrame): Il dataset di test.
        """
        self.train_df = train_df
        self.test_df = test_df

    def normalize(self):
        """
        Normalizza i dataset di addestramento e di test usando la media e la deviazione standard del dataset di addestramento.
        
        Returns:
            tuple: Il dataframe di addestramento e di test normalizzati.
        """
        train_mean = self.train_df.mean()
        train_std = self.train_df.std()

        self.train_df = (self.train_df - train_mean) / train_std
        self.test_df = (self.test_df - train_mean) / train_std

        return self.train_df, self.test_df

    def get_dataset_stats(self):
        """
        Restituisce le statistiche del dataset di addestramento, come la media e la deviazione standard.
        
        Returns:
            tuple: La media e la deviazione standard del dataset di addestramento.
        """
        train_mean = self.train_df.mean()
        train_std = self.train_df.std()
        
        return train_mean, train_std

# Esempio di utilizzo:
if __name__ == "__main__":
    # Caricamento di dati di esempio (da adattare ai tuoi dati reali)
    train_data = {'feature1': [1, 2, 3, 4, 5], 'feature2': [10, 20, 30, 40, 50]}
    test_data = {'feature1': [6, 7, 8, 9, 10], 'feature2': [60, 70, 80, 90, 100]}

    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    # Creazione del dataset
    dataset_creator = DatasetCreator(train_df, test_df)
    
    # Normalizzazione
    normalized_train, normalized_test = dataset_creator.normalize()

    print("Train Data Normalizzato:\n", normalized_train)
    print("Test Data Normalizzato:\n", normalized_test)

    # Statistiche del dataset
    train_mean, train_std = dataset_creator.get_dataset_stats()
    print("\nMedia Train Dataset:\n", train_mean)
    print("Deviazione Standard Train Dataset:\n", train_std)
