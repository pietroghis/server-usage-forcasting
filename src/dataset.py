import os
import pandas as pd

class DatasetCreator:
    def __init__(self, folder_path = '', json_data = ''):
        """
        Inizializza il creatore del dataset con il percorso della cartella contenente i CSV.
        
        Args:
            folder_path (str): Il percorso della cartella contenente i file CSV.
        """
        self.folder_path = folder_path
        self.json_data = json_data

        if(folder_path == '') : 
            self.df = self.create_dataframe_from_json()
        else:    
            self.df = self.create_combined_dataframe()
        
    def create_combined_dataframe(self):
        """
        Crea un unico DataFrame combinando tutti i file CSV nella cartella specificata.
        
        Returns:
            pd.DataFrame: Il DataFrame combinato.
        """
        all_data = pd.DataFrame()
        for filename in os.listdir(self.folder_path):
            if filename.endswith('.csv'):
                file_path = os.path.join(self.folder_path, filename)
                df = pd.read_csv(file_path, sep=';')  # Assumiamo che il separatore sia ';'
                all_data = pd.concat([all_data, df], ignore_index=True)
        
        # Convertire il timestamp
        date_time = pd.to_datetime(all_data.pop('Timestamp [ms]'), unit='ms')
        all_data['timestamp_s'] = date_time.map(pd.Timestamp.timestamp)
        
        return all_data
    
    def create_dataframe_from_json(self):
        """
        Crea un DataFrame a partire da un JSON fornito.
        
        Returns:
            pd.DataFrame: Il DataFrame creato dal JSON.
        """
        # Creare il DataFrame dai dati JSON
        all_data = pd.DataFrame(self.json_data)
        
        # Convertire il timestamp
        if 'Timestamp [ms]' in all_data.columns:
            date_time = pd.to_datetime(all_data.pop('Timestamp [ms]'), unit='ms')
            all_data['timestamp_s'] = date_time.map(pd.Timestamp.timestamp)
        
        return all_data

    def split_dataset(self):
        """
        Divide il dataset in train, validation e test set.
        
        Returns:
            tuple: DataFrame di training, validation e test.
        """
        n = len(self.df)
        train_df = self.df[0:int(n*0.7)]
        val_df = self.df[int(n*0.7):int(n*0.9)]
        test_df = self.df[int(n*0.9):]
        
        train_df, test_df = self.normalize(train_df=train_df, test_df=test_df)

        return train_df, val_df, test_df

    def normalize(self, train_df, test_df):
        """
        Normalizza i dataset di addestramento e di test usando la media e la deviazione standard del dataset di addestramento.
        
        Args:
            train_df (pd.DataFrame): DataFrame di addestramento.
            test_df (pd.DataFrame): DataFrame di test.
        
        Returns:
            tuple: DataFrame di training e test normalizzati.
        """
        train_mean = train_df.mean()
        train_std = train_df.std()

        train_df = (train_df - train_mean) / train_std
        test_df = (test_df - train_mean) / train_std

        return train_df, test_df

    def get_dataset_stats(self, train_df):
        """
        Restituisce le statistiche del dataset di addestramento, come la media e la deviazione standard.
        
        Args:
            train_df (pd.DataFrame): DataFrame di addestramento.
        
        Returns:
            tuple: La media e la deviazione standard del dataset di addestramento.
        """
        train_mean = train_df.mean()
        train_std = train_df.std()
        
        return train_mean, train_std


# Esempio di utilizzo:
if __name__ == "__main__":
    # Sostituisci con il percorso della tua cartella
    folder_path = "/content/drive/MyDrive/server_dataset"

    # Creazione del dataset
    dataset_creator = DatasetCreator(folder_path)

    # Split del dataset
    train_df, val_df, test_df = dataset_creator.split_dataset()

    # Normalizzazione
    normalized_train, normalized_test = dataset_creator.normalize(train_df, test_df)

    # Statistiche del dataset
    train_mean, train_std = dataset_creator.get_dataset_stats(train_df)
    print("\nMedia Train Dataset:\n", train_mean)
    print("Deviazione Standard Train Dataset:\n", train_std)
