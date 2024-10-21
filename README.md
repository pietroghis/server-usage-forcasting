# Server resources usage forcasting

## Introduction

The project propose and insight on server usage forecasting. The purpose is to integrate the system with a web application.

## Requirements

file: requirements.txt

## Models : 

Is possibile to use 3 differents types of models:

- Single step model : src/models/single_step_model.py
- Residual Wrapper (Single step model wrapper for performance) : src/models/residual_wrapper.py
- Multi step model : src/models/multi_step_model.py

## Usage : 

let's to an exemple of usage ( single step model, but it's very similar for each class )

```
# Esempio di utilizzo
if __name__ == "__main__":
    # Sostituisci con il percorso della tua cartella
    folder_path = "/content/drive/MyDrive/server_dataset"

    # Crea il DataFrame combinato
    df = create_combined_dataframe(folder_path)

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

    # Crea il modello SingleStep
    single_step_model = SingleStep(units=32, num_features=num_features)

    # Addestramento del modello
    history = window.compile_and_fit(single_step_model)

    # Valutazione del modello
    val_performance = single_step_model.evaluate(window.make_dataset(window.val_df))
    test_performance = single_step_model.evaluate(window.make_dataset(window.test_df))

    print(f"Prestazioni sul dataset di validazione: {val_performance}")
    print(f"Prestazioni sul dataset di test: {test_performance}")
    
    column_names = df.columns.to_list()

    # Crea un'istanza della classe PredictModel
    predictor = PredictModel(single_step_model, window)

    # Effettua previsioni sul dataset di test
    test_dataset = window.make_dataset(df)
    predictions = predictor.make_prediction(test_dataset)

    # Traccia le previsioni con la funzione plot predefinita del window_generator
    predictor.plot_predictions(plot_col='feature1')  # Sostituisci 'feature1' con il nome della tua colonna
```

