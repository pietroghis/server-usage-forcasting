### **Integrated Project Description**

This project consists of multiple modules and models to handle time-series forecasting of server resources. Each component contributes to building a system capable of ingesting data, preparing it for training, and making accurate predictions using advanced machine learning models. Below is a detailed explanation of each component and its role.

---

### **Dataset and Preparation**

#### **`dataset.py`**:
- **Purpose**:
  - This module handles loading and preprocessing of raw data stored in CSV files.
- **Functionality**:
  - Combines all CSV files into a single `DataFrame`.
  - Splits the combined data into training, validation, and test datasets.
  - Normalizes the data based on training statistics (mean and standard deviation).
- **Usage**:
  - Initialize the class with the folder containing CSV files.
  - Call methods like `split_dataset()` and `normalize()` to prepare the data.
- **Outputs**:
  - Three datasets: `train_df`, `val_df`, and `test_df`.

---

### **Data Windowing**

#### **`window_generator.py`**:
- **Purpose**:
  - Manages the creation of sliding windows of data for feeding into time-series models.
- **Functionality**:
  - Converts time-series data into sequences of input and label pairs.
  - Provides utilities for splitting windows into inputs (features) and labels (targets).
  - Supports visualization of predictions alongside true values.
  - Facilitates model training with built-in `compile_and_fit()` functionality.
- **Usage**:
  - Initialize with `input_width`, `label_width`, `shift`, and the prepared datasets.
  - Call methods like `make_dataset()` to create TensorFlow datasets.
  - Use `plot()` to visualize predictions.

---

### **Models**

#### **1. Single-Step LSTM (`single_step_model.py`)**
- **Purpose**:
  - Predicts the next value of a single feature based on historical data.
- **Architecture**:
  - LSTM layer for learning temporal patterns.
  - Dense layer for generating a single-step output.
- **Use Case**:
  - Predicts the next CPU utilization or a similar metric.
- **How to Use**:
  - Train using sliding windows of data (`WindowGenerator`).
  - Evaluate on validation and test sets for performance metrics.

---

#### **2. Multi-Step LSTM (`multi_step_model.py`)**
- **Purpose**:
  - Predicts multiple future values of a feature in one go.
- **Architecture**:
  - LSTM cell for handling temporal data.
  - Dense layer for predicting multiple future steps.
- **Use Case**:
  - Forecasts a series of future resource utilizations (e.g., the next 24 hours).
- **How to Use**:
  - Train on datasets with appropriate input and label windows.
  - Use the `Feedback` model class for iterative predictions.

---

#### **3. Residual LSTM (`residual_wrapper.py`)**
- **Purpose**:
  - Enhances forecasting by predicting deltas (changes) instead of raw values.
  - Adds a residual connection to improve the learning process for time-series data.
- **Architecture**:
  - LSTM layers for capturing temporal dynamics.
  - Dense layer for calculating residuals (adjustments to the previous value).
- **Use Case**:
  - Refines predictions for long-term forecasting.
- **How to Use**:
  - Use the `ResidualWrapper` to wrap around a base LSTM model.
  - Train and evaluate using the `WindowGenerator`.

---

### **Workflow**

1. **Data Ingestion**:
   - Use `DatasetCreator` from `dataset.py` to load and normalize data.
   - Split data into training, validation, and test datasets.

2. **Window Generation**:
   - Initialize `WindowGenerator` with desired parameters for input and label widths.
   - Create TensorFlow datasets for training and validation.

3. **Model Training**:
   - Choose a model (`SingleStep`, `MultiStep`, or `ResidualWrapper`) based on requirements.
   - Compile and fit the model using the `WindowGenerator`'s utility functions.
   - Monitor metrics like Mean Squared Error (MSE) and Mean Absolute Error (MAE).

4. **Evaluation**:
   - Evaluate trained models on test datasets.
   - Use the `plot()` method in `WindowGenerator` to visualize predictions.

5. **Prediction**:
   - Use trained models to predict future values or sequences.
   - Generate future forecasts based on the last known data point.

---

### **Comparison of Models**

| Model                  | Use Case                            | Strengths                          | Outputs                  |
|------------------------|--------------------------------------|------------------------------------|--------------------------|
| Single-Step LSTM       | Predict the next value              | Simple, efficient for real-time    | Single value            |
| Multi-Step LSTM        | Predict multiple future values      | Suitable for batch forecasting     | Sequence of values      |
| Residual LSTM          | Predict deltas for adjustments      | Improves accuracy for long horizons | Adjusted sequence values |

---

### **How to Use This Project**

1. **Select the Use Case**:
   - For real-time, immediate forecasting: Use **Single-Step LSTM**.
   - For batch predictions over time: Use **Multi-Step LSTM**.
   - For accurate long-term forecasting: Use **Residual LSTM**.

2. **Prepare the Data**:
   - Normalize and split the dataset using `DatasetCreator`.
   - Generate sliding windows of data using `WindowGenerator`.

3. **Train and Evaluate**:
   - Train the model suited to your use case.
   - Evaluate its performance and fine-tune as necessary.

4. **Predict and Visualize**:
   - Use trained models to forecast future metrics.
   - Visualize predictions to validate accuracy and trends.

This modular design ensures flexibility, scalability, and adaptability for different time-series forecasting scenarios.
