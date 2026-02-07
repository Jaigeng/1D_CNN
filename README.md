# 1D Convolutional Neural Network for Time Series Classification

This repository presents a **One Dimensional Convolutional Neural Network (1D CNN)** implemented using **TensorFlow/Keras** for **time series classification**.  


## Datasets
The proposed 1D CNN architecture is suitable for a wide range of time series classification tasks, including but not limited to:
- Signal classification (accelerometer, gyroscope, EEG, ECG)
- Signal monitoring 
- Temporal pattern recognition

## Environment

| Tool / Library | Description |
|---------------|-------------|
| Python 3.6 | Programming language |
| TensorFlow 2.0.0 | Deep learning framework |
| Keras API | High level neural network construction |
| NumPy | Numerical computation |
| Pandas | Data preprocessing and labeling |
| Scikit-learn | Dataset splitting and normalization |
| Matplotlib / Seaborn | Visualization of training results |

Installation:
```bash
pip install tensorflow numpy pandas scikit-learn matplotlib
```

## Input Data Representation
```text
(samples, time_steps, features)
```

```text
timestamp | feature_1 | feature_2 | feature_3
```

## Class Label Encoding
```text
0 → Class 1  
1 → Class 2  
2 → Class 3  
3 → Class 4
```

## Model Training
```python
model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs = 50,
    batch_size = 16
)
```
