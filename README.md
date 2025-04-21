# Deep-Learning-for-Land-Classification
# Deep Learning Model for Land Classification

Welcome to the **Deep Learning Model for Land Classification** repository! This project leverages deep learning techniques to classify land cover types using Landsat satellite imagery and various environmental indices. It demonstrates a powerful workflow for processing remote sensing data and applying a Convolutional Neural Network (CNN) to classify land types into multiple categories.

If you're passionate about remote sensing, satellite imagery, and machine learning, you're in the right place! Let's dive in and learn how we can use cutting-edge AI to understand our planet better.

## 🚀 Features

- **Data Preprocessing**: Import and process satellite imagery using `rasterio` and EarthPy.
- **Feature Engineering**: Use various environmental indices (e.g., NDVI, NDWI) alongside satellite bands for better classification performance.
- **Deep Learning Model**: Train a Convolutional Neural Network (CNN) to classify land cover into multiple categories.
- **Visualization**: Visualize the classified results using custom color palettes and plot the results on the map.
- **Prediction on New Data**: Use the trained model to predict land cover classes on new satellite images.
- **Confusion Matrix and Classification Report**: Evaluate model performance with detailed metrics.

## 🛠️ Requirements

Before running the code, make sure to install the necessary Python packages:

```bash
pip install rasterio
pip install earthpy
pip install keras
pip install scikit-learn
pip install matplotlib
```

## 📁 Project Structure

```
├── deep_learning.ipynb           # Main notebook with code implementation
├── Landsat_Jambi_2023.tif        # Example Landsat satellite image for classification
├── Samples_LC_Jambi_2023.csv     # Sample CSV file containing land cover labels and features
└── README.md                     # This file
```

## 🧑‍💻 How to Use

### 1. **Install Dependencies**
Start by installing the required libraries. You can use the following commands:

```bash
pip install rasterio earthpy keras scikit-learn matplotlib
```

### 2. **Mount Google Drive**
The project assumes you're working with Google Colab. You'll need to mount your Google Drive to access the data. Run the following code:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 3. **Load Satellite Data**
The Landsat satellite image and sample CSV file containing land cover labels and features are loaded from Google Drive.

```python
image = rasterio.open('/content/drive/MyDrive/DL/Landsat_Jambi_2023.tif')
samples = pd.read_csv('/content/drive/MyDrive/DL/Samples_LC_Jambi_2023.csv')
```

### 4. **Preprocess the Data**
Shuffle and split the data into training and test datasets. The features and labels are also prepared for model training.

```python
train = samples[samples['sample'] == 'train']
test = samples[samples['sample'] == 'test']
```

### 5. **Build the Model**
A Convolutional Neural Network (CNN) is created using Keras to classify the land cover based on satellite image features.

```python
model = Sequential([
    Conv1D(64, 2, activation='relu', input_shape=(train_shape[1], 1)),
    MaxPooling1D(2),
    Dropout(0.2),
    GlobalMaxPooling1D(),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(6, activation='softmax')
])
```

### 6. **Train the Model**
Train the model using the training data and monitor the validation accuracy.

```python
history = model.fit(train_input, train_output, epochs=100, validation_data=(test_input, test_output))
```

### 7. **Evaluate Performance**
After training, evaluate the performance of the model using confusion matrices and classification reports to understand how well the model is doing.

```python
cm = confusion_matrix(label, prediction, normalize='true')
ConfusionMatrixDisplay(cm).plot()
```

### 8. **Make Predictions**
The trained model is used to predict land cover classes on new satellite images.

```python
prediction = model.predict(image_input)
```

### 9. **Visualize Results**
The results are visualized in a color-coded map.

```python
ep.plot_bands(prediction, cmap=cmap, norm=norm, figsize=(8, 8))
```

### 10. **Save the Predicted Image**
Finally, the predicted image is saved back to Google Drive.

```python
new_dataset = rasterio.open(location, mode='w', driver='GTiff', height=prediction.shape[0], width=prediction.shape[1], count=1, dtype=str(prediction.dtype), crs=crs, transform=transform)
new_dataset.write(prediction, 1)
new_dataset.close()
```

## 📊 Visualizations

Here’s a sample of what you can expect:

- **Original Satellite Image:**
  ![Satellite Image](Jambi_2023.jpg)

- **Predicted Land Cover Classes:**
  Visualize the classification results with the predicted land cover types on the map.

## 📝 Evaluation Metrics

The model performance is evaluated with:

- **Confusion Matrix**: Visualize the true vs. predicted land cover types.
- **Classification Report**: Precision, Recall, and F1-Score for each class.

## 🤖 Model Architecture

The model architecture consists of multiple Conv1D layers, max pooling, dropout layers for regularization, and a final softmax layer for classification. The key hyperparameters include:

- **Number of filters**: 64, 128
- **Kernel size**: 2
- **Dropout rate**: 0.2
- **Optimizer**: Adam
- **Loss function**: Categorical Cross-Entropy

## 🤩 Let's Get Started

Whether you're exploring remote sensing, satellite imagery, or machine learning, this repository offers a hands-on approach to land cover classification with deep learning. You can extend this project to include more advanced models, try with different datasets, or integrate additional satellite features for even better accuracy!

Let’s make the world more intelligent and sustainable—one pixel at a time. 🌍🚀

---

We hope you enjoy this project. Happy coding! 🎉
