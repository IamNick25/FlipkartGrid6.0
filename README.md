# 1 Fruit Freshness Detection Model

## Overview
This project implements a machine learning model to detect the freshness of fruits based on image data. The model is built using a hybrid architecture combining Convolutional Neural Networks (CNNs) for image feature extraction and Long Short-Term Memory (LSTM) networks for capturing any temporal patterns.

## Libraries Used
- **TensorFlow & Keras**: 
  - Used for building deep learning models.
  - Key components: `Conv2D`, `MaxPooling2D`, `LSTM`, `Dense`, etc.
- **scikit-learn**:
  - Used for evaluating the model and dataset splitting.
  - Key components: `classification_report`, `confusion_matrix`, `train_test_split`.
- **NumPy**:
  - Used for array manipulation and numerical operations.
- **OS & Shutil**:
  - Used for file and directory operations to organize the dataset.
- **ZipFile**:
  - Used to extract the dataset from a zip archive.

## Functions

### 1. Dataset Splitting (`split_data`)
This function is used to split the dataset into training, validation, and test sets.

#### Parameters:
- `SOURCE`: Path to the source directory containing the images.
- `TRAINING`, `VALIDATION`, `TESTING`: Paths to destination directories for the respective splits.
- `train_size`, `val_size`, `test_size`: Ratios for splitting the dataset.

### 2. Model Training & Evaluation
- **Layers like `Conv2D` and `LSTM`**: Used to define the model that extracts image features and processes temporal patterns.
- **Evaluation**:
  - The model is evaluated using `classification_report` and `confusion_matrix` to assess accuracy, precision, recall, and other performance metrics.
## How to Run
1. Unzip the dataset using the `ZipFile` library.
2. Use the `split_data` function to split the dataset into training, validation, and test sets.
3. Define the model architecture using TensorFlow's Keras API.
4. Train the model and evaluate its performance using the provided metrics.

---

## EasyOCR Intermediate Step

In `easyocr.ipynb` file, this is an intermediate step where we attempted to obtain accurate OCR text recognition in a list called `final_text` and then used a Groq API call to clean and arrange the text in a formatted and structured way.

Set the file path in this code block:

```python
image = cv2.imread('/content/WhatsApp Image 2024-10-16 at 12.26.41 AM.jpeg')
imshow("Original Image", image, size=12)

V = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[2]
T = threshold_local(V, 45, offset=5, method="gaussian")
thresh = (V > T).astype("uint8") * 255
cv2_imshow(thresh)




