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
```
Ensure that you install all the required libraries beforehand. You can use a T4 GPU and set the GPU parameter in this code block to be true:

```python
reader = Reader(['en'], gpu=True)
start = time.time()
result = reader.readtext(image)
end = time.time()
time_taken = end - start
print(f"Time Taken: {time_taken}")
print(result)
```
This is for the celaning and proper formatting of the text, so we are using groq api call which is trained on a LLM
```python
from groq import Groq

GROQ_API_KEY= "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"
client = Groq(api_key=GROQ_API_KEY)
completion = client.chat.completions.create(
    model="llama3-8b-8192",
    messages=[{"role": "user", "content": f"Organize and clean up the following text into a proper readable format with appropriate sections:\n\n{final_text}"}],
    temperature=1,
    max_tokens=1024,
    top_p=1,
    stream=True,
    stop=None,
)

for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")
```
Currently, it’s using my API key, but you can create your own in the Groq website at https://console.groq.com/keys. Once everything is set up, the code takes around 10 seconds to run on a CPU and is much faster on a GPU, typically around 2 seconds. The API call takes about 0.5 seconds.

## Fruit Freshness

The notebook FruitFreshness_GaborFilterDefect_PerspectiveTransform.ipynb has given the steps in the comments which you can follow. Make sure to download the model weights in the .h5 format and the upload it to you google drive, set the proper image paths in the respective code blocks. The model summary which mentions its architecture is also mentioned in the colab notebook shared. It was trained upto 80 eppochs due to lack of processing power but can go upto an accuracy of 97% with strong processors like A100. The training process can be checked in Training_steps_of_the_fruitfreshness.ipynb

## Perspective Transform

Since images need to be properly zoomed in after camera takes the picture, we used perspective transform to get the contours of the captured image and took the extreme coordinates of the contours to zoom in the image as much as possible. Still its recommended to go for a unicolour background with proper lighting.
This is the code below mentioned 
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def imshow(title="Image",image=None,size=10):
    w,h=image.shape[0],image.shape[1]
    aspect_ratio=w/h
    plt.figure(figsize=(size*aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

image=cv2.imread('/content/OtrivinBbg.jpeg')
image2=cv2.imread("/content/OtrivinBbg.jpeg")
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

_,th=cv2.threshold(gray,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#imshow("Original",image)
imshow("Threshold",th)

contours,hierarchy=cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image,contours,-1,(0,255,0),thickness=2)
imshow("Contours",image)
print(str(len(contours)))
sorted_contours=sorted(contours,key=cv2.contourArea,reverse=True)

min_x=float('inf')
max_x=float('-inf')
min_y=float('inf')
max_y=float('-inf')
for cnt in sorted_contours:
    x,y,w,h=cv2.boundingRect(cnt)
    min_x=min(min_x, x)
    max_x=max(max_x, x + w)
    min_y=min(min_y, y)
    max_y=max(max_y, y + h)

#print(f"Overall bounding box - min_x:{min_x},max_x:{max_x},min_y:{min_y},max_y:{max_y}")
cv2.rectangle(image,(min_x, min_y),(max_x, max_y),(0, 255, 0),2)
cropped_image=image2[min_y:max_y, min_x:max_x]
cv2_imshow(cropped_image)
cv2.waitKey(0)
