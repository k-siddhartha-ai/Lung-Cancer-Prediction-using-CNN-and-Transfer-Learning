🧬 Lung Cancer Prediction using CNN and Transfer Learning

Deep Learning • Medical Image Classification • Transfer Learning • Computer Vision

Author: K. Siddhartha

This project aims to build a Lung Cancer Prediction System using Convolutional Neural Networks (CNN) and transfer learning. The model classifies lung cancer images into four categories:

Normal

Adenocarcinoma

Large Cell Carcinoma

Squamous Cell Carcinoma

📑 Table of Contents

Introduction

Dataset

Dependencies

Project Structure

Training the Model

Using the Model

Results

Acknowledgements

License

🧠 Introduction

Lung cancer is one of the leading causes of cancer-related deaths worldwide. Early detection and accurate classification are crucial for effective treatment and patient survival. This project leverages deep learning techniques to develop a robust lung cancer classification model using chest X-ray images.

📊 Dataset

The dataset used in this project consists of lung cancer images categorized into four classes:

Normal

Adenocarcinoma

Large Cell Carcinoma

Squamous Cell Carcinoma

The dataset should be organized into training (train), validation (valid), and testing (test) folders with the following subfolders for each class:
train/
 ├── normal/
 ├── adenocarcinoma/
 ├── large_cell_carcinoma/
 └── squamous_cell_carcinoma/

valid/
 ├── normal/
 ├── adenocarcinoma/
 ├── large_cell_carcinoma/
 └── squamous_cell_carcinoma/

test/
 ├── normal/
 ├── adenocarcinoma/
 ├── large_cell_carcinoma/
 └── squamous_cell_carcinoma/

Alternatively, you can download a similar dataset from Kaggle:

https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images

▶ Google Colab Link

To replicate and run the project in Google Colab:

https://colab.research.google.com/drive/1kMTghEwVoJaFmlKydxuhhoyzHluIUjoV?usp=sharing

⚙ Dependencies

Required libraries:

Python 3.x

pandas

numpy

seaborn

matplotlib

scikit-learn

tensorflow

keras

Install dependencies:
pip install pandas numpy seaborn matplotlib scikit-learn tensorflow keras

📁 Project Structure
.
├── Lung_Cancer_Prediction.ipynb
├── README.md
├── dataset/
│ ├── train/
│ ├── test/
│ └── valid/
└── best_model.hdf5

File Description

Lung_Cancer_Prediction.ipynb — Notebook for training and evaluation

dataset/ — Image dataset grouped by cancer type

best_model.hdf5 — Saved trained model weights

🏗 Training the Model

The Jupyter Notebook contains the full training pipeline:

Mount Google Drive

Load and preprocess data using ImageDataGenerator

Define the model using Xception transfer learning

Compile the model

Train with callbacks (EarlyStopping, LR scheduler, checkpoints)

Save the trained model

Example Code
pretrained_model = tf.keras.applications.Xception(
    weights='imagenet',
    include_top=False,
    input_shape=[*IMAGE_SIZE, 3]
)
pretrained_model.trainable = False

model = Sequential([
    pretrained_model,
    GlobalAveragePooling2D(),
    Dense(4, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

🔬 Using the Model

Steps:

Load trained .h5 model

Preprocess input image

Predict class probabilities

Display prediction

Example
model = load_model('/content/drive/MyDrive/dataset/trained_lung_cancer_model.h5')
predictions = model.predict(img)
predicted_class = np.argmax(predictions[0])

📈 Results

After training and evaluation:

Final model accuracy: 93%

The model demonstrates strong performance in classifying lung cancer categories using transfer learning.

🙏 Acknowledgements

Chest CT Scan Images Dataset:

https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images

📜 License

This project is licensed under the MIT License.

Feel free to use or modify this code for educational and non-commercial purposes.



