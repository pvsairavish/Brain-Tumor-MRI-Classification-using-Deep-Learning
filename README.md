# 🧠 NeuroScan AI — Brain Tumor MRI Classification

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.1-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-3.x-D00000?logo=keras&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit&logoColor=white)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-CNN%20%7C%20Transfer%20Learning-8A2BE2)
![Task](https://img.shields.io/badge/Task-Multi--Class%20Image%20Classification-0E7490)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 🚀 Live Demo

**Try the live application here:**

👉 [[https://emipredict-ai-v3fzk4nptkzeovgkngp9kg.streamlit.app/](https://emipredict-ai-4usmwappd7yvakbjs3rwssi.streamlit.app/)]

---

**NeuroScan AI** is an end-to-end deep learning project for classifying brain MRI images into four categories: **Glioma, Meningioma, No Tumor, and Pituitary**.

The project covers the complete machine-learning lifecycle — dataset analysis, preprocessing, augmentation, custom CNN development, transfer learning, fine-tuning, model evaluation, model selection, model serialization, and deployment through an interactive Streamlit application.

> **Important:** NeuroScan AI is an educational and research project. It is **not a medical diagnostic system** and must not be used as a substitute for evaluation by a qualified healthcare professional.

---

## 📌 Project Overview

Brain MRI interpretation requires specialized medical expertise and can be time-consuming, particularly when large numbers of scans need to be reviewed.

This project explores how deep learning and computer vision can be applied to automated multi-class MRI image classification.

### The system classifies an MRI scan as:

| Class | Description |
|---|---|
| 🔴 **Glioma** | MRI image belonging to the Glioma class |
| 🔵 **Meningioma** | MRI image belonging to the Meningioma class |
| 🟢 **No Tumor** | MRI image without a tumor |
| 🟣 **Pituitary** | MRI image belonging to the Pituitary class |

The final selected model is a **fine-tuned MobileNetV2 transfer-learning model**, which is deployed in the Streamlit application.

---

# 🎯 Objectives

The project was designed to:

- Perform exploratory analysis of a labeled MRI image dataset.
- Understand image dimensions, class distribution, and dataset structure.
- Apply appropriate image preprocessing and data augmentation.
- Build a **Custom CNN from scratch** as a baseline.
- Implement **MobileNetV2 Transfer Learning** using ImageNet pretrained weights.
- Implement **EfficientNetB0 Transfer Learning** for model comparison.
- Fine-tune pretrained models for the MRI classification task.
- Use callbacks such as EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint.
- Handle mild class imbalance using class weights.
- Evaluate the selected model using held-out test data.
- Save the final trained model and class mapping.
- Build a professional Streamlit interface for real-time image classification.
- Maintain consistent preprocessing between training and deployment.

---

# 📊 Dataset

The project uses a labeled **Brain Tumor MRI Multi-Class Dataset**.

### Dataset summary

| Property | Value |
|---|---:|
| Total images | **2,443** |
| Training images | **1,695** |
| Validation images | **502** |
| Test images | **246** |
| Original image size | **640 × 640** |
| Color format | **RGB** |
| Number of classes | **4** |
| Classification type | **Multi-class, single-label** |

### Dataset split

```text
Total Dataset: 2,443 images

├── Train
│   └── 1,695 images
│
├── Validation
│   └── 502 images
│
└── Test
    └── 246 images
```

The dataset contains a mild class imbalance. Class weights were therefore calculated using a balanced weighting strategy rather than discarding images from larger classes.

---

# 🔬 Machine Learning Workflow

```text
                 Brain MRI Dataset
                        │
                        ▼
              Dataset Verification
                        │
                        ▼
               Exploratory Analysis
                        │
                        ▼
             Label Transformation
                        │
                        ▼
            Image Preprocessing
                        │
                        ▼
             Data Augmentation
                        │
                        ▼
        ┌───────────────┼────────────────┐
        │               │                │
        ▼               ▼                ▼
   Custom CNN      MobileNetV2      EfficientNetB0
   From Scratch    Transfer Learning Transfer Learning
        │               │                │
        │               ▼                ▼
        │          Fine-Tuning       Fine-Tuning
        │               │                │
        └───────────────┼────────────────┘
                        ▼
                Validation Comparison
                        │
                        ▼
                 Model Selection
                        │
                        ▼
              Held-Out Test Evaluation
                        │
                        ▼
             Save Final .h5 Model
                        │
                        ▼
                Streamlit Application
                        │
                        ▼
             Real-Time MRI Prediction
```

---

# 🧠 Models Implemented

Three deep learning approaches were implemented and compared.

## 1. Custom CNN

A convolutional neural network was designed and trained **from scratch** without ImageNet pretrained weights.

The architecture contains:

- Convolutional layers
- Batch Normalization
- Max Pooling
- Dropout
- Global Average Pooling
- Fully connected classification layer
- Four-class Softmax output

The Custom CNN provides a baseline for measuring how effectively a model trained from scratch can learn the MRI classification task.

### Best validation accuracy

**33.67%**

The relatively low validation performance indicates that this configuration had difficulty generalizing compared with the transfer-learning approaches.

---

## 2. MobileNetV2 — Transfer Learning

MobileNetV2 was initialized with ImageNet pretrained weights.

The training process used two stages:

### Stage 1 — Transfer Learning

- ImageNet pretrained convolutional base
- Base layers initially frozen
- New classification head added
- Class weights applied
- Training performed on the MRI dataset

### Stage 2 — Fine-Tuning

- Upper portion of the pretrained network unfrozen
- Batch Normalization layers kept appropriately controlled
- Lower learning rate used
- Best checkpoint selected according to validation performance

### Best validation accuracy

**93.03%**

MobileNetV2 was selected as the final deployment architecture.

---

## 3. EfficientNetB0 — Transfer Learning

EfficientNetB0 was also initialized with ImageNet pretrained weights.

The same staged strategy was used:

```text
Pretrained Feature Extraction
            ↓
Classification Head Training
            ↓
Fine-Tuning
            ↓
Validation Evaluation
```

### Best validation accuracy

**80.88%**

Although EfficientNetB0 performed substantially better than the Custom CNN, MobileNetV2 achieved the strongest validation performance in this project.

---

# 🏆 Model Comparison

The validation set was used for model selection so that the test set remained isolated for final evaluation.

| Model | Approach | Best Validation Accuracy | Deployment |
|---|---|---:|---|
| Custom CNN | From scratch | **33.67%** | No |
| EfficientNetB0 | Transfer Learning + Fine-Tuning | **80.88%** | No |
| **MobileNetV2** | **Transfer Learning + Fine-Tuning** | **93.03%** | **Yes** |

### Final selected model

**MobileNetV2 (Fine-tuned)**

Saved as:

```text
best_brain_tumor_model.h5
```

The model-selection process is based on validation performance rather than selecting a model using the held-out test set.

---

# 📈 Final Model Test Result

The final saved model was subsequently verified against the held-out test set.

### Latest verified test result

- **Test images:** 246
- **Correct predictions:** 229
- **Incorrect predictions:** 17
- **Test accuracy:** **93.09%**

```text
229 / 246 correctly classified

Test Accuracy = 93.09%
```

The test set was kept separate from validation-based model selection.

> The 93.09% figure represents the latest verified run of the saved final model. It should not be mixed with earlier experimental test results from previous model versions.

---

# ⚙️ Image Preprocessing

A critical part of the project is maintaining identical preprocessing assumptions between training and deployment.

All images are resized to:

```text
224 × 224 × 3
```

The final MobileNetV2 model contains its preprocessing layer internally:

```python
Rescaling(1./127.5, offset=-1)
```

Therefore, the Streamlit application **does not divide uploaded images by 255**.

The deployment pipeline is:

```text
Uploaded Image
      ↓
Convert to RGB
      ↓
Resize to 224 × 224
      ↓
Convert to float32
      ↓
Add Batch Dimension
      ↓
Saved Model
      ↓
Internal MobileNetV2 Preprocessing
      ↓
Prediction
```

This prevents a common deployment error where an image is normalized twice.

---

# 🖥️ Streamlit Application

The project includes a professional Streamlit dashboard named **NeuroScan AI**.

### Application capabilities

- Professional dark medical-AI dashboard
- MRI image upload
- JPG / JPEG / PNG support
- MRI preview
- Real-time prediction
- Confidence score
- Probability distribution for all four classes
- Model information
- Class information
- Navigation between Home, About, and Model Info
- Medical-use disclaimer
- Responsive dashboard layout

### Application output

For every uploaded MRI image, the application displays:

```text
Prediction
    ↓
Predicted Class
    ↓
Confidence Score
    ↓
Probability for:
    • Glioma
    • Meningioma
    • No Tumor
    • Pituitary
```

---

# 🧩 Technology Stack

| Category | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning | TensorFlow / Keras |
| Computer Vision | Pillow |
| Numerical Computing | NumPy |
| Data Processing | Pandas |
| Visualization | Matplotlib / Seaborn |
| Model Architectures | Custom CNN, MobileNetV2, EfficientNetB0 |
| Transfer Learning | ImageNet |
| Web Application | Streamlit |
| Model Format | HDF5 `.h5` |
| Development Environment | Jupyter / Kaggle Notebook / VS Code |
| Deployment | Streamlit-compatible deployment |

---

# 📁 Project Structure

A recommended GitHub repository structure is:

```text
NeuroScan-AI/
│
├── app.py
├── requirements.txt
├── README.md
│
├── best_brain_tumor_model.h5
├── class_names.json
│
├── notebooks/
│   └── Brain_Tumor_MRI_Model_Corrected.ipynb
│
├── reports/
│   └── Model_Comparison_Report.md
│
└── dataset/
    └── README.md
```

> The complete raw MRI dataset is not required to be committed to GitHub. Large datasets and model artifacts can be distributed separately when repository-size or hosting limits apply.

---

# 🚀 Run the Project Locally

## 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/NeuroScan-AI.git
cd NeuroScan-AI
```

Replace `YOUR_USERNAME` with your GitHub username.

---

## 2. Create a virtual environment

### Windows

```powershell
python -m venv venv
```

Activate it:

```powershell
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Verify TensorFlow

The saved `.h5` model was created using a newer Keras/TensorFlow serialization format.

The tested local environment uses:

```text
TensorFlow 2.19.1
```

Verify:

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

Expected:

```text
2.19.1
```

---

## 5. Run Streamlit

```bash
streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

---

# 📦 Required Deployment Files

The Streamlit application requires:

```text
app.py
best_brain_tumor_model.h5
class_names.json
requirements.txt
```

### `class_names.json`

The class order is:

```json
[
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]
```

This order must remain consistent with the model's Softmax output.

---

# ⚠️ Important Model Compatibility Note

The `.h5` file should be loaded using a compatible TensorFlow/Keras environment.

During development, TensorFlow 2.15 produced an `InputLayer` deserialization error for the saved model.

The model was successfully loaded and verified using:

```text
TensorFlow 2.19.1
```

Therefore, use the environment specified in `requirements.txt` rather than mixing incompatible TensorFlow/Keras versions.

---

# 🔎 Model Verification

A simple verification script can be used to confirm that the model loads correctly:

```python
import tensorflow as tf

model = tf.keras.models.load_model(
    "best_brain_tumor_model.h5",
    compile=False
)

print("TensorFlow:", tf.__version__)
print("Model:", model.name)
print("Input:", model.input_shape)
print("Output:", model.output_shape)
```

Expected model information:

```text
TensorFlow: 2.19.1
Model: MobileNetV2_TL
Input: (None, 224, 224, 3)
Output: (None, 4)
```

---

# 🧪 Example Prediction

For an uploaded MRI image, the application may produce output similar to:

```text
AI ANALYSIS

Prediction:
Glioma

Confidence:
99.98%

Class Probabilities:

Glioma        99.98%
Meningioma     0.01%
No Tumor       0.02%
Pituitary      0.00%
```

The displayed probabilities are model outputs and should not be interpreted as clinical certainty.

---

# 📊 Evaluation Strategy

The evaluation pipeline follows a strict separation of data:

```text
Training Set
    ↓
Model Training
    ↓
Validation Set
    ↓
Model Selection
    ↓
Final Saved Model
    ↓
Held-Out Test Set
    ↓
Final Evaluation
```

The test set is not used to choose the final deployment model.

Evaluation includes:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Class-wise performance analysis
- Training/validation accuracy curves
- Training/validation loss curves

---

# 🧠 Why MobileNetV2?

MobileNetV2 was selected because it provided the strongest validation performance among the implemented models while remaining relatively lightweight and suitable for an interactive deployment.

Advantages include:

- Strong transfer-learning capability
- ImageNet pretrained visual features
- Relatively compact architecture
- Efficient inference
- Suitable for web-based deployment
- Effective performance on a comparatively small image dataset

The final project therefore balances **classification performance and deployment practicality**.

---

# 🔐 Medical Safety & Responsible AI

This project is intended strictly for:

- Academic projects
- Deep-learning experimentation
- Computer-vision research
- Educational demonstrations
- Model-development practice

It is **not intended to**:

- Diagnose patients
- Replace radiologists
- Recommend medical treatment
- Determine patient prognosis
- Be used as an autonomous clinical decision system

MRI interpretation requires clinical context, appropriate imaging protocols, specialist review, and professional medical judgment.

---

# 🔮 Future Improvements

Possible future extensions include:

- Larger and more diverse MRI datasets
- External validation on independent datasets
- Explainable AI using Grad-CAM
- Visualization of model attention regions
- Calibration of confidence scores
- Cross-validation experiments
- More extensive hyperparameter optimization
- Model quantization for lightweight deployment
- Docker-based deployment
- Automated model monitoring
- Experiment tracking using MLflow
- Authentication and user management
- Multi-language application support
- Integration with clinical workflow systems in controlled research environments

---

# 📚 Project Deliverables

The completed project includes:

- ✅ Exploratory Data Analysis
- ✅ Dataset preprocessing
- ✅ Data augmentation
- ✅ Class imbalance handling
- ✅ Custom CNN
- ✅ MobileNetV2 transfer learning
- ✅ EfficientNetB0 transfer learning
- ✅ Fine-tuning
- ✅ Model comparison
- ✅ Validation-based model selection
- ✅ Test-set evaluation
- ✅ Saved `.h5` model
- ✅ Class mapping JSON
- ✅ Streamlit web application
- ✅ Professional UI/UX
- ✅ Deployment-ready requirements
- ✅ GitHub documentation

---

# 👨‍💻 Author

**Punati Venkata Sai Ravish**

📧 **Email:** punatiravish@gmail.com

🔗 **LinkedIn:**  
https://www.linkedin.com/in/ravish-punati-319a84253/

---

# ⭐ If You Find This Project Useful

If this project helped you understand deep learning, medical image classification, transfer learning, or Streamlit deployment:

- ⭐ Star the repository
- 🍴 Fork the project
- 🐛 Open an issue for improvements
- 💡 Share suggestions and ideas

---

## 📜 License

This project is intended for educational and research purposes.

If you publish or redistribute the project, please retain appropriate attribution to the original author and dataset source.
