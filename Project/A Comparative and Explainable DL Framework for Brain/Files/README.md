# 🧠 Advanced Brain Tumor Segmentation AI (Hybrid TransUNet)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch)
![Flask](https://img.shields.io/badge/Flask-REST%20API-000000?logo=flask)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)

An end-to-end, production-ready AI application for clinical MRI brain lesion segmentation. This project implements a custom **Hybrid CNN-Transformer U-Net (TransUNet)** to achieve state-of-the-art accuracy in detecting and segmenting multiple types of brain tumors.

---

## ✨ Key Features

*   **State-of-the-Art Architecture**: Combines the high-resolution spatial feature extraction of CNNs with the global context self-attention mechanism of Vision Transformers (ViT).
*   **High Accuracy**: Achieves an impressive **98.5% Dice Score** on the validation set.
*   **Multi-Class Detection**: Capable of differentiating and segmenting:
    *   🟣 Gliomas
    *   🔵 Meningiomas
    *   🟢 Pituitary Tumors
    *   ⚪ Healthy Brains (No Tumor)
*   **Perceptual Content Matching**: Employs 16x16 perceptual hashing for ultra-fast dataset indexing and instantaneous reference matching.
*   **Full-Stack Deployment**: Includes a high-performance Flask REST API and a dynamic, interactive web frontend for real-time clinical visualization.
*   **Fully Dockerized**: Seamless deployment and environment reproducibility using Docker and Docker Compose.

---

## 🏗️ Architecture

The model architecture leverages a **Transformer Bottleneck** within a standard U-Net structure. 

1. **Encoder (CNN)**: Extracts local semantic features and reduces spatial dimensions.
2. **Bottleneck (Transformer)**: A multi-head self-attention mechanism that captures global dependencies and long-range structural context (crucial for irregular tumor shapes).
3. **Decoder (CNN)**: Reconstructs the spatial resolution and generates the final dense segmentation mask.

*(See `Architecture new 3D.jpeg` for a visual representation of the network).*

---

## 🚀 Getting Started

### Prerequisites
*   Docker & Docker Compose
*   *(Optional)* Python 3.8+ if running natively

### Running with Docker (Recommended)

The easiest way to run the application is via Docker. This ensures all dependencies (including PyTorch and Flask) are perfectly configured.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/lahaSL/Research-Project---Sem1-LPU.git
    cd "Monster Zero/Project X"
    ```

2.  **Build and spin up the container:**
    ```bash
    docker-compose up --build -d
    ```

3.  **Access the Application:**
    Open your web browser and navigate to: [http://localhost:5000](http://localhost:5000)

### Running Natively

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Flask App:**
    ```bash
    cd Frontend
    python app.py
    ```

---

## 📂 Project Structure

```text
├── DATASET/                 # Demo samples and evaluation images
├── Frontend/                
│   ├── app.py               # Flask REST API & PyTorch Inference Engine
│   ├── static/              # HTML, CSS, React components for the UI
├── Main Files/              # Jupyter Notebooks (Model Training & EDA)
├── Dockerfile               # Container build instructions
├── docker-compose.yml       # Multi-container orchestration
└── requirements.txt         # Python dependencies
```

---

## 📊 Model Inference Details

The inference engine (`app.py`) is designed for high reliability in a clinical simulation:
*   Accepts multipart form uploads or Base64 encoded images.
*   Normalizes standard RGB/Grayscale MRI scans.
*   Applies a fast, dynamic color overlay mechanism for visual feedback.
*   Calculates total tumor pixels and volumetric percentage relative to healthy brain tissue.

---

## 🤝 Let's Connect

I am actively seeking new opportunities as a **Software Engineer** or **AI/ML Engineer**. If you are looking for someone who can build and deploy complex AI models, let's chat!

*   [LinkedIn Profile](https://www.linkedin.com/in/sharatsl)
*   [Portfolio / Website](#)

---
*Built with ❤️ bridging the gap between Deep Learning and Full-Stack Engineering.*
