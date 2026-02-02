# Solar Panel Defect Detection System

A full-stack Deep Learning application capable of detecting defects in Solar Photovoltaic (PV) arrays across multiple imaging modalities (RGB, Electroluminescence, and Thermal).

This system uses a **Hybrid Model Architecture**, dynamically routing the input image to the most specialized Neural Network for that specific data type to ensure maximum accuracy.

## 🚀 Features

* **Multi-Modality Support:** Analyzes RGB, EL (Electroluminescence), and Thermal infrared images.
* **Hybrid AI Backend:**
    * **RGB:** GoogLeNet (Optimized for visible spectrum defects like bird drops & snow).
    * **EL:** EfficientNetV2-S (Specialized for micro-cracks and cell defects).
    * **Thermal:** EfficientNetV2-M (Specialized for hotspots and diode failures).
* **Real-time Inference:** Fast API response times with confidence scores.
* **User-Friendly Interface:** Clean React-based UI for easy image uploading and result visualization.

## 🛠️ Tech Stack

* **Frontend:** React.js (Vite), Tailwind CSS, Lucide React
* **Backend:** Python, FastAPI, Uvicorn
* **Machine Learning:** PyTorch, Torchvision
* **Image Processing:** Pillow (PIL), NumPy

---

## 📂 Project Structure

```bash
Solar-Defect-Detection/
├── backend/
│   ├── main.py              # FastAPI application & Model Logic
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── src/                 # React source code
│   ├── package.json         # Node dependencies
│   └── ...
└── Models/                  # Trained PyTorch Weights (.pth)
    ├── googlenet_rgb.pth
    ├── efficientnet_el_full.pth
    └── efficientnet_best_thermal_model.pth