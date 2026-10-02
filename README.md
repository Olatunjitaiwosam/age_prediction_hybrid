# Age Verification System — Hybrid CNN–Vision Language Model

A Python research prototype for exploring facial age estimation and age-group classification, with interactive Streamlit and Flask interfaces.

This repository accompanies a final-year Computer Science project at the **University of East London**. The project received **91%** and was selected among the **top 12 projects as a CDT Finalist in 2026**. The 91% is the academic project mark, not a model accuracy score.

## Overview

The application combines face detection, neural-network inference and optional vision-language model (VLM) reasoning. It lets a user inspect predictions from images, uploaded videos and webcam input.

The code demonstrates integration across computer vision, Python web interfaces, model loading, annotated media output and HTTP APIs. It is an experimental prototype; its age estimates and displayed allow/restrict labels are not validated identity or legal-age verification.

## Features

- Detect multiple faces and display bounding boxes.
- Estimate age and classify each face as child, teen or adult.
- Show predicted age, group probabilities and a classification confidence score.
- Compare CNN output with optional VLM reasoning for image analysis.
- Process uploaded videos and produce annotated output.
- Support webcam input through Streamlit WebRTC or the Flask frame endpoint.
- Select between configured model backbones, subject to compatible trained weights.

### Technologies

| Area | Tools |
| --- | --- |
| Application | Python, Streamlit, Flask |
| Model inference | PyTorch, torchvision, timm |
| Face detection and image processing | Ultralytics YOLO, OpenCV, Albumentations |
| Optional VLM integration | OpenAI Python SDK; the current code requests `gpt-4o` |
| Media and connectivity | streamlit-webrtc, aiortc, PyAV |
| Web interface | HTML templates and browser-side JavaScript |

## How it works

1. Decode an image or video frame.
2. Detect candidate faces using the configured YOLO model, with an OpenCV Haar-cascade fallback.
3. Crop and resize each region to 224 × 224 pixels and apply ImageNet normalisation.
4. Run a selected neural-network backbone with separate age-regression and age-group classification heads.
5. Display annotations and per-face results.
6. When enabled for image analysis, send the face crop and CNN result to the VLM and display its response alongside the prediction.

The age-group classification head determines the current allow/restrict label. It does not simply threshold the numerical age estimate, so the two outputs may disagree. VLM responses are shown as a comparison rather than replacing the CNN decision.

## Repository guide

| Path | Purpose |
| --- | --- |
| `streamlit_app.py` | Streamlit interface, inference pipeline and media processing |
| `requirements.txt` | Streamlit application dependencies |
| `packages.txt` | System packages configured for the cloud environment |
| `flask_app/app.py` | Flask routes, JSON APIs and model initialisation |
| `flask_app/core/config.py` | Flask settings and model download locations |
| `flask_app/core/models.py` | Neural-network definitions and model loading |
| `flask_app/core/predictor.py` | Detection, prediction, annotation and VLM helpers |
| `flask_app/templates/` | Web interface and demonstration age-gate pages |
| `.github/workflows/deploy.yml` | Existing EC2 deployment workflow |

## Running locally

These commands describe the entry points in the repository. They have not yet been validated in a clean installation; package compatibility, model downloads and local system libraries can affect startup.

### Streamlit

Clone the repository and create an isolated environment:

```bash
git clone https://github.com/Olatunjitaiwosam/age_prediction_hybrid.git
cd age_prediction_hybrid
python -m venv .venv
```

Activate it:

```bash
# macOS / Linux
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

Install dependencies and launch:

```bash
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Open the local address printed by Streamlit. Start with **Image Upload**, keep VLM reasoning disabled and check the model-loading logs before interpreting results.

### Flask

From the repository root, using an activated environment:

```bash
python -m pip install -r flask_app/requirements.txt
cd flask_app
python app.py
```

The default local address is `http://localhost:5000`. Flask initialises models at startup, so the first launch can take time and require substantial memory.

Selected endpoints:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| POST | `/api/predict` | Image predictions and annotated image |
| POST | `/api/predict/video` | Video analysis and output download URL |
| POST | `/api/stream/frame` | Prediction for a browser webcam frame |
| GET | `/api/models/status` | Model-file presence and format checks |

## Models and configuration

Both interfaces attempt to download missing model files at startup. The configured release contains six files totalling approximately **1.07 GB**. Streamlit stores them under `models/` by default; Flask uses `flask_app/models/`. Set `MODEL_DIR` to choose a different directory.

- Streamlit defaults to DenseNet; Flask defaults to ViT Base.
- The Streamlit sidebar lets you choose a backbone and model paths.
- Weights must match the selected architecture.
- The `vit-tiny` option currently points to the ViT Base checkpoint; it is not a separately supplied Tiny model.

### External model source

The configured checkpoint downloads come from [Mystique1337/age_prediction_hybrid- — v1.0-models](https://github.com/Mystique1337/age_prediction_hybrid-/releases/tag/v1.0-models), an external source used by this project. These checkpoints are not presented as models trained by the repository owner. Check the source's terms and provenance before redistribution or other use.

The repository currently contains application and inference code. Training scripts, datasets and a reproducible evaluation report are not included.

## Optional VLM reasoning

VLM reasoning is disabled by default in Streamlit. To enable it, provide an `OPENAI_API_KEY` through an environment variable, Streamlit secrets or the local password field. Flask reads the environment variable and also supports the optional request field.

Enabling this feature sends the face crop and prediction context to the configured external API and may incur API charges. Keep keys out of source control and use only images you have permission to process.

The prompt contains assertions about consent and study approval; those assertions are not evidence that any particular uploaded image has those permissions.

## Evaluation and limitations

No verified benchmark figures are published here yet. Classification confidence is a model output, not demonstrated accuracy or a guarantee of correct age.

Known implementation limitations include:

- **Model loading:** Streamlit can continue with random weights if a checkpoint is missing or incompatible. A running interface alone does not prove a trained model loaded successfully; check the logs.
- **Face detection:** the generic YOLOv8n fallback is an object detector, not a substitute for a trained face detector.
- **Video processing:** detections across frames are not tracked as unique people; summary counts can include repeated observations of the same face.
- **VLM comparison:** the VLM receives the CNN result in its prompt, so this is not a blinded independent evaluation.
- **Reproducibility:** dependencies use minimum versions, and a clean environment and compatible checkpoint set have not yet been documented.
- **Privacy:** the Flask video route saves uploaded and annotated videos to its upload directory; automatic retention cleanup is not implemented in that route.

The deployment workflow and demonstration pages illustrate integration work. They do not establish production readiness or regulatory compliance.

## Portfolio evidence to add

The next useful additions are a confirmed contribution statement, permitted screenshots or a short demo, and an evaluation summary with dataset source, test split, sample size, metrics and failure cases.

These would make it easier to assess what was implemented, how it was evaluated and which components were adapted from external sources.
