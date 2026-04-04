# AI-Based Fake News Detection

A lightweight React + Flask app that predicts whether a news article is real or fake using a scikit-learn model trained on the included dataset.

## Tech Stack

- Frontend: React, HTML, CSS, JavaScript
- Backend: Flask
- ML Model: scikit-learn

## Features

- Clean React dashboard with tabs for prediction, dataset preview, model insights, and explanation
- Flask API for predictions and metrics
- TF-IDF + Logistic Regression classification pipeline
- Confidence score and class probabilities

## Run Locally

Open two terminals in `D:\ML_project`.

### Backend

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m backend.app
```

The API runs on `http://127.0.0.1:5000`.

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

The React app runs on `http://127.0.0.1:5173` and proxies API requests to Flask.

## Build for Production

```powershell
cd frontend
npm run build
cd ..
python -m backend.app
```

The Flask app serves the built React files from `frontend/dist`.

## Dataset

The app now uses the provided dataset archive extracted into `data/news_archive/` with `Fake.csv` and `True.csv`. The backend combines each article's title and text into one training field.
