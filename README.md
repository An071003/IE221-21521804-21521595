# Toxic Comment Classifier

Multi-label toxic comment detection with a Bidirectional LSTM, served through a Flask web app and a Tkinter desktop app.

Course project for **IE221.O23.CNCL – Python Programming**, University of Information Technology (UIT, VNU-HCM).

| Member | Student ID |
|---|---|
| Hồ Vũ An | 21521804 |
| Nguyễn Thành Trung | 21521595 |

## Overview

The model reads an English comment and scores it on six toxicity labels: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`.

- **Web app (Flask):** register / log in, then post comments under an article. A comment is blocked and the detected labels are shown when any score exceeds 0.5; otherwise it is published.
- **Desktop app (Tkinter):** type a comment and see the six label scores as a bar chart.

## Model

| Item | Detail |
|---|---|
| Dataset | [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) (`train.csv`) |
| Text vectorization | Keras `TextVectorization`, 20,000-token vocabulary, sequence length 1,800 |
| Architecture | Embedding(32) → Bidirectional LSTM(32) → Dense 128 → 256 → 128 → Dropout(0.5) → Dense(6, sigmoid) |
| Loss / optimizer | Binary cross-entropy / Adam |
| Metrics | accuracy, precision, recall |

Exploration, training and evaluation are in `Jupyter-notebook/jupyter_notebook.ipynb`. Pre-trained weights and the vocabulary are in `Save_model/`.

## Project structure

```text
├── main.py                  # Flask web app (auth + comment moderation)
├── App_desktop/             # Tkinter desktop app
├── Model/Model.py           # BiLSTM model, training, evaluation, plots
├── Data/Data.py             # Loading, analysis and preprocessing of the dataset
├── Library/Library.py       # Shared imports and the Vectorize layer
├── Train_save_model/        # Script that trains the model and saves the weights
├── Save_model/              # model.weights.h5, vectorizer.pkl
├── templates/, static/      # Web UI
├── data.json                # Sample article and comments
└── users.example.json       # Empty user store template
```

## Getting started

Requirements: Python 3.10+ (Tkinter is bundled with the standard Python installer).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

### Run the web app

```bash
export SECRET_KEY="change-me"      # Windows PowerShell: $env:SECRET_KEY="change-me"
python main.py
```

Open http://127.0.0.1:5000. `users.json` is created on the first registration. Set `FLASK_DEBUG=1` to enable debug mode during development.

### Run the desktop app

```bash
python -m App_desktop.App_desktop
```

### Retrain the model

1. Download `train.csv` from the Kaggle competition and place it at `jigsaw-toxic-comment-classification-challenge/train.csv/train.csv`.
2. Run the training script from its own folder (paths are relative):

```bash
cd Train_save_model
python Train_save_model.py
```

## Notes

- `users.json`, datasets and local tools are git-ignored; the repository does not ship credentials or the Kaggle data.
- Passwords are stored as SHA-256 hashes. This is acceptable for a course project, but a production system should use a salted, slow hash (bcrypt or argon2).
- The model only supports English text.
