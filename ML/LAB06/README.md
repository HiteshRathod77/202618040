# DS605 Lab 6 — Feature Extraction with Image and Text Data

Name: <your name>
ID: <your id>

## Datasets
- Asphalt Crack Dataset — 400 images (200 Cracks + 200 NonCracks), resized to 448x448.
- Email Spam Classification Dataset — 5172 emails, 3000 word-count features, binary spam label.

## Part A — Image Features
- Resize to 448x448, convert to grayscale.
- 7 baseline features: mean, std, dark_ratio, bright_ratio, median, edge_count, edge_density (Canny 100-200).
- Models: Logistic Regression (scaled) and Random Forest (200 trees).
- Results (80/20 stratified split):
  - LogReg   acc 0.8375  F1 0.8312
  - RF       acc 0.9750  F1 0.9756
- Best: Random Forest. Recall on crack class = 1.0.

## Part B — Text Vectorization
- Two representations of the same word-count matrix:
  1. Raw counts.
  2. TF-IDF (TfidfTransformer).
- Models: Logistic Regression, Multinomial Naive Bayes.
- Results:
  | Model | Features | Acc | F1 | Train (s) |
  |---|---|---|---|---|
  | LogReg + Count | 3000 | 0.9826 | 0.9704 | 3.91 |
  | NB + Count | 3000 | 0.9420 | 0.9042 | 0.03 |
  | LogReg + TFIDF | 3000 | 0.9507 | 0.9134 | 0.05 |
  | NB + TFIDF | 3000 | 0.8802 | 0.7510 | 0.004 |
- Best: LogReg on raw counts.

## Part C — Representation Improvement
- Image side: added fine Canny (50-150) edge count/density + mid-intensity ratio
  → precision improved 0.952 → 0.975 at the same accuracy (0.975).
- Text side: cap vocabulary at top-500 words, use sublinear TF-IDF.
  → train time 3.9 s → 0.03 s, accuracy 0.9826 → 0.9729.

## Files
- `main.ipynb` — full notebook.
- `image_features.csv` — one row per image with extracted features.
- `emails.csv` — word-count email dataset.
- `448/` — image dataset (Cracks / NonCracks).

## Notes
- No CNNs, no pretrained embeddings.
- All image features extracted manually with NumPy and OpenCV.
- Text representation done with sklearn vectorizers / transformers.


## Setup

Clone the repo, then download the two datasets:

1. Asphalt Crack Dataset (400 images, 448x448):
   - Source: Mendeley Data (Asphalt Crack Dataset)
   - Place the folder as `448/` next to the notebook, with `Cracks/` and
     `NonCracks/` subfolders (200 images each).

2. Email Spam Classification Dataset:
   - Source: Kaggle (Email Spam Classification Dataset CSV)
   - File: `emails.csv`
   - Place it next to the notebook.

Then open `202618040_Lab06.ipynb` and run top to bottom.
If `image_features.csv` is already present, Part A's extraction cell can be
skipped to save time.
