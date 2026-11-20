# Social Guard

**AI-based analysis of fake and bot social media accounts.**

Social Guard looks at the publicly visible profile statistics of an Instagram, Twitter/X or Facebook account and classifies it as **Real**, **Fake** or **Bot**. It shows a probability for each class, explains the result with SHAP and LIME, and can export a PDF report.

> Final-year B.Tech project (Semester VII, 2026-27), K. J. Somaiya School of Engineering, Somaiya Vidyavihar University.
> **Authors:** Shweta Yadav, Swara Sawant  |  **Guide:** Dr. Sonali Patil

This is an academic proof of concept trained on public research datasets. It can make mistakes and should support a human reviewer, never replace one.

---

## What the three labels mean

| Label | Meaning in this project |
|---|---|
| **Real** | A genuine, human-run account |
| **Fake** | A fake human-style account, such as purchased fake followers or a fake profile |
| **Bot** | An automated account, such as a spambot |

These definitions come from how the source datasets were labelled, so the model learns what those datasets call fake or bot.

---

## How it works

```
profile link / manual form / screenshots
        |
   src/scraper.py or src/image_reader.py     -> raw profile (followers, following, posts, bio, ...)
        |
   src/feature_engineering.py                -> 10 model features (missing fields stay NaN, never 0)
        |
   StandardScaler  ->  VotingClassifier      -> probabilities for Real / Fake / Bot
        |
   src/xai.py + src/explain.py               -> SHAP, LIME and plain-English indicators
        |
   Flask web page + PDF report
```

### The 10 features

| Feature | What it captures |
|---|---|
| `followers_pr`, `following_pr`, `posts_pr` | Percentile of each count within its own platform, so "5,000 followers" is judged against that platform |
| `follower_following_ratio` | `followers / (following + 1)` |
| `activity_rate` | `posts / (followers + 1)` |
| `bio_len` | Length of the bio |
| `username_digit_ratio` | Share of digits in the username |
| `name_eq_username` | Display name matches the username |
| `has_profile_pic` | Profile has a picture |
| `is_private` | Account is private |

The platform itself is never a model input. A field that cannot be read is passed to the model as unknown (NaN), not as zero.

### Models and training

- Four classifiers (XGBoost, LightGBM, Random Forest, AdaBoost) and a soft-voting ensemble of all four.
- 80/20 train/test split, stratified by platform and class, `random_state = 42`.
- Class imbalance: SMOTE applied separately inside each platform x class group (training split only), then random feature-group masking so the model copes with missing fields, then equal sample weights per platform x class group.
- The deployed model is the one with the best weighted F1 (the voting ensemble).

### Results (held-out test set of 4,123 accounts)

| Model | Accuracy |
|---|---|
| **Voting ensemble (deployed)** | **95.97%** |
| XGBoost | 95.95% |
| LightGBM | 95.88% |
| Random Forest | 94.96% |
| AdaBoost | 88.29% |

Ensemble: weighted F1 0.9598, macro F1 0.9615, ROC-AUC 0.9933. Below 70% confidence the app still gives a verdict and adds a caution note. Above 70% confidence the test accuracy is about 98%.

**Important limit:** these scores are for accounts that look like the training data. When a whole platform is held out of training, performance drops sharply (for example Real recall on held-out Instagram is about 28%, and Bot recall on held-out Twitter is 0%). The model is not claimed to work on a platform it has not seen.

---

## Installation

Requires **Python 3.10 or newer**.

```bash
git clone https://github.com/swarasawanttt/Social-Guard.git
cd Social-Guard

python -m venv venv
# Windows:  venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

pip install -r requirements.txt
```

**Screenshot reading (optional):** the OCR feature needs the Tesseract program installed on your computer, not just the Python package. On Windows, install it from the Tesseract OCR builds and make sure it is on your PATH. Everything else works without it.

## Running the app

Pre-trained models are included in `models/`, so you can start straight away:

```bash
python app.py
```

Then open <http://127.0.0.1:5000>.

1. Choose a platform and paste a profile link or username.
2. Public pages are read automatically where possible. If a page cannot be read (private account, login wall), enter the visible numbers by hand or upload screenshots, and review the extracted values before analysis.
3. Read the verdict, the confidence, the explanations, and download the PDF report if you want one.

Accounts with 0 followers, 0 following and 0 posts are rejected as having too little information. A platform-verified badge is treated as strong evidence of a genuine account by a documented rule, which overrides the model's score and is shown as a rule.

## Retraining the model

```bash
python src/train_model.py
```

This reads `data/processed/unified_raw.csv`, cleans it, builds the percentile tables, trains all five models, compares them, runs a leave-one-platform-out check, and saves everything to `models/`, including `training_metadata.json` with the exact numbers of the run.

## Running the tests

From the project root:

```bash
python tests/run_test_cases.py
```

This runs 43 software test cases against the real code (input validation, scraping behaviour, prediction, explanations, reports) without using the internet, and writes `tests/test_results.csv` and `tests/test_results.md`.

---

## Project structure

```
app.py                  Flask web application (routes and pages)
config.py               Paths, feature list, training and verdict settings
src/
  scraper.py            Reads public profile statistics from a profile page
  image_reader.py       Reads statistics from uploaded screenshots (OCR)
  url_resolver.py       Offline checks of links and usernames
  platform_check.py     Checks the selected platform matches the link
  feature_engineering.py  Raw profile -> the 10 model features
  model_utils.py        Shared maths: ratios, percentile tables, imputing wrapper
  train_model.py        Cleaning, augmentation, training, evaluation, saving
  predict.py            Full prediction pipeline used by the app
  xai.py                SHAP and LIME explanations
  explain.py            Plain-English behavioural indicators
  report_generator.py   PDF report
  collect_data.py       Optional tool to collect new labelled public accounts
  utils.py              Small helpers (URL parsing, number formatting)
templates/              HTML pages
models/                 Trained models, scaler, percentile tables, metadata
data/processed/         unified_raw.csv (training table), model_features.csv
tests/                  Test cases and their recorded results
```

---

## Data

Training rows come from public research datasets, merged into one schema:

- **Cresci-2017** (Twitter): genuine accounts, fake followers, social spambots, traditional spambots
- **Kaggle Instagram fake-profile dataset** (5,000 accounts)
- **Instagram bot / not-bot accounts** (700 each)
- **Facebook spam detection** (600 accounts, 103 spam)

After cleaning there are 20,612 accounts (Real 7,316, Fake 5,775, Bot 7,521). The raw dataset files are not included; `data/processed/unified_raw.csv` holds only counts and simple flags, with no usernames, names or bios.

## Limitations

- Labels are inherited from the source datasets, not independently verified.
- The Facebook data has only Real and Bot examples, with no Fake class.
- Cross-platform generalisation is weak (see above), and a bot built to look like a normal user can be missed.
- Only profile-level features are used, with no posting content or timing.
- Automatic reading of public pages depends on how each platform currently serves its pages and may stop working when they change.

## Responsible use

Use Social Guard only on public accounts. A wrong "Fake" or "Bot" label can harm a real person, so treat every result as a hint for a human to check, not as a decision. Respect each platform's terms of service.
