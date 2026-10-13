import pandas as pd
import pickle

from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

print("=" * 65)
print("CITIZEN GRIEVANCE AI - MODEL TRAINING")
print("=" * 65)

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "complaints_dataset.csv"
MODEL_PATH = BASE_DIR / "complaint_model.pkl"

EXPECTED_CATEGORIES = {
    "Garbage Disposal",
    "Drainage / Sanitation",
    "Water Pollution",
    "Air Pollution",
    "Noise Pollution"
}


# ============================================================
# 2. LOAD AND VALIDATE DATASET
# ============================================================

print("\nLoading complaint dataset...")

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}"
    )

data = pd.read_csv(DATASET_PATH)

required_columns = ["complaint", "category"]

if not all(column in data.columns for column in required_columns):
    raise ValueError(
        "CSV must contain these columns: complaint, category"
    )

data = data.dropna(
    subset=["complaint", "category"]
).copy()

data["complaint"] = (
    data["complaint"]
    .astype(str)
    .str.strip()
)

data["category"] = (
    data["category"]
    .astype(str)
    .str.strip()
)

data = data[
    (data["complaint"] != "")
    & (data["category"] != "")
].copy()

# Detect identical complaint text assigned to different categories.
normalized_text = (
    data["complaint"]
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

conflicting_labels = (
    data.assign(normalized_text=normalized_text)
    .groupby("normalized_text")["category"]
    .nunique()
)

conflicting_labels = conflicting_labels[
    conflicting_labels > 1
]

if not conflicting_labels.empty:
    raise ValueError(
        f"Found {len(conflicting_labels)} identical complaint "
        "texts assigned to different categories. Correct these "
        "labels in complaints_dataset.csv before training."
    )

# Remove duplicate complaint text.
data = data.drop_duplicates(
    subset=["complaint"]
).reset_index(drop=True)

print("\nDataset information")
print("-" * 65)

print("Total usable complaints:", len(data))

print("\nExamples in each category:")
print(data["category"].value_counts().sort_index().to_string())

actual_categories = set(data["category"].unique())

print("\nCategories:", sorted(actual_categories))

missing_categories = EXPECTED_CATEGORIES - actual_categories
unexpected_categories = actual_categories - EXPECTED_CATEGORIES

if missing_categories:
    print("\nWARNING: Missing expected categories:")
    print(sorted(missing_categories))

if unexpected_categories:
    print("\nWARNING: Unexpected category names:")
    print(sorted(unexpected_categories))

if len(data) < 20:
    raise ValueError(
        "Too few usable complaints. Add more labelled examples."
    )

if data["category"].nunique() < 2:
    raise ValueError(
        "At least two categories are required."
    )

if data["category"].value_counts().min() < 5:
    raise ValueError(
        "Each category should have at least five examples "
        "for this training configuration."
    )


# ============================================================
# 3. SPLIT DATA
# ============================================================

X = data["complaint"]
y = data["category"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nTraining examples:", len(X_train))
print("Testing examples:", len(X_test))


# ============================================================
# 4. DEFINE MODELS
# ============================================================

experiments = [
    {
        "name": "Logistic Regression - Unigrams",
        "vectorizer": TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 1),
            sublinear_tf=True
        ),
        "classifier": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        )
    },
    {
        "name": "Logistic Regression - Unigrams and Bigrams",
        "vectorizer": TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True
        ),
        "classifier": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        )
    },
    {
        "name": "Linear SVM - Unigrams and Bigrams",
        "vectorizer": TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True
        ),
        "classifier": LinearSVC(
            class_weight="balanced",
            random_state=42
        )
    }
]


# ============================================================
# 5. COMPARE MODELS USING CROSS-VALIDATION
# ============================================================

results = []
trained_models = {}

number_of_splits = min(
    5,
    int(y_train.value_counts().min())
)

cv = StratifiedKFold(
    n_splits=number_of_splits,
    shuffle=True,
    random_state=42
)

print("\nComparing machine-learning models...")
print("=" * 65)

for experiment in experiments:

    name = experiment["name"]

    print(f"\nTraining: {name}")

    pipeline = Pipeline([
        ("tfidf", experiment["vectorizer"]),
        ("classifier", experiment["classifier"])
    ])

    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="f1_macro"
    )

    mean_cv_f1 = cv_scores.mean()

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    results.append({
        "Model": name,
        "Cross-validation Macro F1": mean_cv_f1,
        "Test Accuracy": accuracy,
        "Test Macro Precision": precision,
        "Test Macro Recall": recall,
        "Test Macro F1": f1
    })

    trained_models[name] = pipeline

    print(f"Cross-validation Macro F1: {mean_cv_f1:.2%}")
    print(f"Test accuracy:             {accuracy:.2%}")
    print(f"Test macro precision:      {precision:.2%}")
    print(f"Test macro recall:         {recall:.2%}")
    print(f"Test macro F1-score:       {f1:.2%}")


# ============================================================
# 6. SELECT THE BEST MODEL
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Cross-validation Macro F1",
    ascending=False
).reset_index(drop=True)

print("\n")
print("=" * 65)
print("MODEL COMPARISON")
print("=" * 65)

print(
    results_df.to_string(
        index=False,
        formatters={
            "Cross-validation Macro F1": "{:.2%}".format,
            "Test Accuracy": "{:.2%}".format,
            "Test Macro Precision": "{:.2%}".format,
            "Test Macro Recall": "{:.2%}".format,
            "Test Macro F1": "{:.2%}".format
        }
    )
)

best_name = results_df.iloc[0]["Model"]

best_model = trained_models[best_name]

print("\nBest model:", best_name)


# ============================================================
# 7. EVALUATE THE SELECTED MODEL
# ============================================================

final_predictions = best_model.predict(X_test)

print("\nFinal classification report")
print("=" * 65)

print(
    classification_report(
        y_test,
        final_predictions,
        labels=sorted(y.unique()),
        zero_division=0
    )
)

print("\nConfusion matrix")
print("=" * 65)

matrix = confusion_matrix(
    y_test,
    final_predictions,
    labels=sorted(y.unique())
)

matrix_df = pd.DataFrame(
    matrix,
    index=sorted(y.unique()),
    columns=sorted(y.unique())
)

print(matrix_df)


# ============================================================
# 8. RETRAIN THE SELECTED CONFIGURATION ON ALL DATA
# ============================================================

print("\nTraining the selected model on the complete dataset...")

final_model = Pipeline([
    (
        "tfidf",
        best_model.named_steps["tfidf"]
    ),
    (
        "classifier",
        best_model.named_steps["classifier"]
    )
])

final_model.fit(
    X,
    y
)


# ============================================================
# 9. SAVE THE MODEL
# ============================================================

with open(MODEL_PATH, "wb") as file:
    pickle.dump(final_model, file)

print("\nModel saved successfully!")
print("Location:", MODEL_PATH)


# ============================================================
# 10. TEST NEW COMPLAINT DESCRIPTIONS
# ============================================================

test_examples = [
    (
        "Garbage is overflowing near the school",
        "Garbage Disposal"
    ),
    (
        "The municipal truck has not collected our household waste",
        "Garbage Disposal"
    ),
    (
        "Plastic rubbish is scattered around the public garden",
        "Garbage Disposal"
    ),
    (
        "Sewage is flowing onto the road outside my house",
        "Drainage / Sanitation"
    ),
    (
        "The drain is blocked and rainwater cannot pass",
        "Drainage / Sanitation"
    ),
    (
        "The river has turned black because of factory waste",
        "Water Pollution"
    ),
    (
        "There is a chemical smell coming from the lake",
        "Water Pollution"
    ),
    (
        "Smoke from burning plastic is entering my home",
        "Air Pollution"
    ),
    (
        "Dust from a construction site is affecting nearby residents",
        "Air Pollution"
    ),
    (
        "Loud music from a nearby party is disturbing our sleep",
        "Noise Pollution"
    )
]

print("\nTesting additional complaint descriptions")
print("=" * 65)

correct = 0

for complaint, expected_category in test_examples:

    prediction = final_model.predict([complaint])[0]

    is_correct = prediction == expected_category

    if is_correct:
        correct += 1

    status = "PASS" if is_correct else "CHECK"

    print(f"\nComplaint: {complaint}")
    print(f"Expected:  {expected_category}")
    print(f"Predicted: {prediction}")
    print(f"Result:    {status}")

print("\nAdditional example results")
print("-" * 65)
print(f"Correct predictions: {correct}/{len(test_examples)}")
print(f"Accuracy: {correct / len(test_examples):.2%}")

print("\nTraining process completed!")
print("=" * 65)