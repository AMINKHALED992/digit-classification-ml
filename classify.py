"""Handwritten digit classification (Computer Vision) with scikit-learn.
Dataset: UCI Optical Recognition of Handwritten Digits (1,797 images, 8x8 pixels, 10 classes),
shipped with scikit-learn, so no download is needed.
Trains three models, compares them on a held-out test set, saves a confusion matrix and sample predictions.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, classification_report

SEED = 42
digits = load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=SEED, stratify=y)

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "SVM (RBF kernel)": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale")),
    "Neural network (MLP, 2 hidden layers)": make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=SEED)),
}

results, preds = {}, {}
for name, model in models.items():
    model.fit(X_train, y_train)
    p = model.predict(X_test)
    preds[name] = p
    results[name] = round(float(accuracy_score(y_test, p)), 4)
    print(f"{name}: test accuracy = {results[name]:.4f}")

best = max(results, key=results.get)
print(f"\nBest model: {best}\n")
print(classification_report(y_test, preds[best]))

with open("results.json", "w") as f:
    json.dump({"train_images": len(X_train), "test_images": len(X_test),
               "test_accuracy": results, "best_model": best}, f, indent=2)

fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay.from_predictions(y_test, preds[best], ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Confusion matrix: {best}")
fig.tight_layout(); fig.savefig("confusion_matrix.png", dpi=120)

idx = np.random.RandomState(SEED).choice(len(X_test), 16, replace=False)
fig, axes = plt.subplots(4, 4, figsize=(6, 6))
for a, i in zip(axes.ravel(), idx):
    a.imshow(X_test[i].reshape(8, 8), cmap="gray_r")
    ok = preds[best][i] == y_test[i]
    a.set_title(f"pred {preds[best][i]} / true {y_test[i]}", color="green" if ok else "red", fontsize=8)
    a.axis("off")
fig.tight_layout(); fig.savefig("sample_predictions.png", dpi=120)
