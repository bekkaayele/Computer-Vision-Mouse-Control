"""Measures how well the gesture classifier works and saves a confusion matrix.

Run:  python evaluate_model.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write the figure to a file, no window needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split

from gesture_model import GestureClassifier, load_dataset, mirror

SEED = 42
FIGURE_PATH = Path(__file__).parent / "docs" / "confusion_matrix.png"


def cross_validated_accuracy(landmarks, labels, folds=5):
    scores = []
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    for train, test in splitter.split(landmarks, labels):
        model = GestureClassifier().fit(landmarks[train], labels[train])
        scores.append(accuracy_score(labels[test], model.predict(landmarks[test])))
    return float(np.mean(scores)), float(np.std(scores))


def moved_resized_and_tilted(landmarks, rng):
    """The same hands, but elsewhere in the frame, at another size and tilted up to 45 degrees."""
    angle = np.radians(rng.uniform(-45, 45, size=(len(landmarks), 1)))
    x, y = landmarks[:, :, 0], landmarks[:, :, 1]
    changed = landmarks.copy()
    changed[:, :, 0] = x * np.cos(angle) - y * np.sin(angle)
    changed[:, :, 1] = x * np.sin(angle) + y * np.cos(angle)
    scale = rng.uniform(0.5, 1.5, size=(len(landmarks), 1, 1))
    shift = rng.uniform(-0.3, 0.3, size=(len(landmarks), 1, 3))
    return changed * scale + shift


def with_noise(landmarks, rng, fraction=0.05):
    """Every landmark nudged by a random amount, as a stand-in for a different person's hand."""
    palm = np.linalg.norm(landmarks[:, 9, :2] - landmarks[:, 0, :2], axis=1)[:, None, None]
    noisy = landmarks.copy()
    noisy[:, :, :2] += rng.normal(0, fraction, size=(len(landmarks), 21, 2)) * palm
    return noisy


def unknown_pose_rejection(landmarks, labels):
    """Hide one digit from training and count how often it is called "unknown"."""
    rates = []
    for digit in np.unique(labels):
        seen = labels != digit
        model = GestureClassifier().fit(landmarks[seen], labels[seen])
        rates.append(float(np.mean(model.predict(landmarks[~seen]) == -1)))
    return float(np.mean(rates))


def main():
    landmarks, labels = load_dataset()
    print(f"{len(labels)} samples after removing duplicates, {len(np.unique(labels))} classes")

    train_x, test_x, train_y, test_y = train_test_split(
        landmarks, labels, test_size=0.2, stratify=labels, random_state=SEED)
    model = GestureClassifier().fit(train_x, train_y)
    predicted = model.predict(test_x)

    print(f"\nHeld-out test accuracy: {accuracy_score(test_y, predicted):.4f}")
    print(f"Test poses answered with 'unknown': {np.mean(predicted == -1):.2%}")
    print(classification_report(test_y, predicted, labels=np.unique(labels), zero_division=0))

    mean, std = cross_validated_accuracy(landmarks, labels)
    print(f"5-fold cross-validated accuracy: {mean:.4f} (+/- {std:.4f})")

    rng = np.random.default_rng(SEED)
    moved = accuracy_score(test_y, model.predict(moved_resized_and_tilted(test_x, rng)))
    print(f"Test accuracy with hands moved, resized and tilted: {moved:.4f}")
    noisy = accuracy_score(test_y, model.predict(with_noise(test_x, rng)))
    print(f"Test accuracy with every landmark nudged by 5% of the palm length: {noisy:.4f}")
    print(f"Test accuracy on mirrored (other) hands: "
          f"{accuracy_score(test_y, model.predict(mirror(test_x))):.4f}")
    print(f"Unseen digits rejected as unknown: {unknown_pose_rejection(landmarks, labels):.0%}")

    display_labels = ["unknown" if c == -1 else str(c) for c in np.unique(np.concatenate([test_y, predicted]))]
    figure, axis = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        test_y, predicted, display_labels=display_labels, cmap="Blues", colorbar=False, ax=axis)
    axis.set_title("Confusion matrix on the held-out test set")
    axis.set_xlabel("Predicted gesture")
    axis.set_ylabel("Actual gesture")
    figure.tight_layout()
    FIGURE_PATH.parent.mkdir(exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=150)
    print(f"\nConfusion matrix saved to {FIGURE_PATH.relative_to(Path(__file__).parent)}")


if __name__ == "__main__":
    main()
