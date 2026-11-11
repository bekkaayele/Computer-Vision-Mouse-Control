"""Gesture classifier: turns 21 MediaPipe hand landmarks into an ASL digit (0-9).

The same feature code is used for training and at run time, so the model
always sees landmarks in the form it was trained on.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

DATASET_PATH = Path(__file__).parent / "data" / "ASL_numbers_dataset.csv"

WRIST = 0
MIDDLE_MCP = 9  # knuckle at the base of the middle finger

N_NEIGHBORS = 5
# A pose is "unknown" when it is farther from its neighbours than this
# percentage of the training poses are from theirs.
REJECT_PERCENTILE = 99


def landmarks_to_features(landmarks):
    """Make landmarks independent of where the hand is and how big it looks.

    landmarks: array of shape (21, 2+) or (n, 21, 2+) with x, y in any unit.
    Returns an array of shape (42,) or (n, 42).

    The wrist is moved to the origin and every point is divided by the palm
    length (wrist to middle knuckle). Only x and y are used: MediaPipe's z is
    a rough estimate and adds more noise than information here.
    """
    points = np.asarray(landmarks, dtype=float)
    single = points.ndim == 2
    if single:
        points = points[None]
    points = points[:, :, :2]
    points = points - points[:, [WRIST], :]
    palm = np.linalg.norm(points[:, MIDDLE_MCP, :], axis=1)
    palm = np.where(palm < 1e-9, 1.0, palm)  # degenerate hand: avoid dividing by zero
    features = (points / palm[:, None, None]).reshape(len(points), -1)
    return features[0] if single else features


def load_dataset(path=DATASET_PATH):
    """Read the landmark CSV and return (landmarks (n, 21, 3), labels (n,)).

    Exact duplicate rows are dropped so the same sample cannot end up in both
    the training and the test set.
    """
    df = pd.read_csv(path).drop_duplicates()
    labels = df["Label"].to_numpy()
    landmarks = df.drop(columns="Label").to_numpy().reshape(-1, 21, 3)
    return landmarks, labels


def mirror(landmarks):
    """Flip hands left-right, turning right-hand samples into left-hand ones."""
    flipped = np.array(landmarks, dtype=float)
    flipped[..., 0] = -flipped[..., 0]
    return flipped


class GestureClassifier:
    """K-nearest-neighbours on normalised landmarks, with an "unknown" answer."""

    def __init__(self, n_neighbors=N_NEIGHBORS, reject_percentile=REJECT_PERCENTILE):
        self.n_neighbors = n_neighbors
        self.reject_percentile = reject_percentile
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.reject_distance = np.inf

    def fit(self, landmarks, labels):
        # The dataset was recorded with one hand; adding mirrored copies makes
        # the model work for the other hand too.
        features = landmarks_to_features(np.concatenate([landmarks, mirror(landmarks)]))
        labels = np.concatenate([labels, labels])
        self.knn.fit(features, labels)

        # How far is a normal training pose from its neighbours? (The first
        # neighbour of a training point is itself, so it is skipped.)
        distances, _ = self.knn.kneighbors(features, n_neighbors=self.n_neighbors + 1)
        typical = distances[:, 1:].mean(axis=1)
        self.reject_distance = float(np.percentile(typical, self.reject_percentile))
        return self

    def predict(self, landmarks):
        """Return one label per hand; -1 means "not a gesture I know"."""
        features = np.atleast_2d(landmarks_to_features(landmarks))
        distances, _ = self.knn.kneighbors(features)
        labels = self.knn.predict(features)
        return np.where(distances.mean(axis=1) > self.reject_distance, -1, labels)

    def predict_one(self, landmarks):
        """Return the digit for a single hand, or None if the pose is unknown."""
        label = int(self.predict(landmarks)[0])
        return None if label == -1 else label


def build_classifier(path=DATASET_PATH):
    """Train the classifier on the whole dataset (takes well under a second)."""
    landmarks, labels = load_dataset(path)
    return GestureClassifier().fit(landmarks, labels)
