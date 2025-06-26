import os
import cv2
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def line_sweep_features(image_path):
    """Extracts features from an image using the Line Sweep Algorithm."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    _, binary_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    height, width = binary_img.shape
    features = []

    for i in range(height):
        row = binary_img[i, :]
        features.append(np.sum(row[:-1] != row[1:]))

    for j in range(width):
        col = binary_img[:, j]
        features.append(np.sum(col[:-1] != col[1:]))

    for d in range(-height + 1, width):
        diag = np.diag(binary_img, k=d)
        features.append(np.sum(diag[:-1] != diag[1:]))
    
    flipped_img = np.fliplr(binary_img)
    for d in range(-height + 1, width):
        diag = np.diag(flipped_img, k=d)
        features.append(np.sum(diag[:-1] != diag[1:]))
    
    return np.array(features, dtype=np.float32)

def load_dataset(genuine_path, forged_path):
    """Loads dataset and extracts features."""
    genuine_signatures = [os.path.join(genuine_path, f) for f in os.listdir(genuine_path)]
    forged_signatures = [os.path.join(forged_path, f) for f in os.listdir(forged_path)]
    X, y = [], []

    for img_path in genuine_signatures:
        X.append(line_sweep_features(img_path))
        y.append(1)
    for img_path in forged_signatures:
        X.append(line_sweep_features(img_path))
        y.append(0)
    
    return np.array(X), np.array(y)

def train_model(genuine_path, forged_path, model_path="model/svm_signature.pkl"):
    """Trains and saves an SVM model."""
    X, y = load_dataset(genuine_path, forged_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    svm_model = SVC(kernel='linear', probability=True)
    svm_model.fit(X_train, y_train)
    joblib.dump(svm_model, model_path)
    y_pred = svm_model.predict(X_test)
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("Classification Report:\n", classification_report(y_test, y_pred))

def verify_signature(image_path, model_path="model/svm_signature.pkl"):
    """Verifies a signature using the trained model."""
    svm_model = joblib.load(model_path)
    features = line_sweep_features(image_path).reshape(1, -1)
    return "Genuine" if svm_model.predict(features) == 1 else "Forged"

if __name__ == "__main__":
    genuine_path = "dataset/genuine/"
    forged_path = "dataset/forged/"
    model_path = "model/svm_signature.pkl"
    train_model(genuine_path, forged_path, model_path)
    test_image = "test_signature.jpg"
    print("Verification Result:", verify_signature(test_image, model_path))
