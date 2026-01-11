import numpy as np
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time

def load_data(path):
    data = np.load(path)
    return data['embeddings'], data['labels']

def run_task4():
    print("--- TASK 4: SVM vs End-to-End ---")
    
    # 1. Load data
    print("Loading embeddings...")
    X_train, y_train = load_data("/ghome/group07/week2/runs/week2_group07_97291/week2_group07_97291best.pttrain_embeddings.npz")
    X_test, y_test = load_data("/ghome/group07/week2/runs/week2_group07_97291/week2_group07_97291best.pttest_embeddings.npz")
    
    print(f"Train Data: {X_train.shape}")
    print(f"Test Data:  {X_test.shape}")

    # 2. Normalization (CRITICAL for SVM to work well)
    print("Normalizing features (StandardScaler)...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 3. Train SVM
    # Using RBF kernel which usually works better than linear for complex features
    print("Training SVM (RBF Kernel)...")
    start_time = time.time()
    svm = SVC(kernel='rbf', C=1.0, verbose=False)
    svm.fit(X_train, y_train)
    print(f"Training completed in {time.time() - start_time:.2f} seconds.")

    # 4. Evaluate
    print("Evaluating predictions...")
    train_pred = svm.predict(X_train)
    test_pred = svm.predict(X_test)

    acc_train = accuracy_score(y_train, train_pred)
    acc_test = accuracy_score(y_test, test_pred)

    print("\n" + "="*40)
    print(f"TASK 4 RESULTS")
    print("="*40)
    print(f"SVM Train Accuracy: {acc_train:.4f} ({acc_train*100:.2f}%)")
    print(f"SVM Test Accuracy:  {acc_test:.4f} ({acc_test*100:.2f}%)")
    print("-" * 40)
    print(f"End-to-End Train Accuracy: 0.3264 (32.64%)")
    print(f"End-to-End Test Accuracy:  0.2800 (28.00%)")
    print("="*40)

if __name__ == "__main__":
    run_task4()