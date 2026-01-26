import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold
from bovw import BOVW
from utils import load_dataset, extract_descriptors, extract_bovw_histograms
import os

def train_evaluate(train_hist, train_labels, val_hist, val_labels, classifier_type="logreg", **kwargs):
    if classifier_type == "logreg":
        clf = LogisticRegression(class_weight="balanced", **kwargs)
    elif classifier_type == "svm_linear":
        clf = SVC(kernel="linear", **kwargs)
    elif classifier_type == "svm_rbf":
        clf = SVC(kernel="rbf", **kwargs)
    else:
        raise ValueError(f"Unknown classifier: {classifier_type}")
    
    clf.fit(train_hist, train_labels)
    train_acc = accuracy_score(train_labels, clf.predict(train_hist))
    val_acc = accuracy_score(val_labels, clf.predict(val_hist))
    
    return train_acc, val_acc, clf

def cross_validate(dataset, bovw_params, classifier_params, n_splits=5, classifier_type="logreg"):
    # Extraer labels para el split estratificado
    labels = [label for _, label in dataset]

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(dataset, labels)):
        print(f"\nFold {fold_idx + 1}/{n_splits}")
        
        train_set = [dataset[i] for i in train_idx]
        val_set = [dataset[i] for i in val_idx]
        
        bovw = BOVW(**bovw_params)
        train_desc, train_labels = extract_descriptors(bovw, train_set, "Extracting train descriptors")
        
        bovw._update_fit_codebook(train_desc)
        train_hist = extract_bovw_histograms(bovw, train_desc, fit_pca=True)
        
        val_desc, val_labels = extract_descriptors(bovw, val_set, "Extracting val descriptors")
        val_hist = extract_bovw_histograms(bovw, val_desc, fit_pca=False)
        
        train_acc, val_acc, _ = train_evaluate(train_hist, train_labels, 
                                               val_hist, val_labels, 
                                               classifier_type, **classifier_params)
        
        fold_results.append({"train": train_acc, "val": val_acc})
        print(f"Train: {train_acc:.4f}, Val: {val_acc:.4f}")
    
    mean_val = np.mean([r["val"] for r in fold_results])
    std_val = np.std([r["val"] for r in fold_results])
    
    return mean_val, std_val

def test_final(train_set, test_set, bovw_params, classifier_params, classifier_type="logreg"):
    bovw = BOVW(**bovw_params)
    train_desc, train_labels = extract_descriptors(bovw, train_set, "Extracting train descriptors")
    
    bovw._update_fit_codebook(train_desc)
    train_hist = extract_bovw_histograms(bovw, train_desc, fit_pca=True)
    
    test_desc, test_labels = extract_descriptors(bovw, test_set, "Extracting test descriptors")
    test_hist = extract_bovw_histograms(bovw, test_desc, fit_pca=False)
    
    train_acc, test_acc, clf = train_evaluate(train_hist, train_labels,
                                              test_hist, test_labels,
                                              classifier_type, **classifier_params)
    
    return train_acc, test_acc, bovw, clf

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "..", "data", "MIT_split")

    train_data = load_dataset(os.path.join(DATA_DIR, "train"))
    test_data  = load_dataset(os.path.join(DATA_DIR, "test"))
    classifier_type = 'logreg'

    print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")

    bovw_params = {
        "detector_type": "SIFT",
        "codebook_size": 200,
        "dense_sift": True,
        "sift_step": 10,
        "sift_scales": 2,
        "pca_components": 0.90, # Can specify number of components (int) or retained variance (float)
    }

    classifier_params = {}

    print("\n=== Cross-Validation ===")
    print(f"\n=== CLASSIFIER : {classifier_type} ===")
    mean_acc, std_acc = cross_validate(train_data, bovw_params, classifier_params,
                                       n_splits=5, classifier_type=classifier_type)
    print(f"\nMean CV Accuracy: {mean_acc:.4f} +/- {std_acc:.4f}")

    print("\n=== Final Test ===")
    train_acc, test_acc, bovw, clf = test_final(train_data, test_data,
                                                bovw_params, classifier_params,
                                                classifier_type=classifier_type)
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")