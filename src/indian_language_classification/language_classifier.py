import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import logging
import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import setup_seed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set seed for reproducibility
setup_seed(42)

def train_svm_classifier(X_train, y_train, X_test, y_test, output_dir, 
                        kernel='rbf', C=1.0):
    """
    Train an SVM classifier on MFCC features
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        output_dir: Directory to save model and results
        kernel: SVM kernel function
        C: Regularization parameter
        
    Returns:
        Trained model and classification report
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create SVM classifier
    svm_clf = SVC(kernel=kernel, C=C, probability=True)
    
    # Train classifier
    logger.info("Training SVM classifier...")
    svm_clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = svm_clf.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    logger.info(f"SVM classifier accuracy: {accuracy:.4f}")
    logger.info("\nClassification Report:\n" + 
               classification_report(y_test, y_pred))
    
    # Save model
    joblib.dump(svm_clf, output_dir / "svm_classifier.pkl")
    
    # Save classification report
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(output_dir / "svm_classification_report.csv")
    
    # Visualize confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("SVM Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(output_dir / "svm_confusion_matrix.png")
    
    return svm_clf, report

def train_random_forest_classifier(X_train, y_train, X_test, y_test, output_dir, 
                                 n_estimators=100, max_depth=None):
    """
    Train a Random Forest classifier on MFCC features
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        output_dir: Directory to save model and results
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of trees
        
    Returns:
        Trained model and classification report
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create Random Forest classifier
    rf_clf = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        random_state=42
    )
    
    # Train classifier
    logger.info("Training Random Forest classifier...")
    rf_clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_clf.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    logger.info(f"Random Forest classifier accuracy: {accuracy:.4f}")
    logger.info("\nClassification Report:\n" + 
               classification_report(y_test, y_pred))
    
    # Save model
    joblib.dump(rf_clf, output_dir / "rf_classifier.pkl")
    
    # Save classification report
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(output_dir / "rf_classification_report.csv")
    
    # Visualize confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Random Forest Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(output_dir / "rf_confusion_matrix.png")
    
    # Feature importance
    if hasattr(rf_clf, 'feature_importances_'):
        n_features = X_train.shape[1]
        plt.figure(figsize=(12, 6))
        
        # First half features are means, second half are variances
        feature_names = [f"Mean MFCC {i+1}" for i in range(n_features//2)] + \
                       [f"Var MFCC {i+1}" for i in range(n_features//2)]
        
        indices = np.argsort(rf_clf.feature_importances_)[::-1]
        
        plt.bar(range(n_features), 
                rf_clf.feature_importances_[indices],
                color='b')
        plt.title("Feature Importance (Random Forest)")
        plt.xlabel("Feature Index")
        plt.ylabel("Importance")
        plt.xticks(range(n_features), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(output_dir / "rf_feature_importance.png")
    
    return rf_clf, report

def train_neural_network_classifier(X_train, y_train, X_test, y_test, output_dir, 
                                 hidden_layer_sizes=(100,), max_iter=200):
    """
    Train a Neural Network classifier on MFCC features
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        output_dir: Directory to save model and results
        hidden_layer_sizes: Hidden layer sizes
        max_iter: Maximum number of iterations
        
    Returns:
        Trained model and classification report
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create Neural Network classifier
    nn_clf = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes, 
        max_iter=max_iter, 
        random_state=42
    )
    
    # Train classifier
    logger.info("Training Neural Network classifier...")
    nn_clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = nn_clf.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    logger.info(f"Neural Network classifier accuracy: {accuracy:.4f}")
    logger.info("\nClassification Report:\n" + 
               classification_report(y_test, y_pred))
    
    # Save model
    joblib.dump(nn_clf, output_dir / "nn_classifier.pkl")
    
    # Save classification report
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(output_dir / "nn_classification_report.csv")
    
    # Visualize confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Neural Network Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(output_dir / "nn_confusion_matrix.png")
    
    # Learning curve if the model has a loss_curve_ attribute
    if hasattr(nn_clf, 'loss_curve_'):
        plt.figure(figsize=(10, 6))
        plt.plot(nn_clf.loss_curve_)
        plt.title("Neural Network Learning Curve")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / "nn_learning_curve.png")
    
    return nn_clf, report

def compare_classifiers(clf_reports, output_dir):
    """
    Compare performance of different classifiers
    
    Args:
        clf_reports: Dictionary of classifier reports
        output_dir: Directory to save comparison results
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compare accuracy
    accuracies = {
        name: report['accuracy'] for name, report in clf_reports.items()
    }
    
    plt.figure(figsize=(10, 6))
    plt.bar(accuracies.keys(), accuracies.values())
    plt.title("Classification Accuracy Comparison")
    plt.xlabel("Classifier")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.0)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_comparison.png")
    
    # Compare macro F1 scores
    f1_scores = {
        name: report['macro avg']['f1-score'] for name, report in clf_reports.items()
    }
    
    plt.figure(figsize=(10, 6))
    plt.bar(f1_scores.keys(), f1_scores.values())
    plt.title("F1 Score Comparison (Macro Average)")
    plt.xlabel("Classifier")
    plt.ylabel("F1 Score")
    plt.ylim(0, 1.0)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "f1_comparison.png")
    
    # Save comparison results
    comparison_df = pd.DataFrame({
        'Accuracy': accuracies,
        'F1 Score (Macro)': f1_scores
    })
    comparison_df.to_csv(output_dir / "classifier_comparison.csv")
    
    # Print comparison results
    logger.info("\nClassifier Comparison:")
    logger.info(comparison_df.to_string())
    
    return comparison_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train language classifiers based on MFCC features")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Directory containing prepared dataset")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save model and results")
    parser.add_argument("--label_map", type=str, default=None,
                        help="Path to label encoder (for interpreting results)")
    
    args = parser.parse_args()
    
    # Load dataset
    X_train = np.load(os.path.join(args.dataset_dir, "X_train.npy"))
    X_test = np.load(os.path.join(args.dataset_dir, "X_test.npy"))
    y_train = np.load(os.path.join(args.dataset_dir, "y_train.npy"))
    y_test = np.load(os.path.join(args.dataset_dir, "y_test.npy"))
    
    # Load label encoder if provided
    if args.label_map:
        label_encoder = joblib.load(args.label_map)
        class_names = label_encoder.classes_
        logger.info(f"Loaded class names: {class_names}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train classifiers
    svm_clf, svm_report = train_svm_classifier(
        X_train, y_train, X_test, y_test, 
        output_dir / "svm"
    )
    
    rf_clf, rf_report = train_random_forest_classifier(
        X_train, y_train, X_test, y_test, 
        output_dir / "random_forest"
    )
    
    nn_clf, nn_report = train_neural_network_classifier(
        X_train, y_train, X_test, y_test, 
        output_dir / "neural_network"
    )
    
    # Compare classifiers
    clf_reports = {
        'SVM': svm_report,
        'Random Forest': rf_report,
        'Neural Network': nn_report
    }
    
    comparison = compare_classifiers(clf_reports, output_dir / "comparison")
