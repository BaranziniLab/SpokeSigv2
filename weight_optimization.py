import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, f1_score, balanced_accuracy_score
from imblearn.ensemble import BalancedRandomForestClassifier
from timeit import default_timer as timer

def load_data(clin_path, gen_path, endpoints_path):
    """Load and return the clinical, genetic, and endpoint data."""
    return (
        np.load(clin_path),
        np.load(gen_path),
        np.load(endpoints_path)
    )

def combine_data(file_clin, file_gen, weight):
    """Combine clinical and genetic data based on the given weight."""
    return ((1 - weight) * file_clin) + (weight * file_gen)

def train_and_evaluate(X, y, n_splits=10):
    """Train the model and evaluate its performance."""
    clf = BalancedRandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    tprs, aucs, f1_scores, balanced_accuracies = [], [], [], []
    feature_importances = []
    mean_fpr = np.linspace(0, 1, 100)

    for i in range(n_splits):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
        
        start = timer()
        clf.fit(X_train, y_train)
        feature_importances.append(clf.feature_importances_)
        
        probs = clf.predict_proba(X_test)
        preds = clf.predict(X_test)
        
        fpr, tpr, _ = roc_curve(y_test, probs[:, 1])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        
        f1 = f1_score(y_test, preds)
        f1_scores.append(f1)
        
        balanced_acc = balanced_accuracy_score(y_test, preds)
        balanced_accuracies.append(balanced_acc)
        
        end = timer()
        print(f"Fold {i+1} completed in {end - start:.2f} seconds")

    return tprs, aucs, f1_scores, balanced_accuracies, feature_importances, mean_fpr

def plot_roc_curve(tprs, aucs, mean_fpr, weight):
    """Plot ROC curve and save the figure."""
    plt.figure()
    for i, (tpr, roc_auc) in enumerate(zip(tprs, aucs), 1):
        plt.plot(mean_fpr, tpr, lw=2, alpha=0.3, 
                 label=f'ROC fold {i} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black')
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='blue',
             label=f'Mean ROC (AUC = {mean_auc:.2f})', lw=2, alpha=1)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (Weight: {weight})')
    plt.legend(loc="lower right")
    plt.savefig(f'/pool0/data/archive/ukbiobank_vencida/spoke_signatures_ml_10years/results_prediction/roc_curve_combined_10k_norm_finito_{weight}')
    plt.close()

    return mean_auc

def save_results(file_path, data):
    """Save results to a numpy file."""
    np.save(file_path, data)

def main():
    # Load data
    file_clin, file_gen, endpoints = load_data(
        'clin_train.npy',
        'gen_train.npy',
        'ms_train.npy'
    )

    weights = np.linspace(0, 1, 11)
    auc_weights, f1_scores, balanced_accuracies, feature_importances = [], [], [], []

    for weight in weights:
        print(f"Processing weight: {weight}")
        X = combine_data(file_clin, file_gen, weight)
        
        tprs, aucs, fold_f1_scores, fold_balanced_accuracies, fold_feature_importances, mean_fpr = train_and_evaluate(X, endpoints)
        
        mean_auc = plot_roc_curve(tprs, aucs, mean_fpr, weight)
        
        auc_weights.append(mean_auc)
        f1_scores.append(fold_f1_scores)
        balanced_accuracies.append(fold_balanced_accuracies)
        feature_importances.extend(fold_feature_importances)

    # Save results
    save_results('/f1_results', f1_scores)
    save_results('/scores_combined_gradient', auc_weights)
    save_results('/balanced_accuracies', balanced_accuracies)
    save_results('/combined_featureimp_normed_finito', feature_importances)

if __name__ == "__main__":
    main()
