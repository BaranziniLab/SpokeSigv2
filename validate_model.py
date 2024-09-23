import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, f1_score, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
import joblib

def load_data(clin_path, gen_path, endpoints_path):
    """Load and return the clinical, genetic, and endpoint data for validation."""
    return (
        np.load(clin_path),
        np.load(gen_path),
        np.load(endpoints_path)
    )

def load_model(model_path):
    """Load the trained Random Forest model."""
    return joblib.load(model_path)

def combine_data(file_clin, file_gen, weight):
    """Combine clinical and genetic data based on the given weight."""
    return ((1 - weight) * file_clin) + (weight * file_gen)

def evaluate_model(model, X, y):
    """Evaluate the model's performance on the validation set."""
    probs = model.predict_proba(X)
    preds = model.predict(X)
    
    fpr, tpr, _ = roc_curve(y, probs[:, 1])
    roc_auc = auc(fpr, tpr)
    f1 = f1_score(y, preds)
    balanced_acc = balanced_accuracy_score(y, preds)
    
    return fpr, tpr, roc_auc, f1, balanced_acc, probs, preds

def plot_roc_curve(fpr, tpr, roc_auc, weight):
    """Plot ROC curve and save the figure."""
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (Weight: {weight})')
    plt.legend(loc="lower right")
    plt.savefig(f'/pool0/data/archive/ukbiobank_vencida/spoke_signatures_ml_10years/results_validation/roc_curve_validation_weight_{weight}.png')
    plt.close()

def save_results(file_path, data):
    """Save results to a numpy file."""
    np.save(file_path, data)

def main():
    # Load validation data
    file_clin, file_gen, endpoints = load_data(
        'clin_validation.npy',
        'gen_validation.npy',
        'end_points.npy'
    )

    # Load the trained model
    model = load_model('trained_rf_model.joblib')

    weights = list(np.load('optimized_weight.npy'))
    results = []

    for weight in weights:
        print(f"Evaluating model with weight: {weight}")
        X = combine_data(file_clin, file_gen, weight)
        
        fpr, tpr, roc_auc, f1, balanced_acc, probs, preds = evaluate_model(model, X, endpoints)
        
        plot_roc_curve(fpr, tpr, roc_auc, weight)
        
        results.append({
            'weight': weight,
            'auc': roc_auc,
            'f1_score': f1,
            'balanced_accuracy': balanced_acc
        })

        # Save predictions and probabilities
        save_results(f'predictions_weight_{weight}.npy', preds)
        save_results(f'probabilities_weight_{weight}.npy', probs)

    # Save overall results
    results_df = pd.DataFrame(results)
    results_df.to_csv('validation_results.csv', index=False)

    print("Evaluation completed. Results saved.")

if __name__ == "__main__":
    main()
