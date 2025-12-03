import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
    classification_report
)
import torch


class MetricsCalculator:
    """Calculate various evaluation metrics"""
    
    def __init__(self, num_classes):
        self.num_classes = num_classes
    
    def calculate_all_metrics(self, y_true, y_pred, y_probs=None):
        """Calculate all metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        if self.num_classes <= 20:  # Avoid huge output for many classes
            metrics['precision_per_class'] = precision_score(y_true, y_pred, average=None, zero_division=0)
            metrics['recall_per_class'] = recall_score(y_true, y_pred, average=None, zero_division=0)
            metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # AUC-ROC if probabilities are provided
        if y_probs is not None and self.num_classes > 1:
            try:
                metrics['auc_ovr'] = roc_auc_score(
                    y_true, y_probs, multi_class='ovr', average='weighted'
                )
                metrics['auc_ovo'] = roc_auc_score(
                    y_true, y_probs, multi_class='ovo', average='weighted'
                )
            except Exception as e:
                metrics['auc_ovr'] = 0.0
                metrics['auc_ovo'] = 0.0
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        return metrics
    
    def print_detailed_report(self, y_true, y_pred, class_names=None):
        """Print detailed classification report"""
        if class_names is None:
            class_names = [f'Class_{i}' for i in range(self.num_classes)]
        
        print("\n" + "="*60)
        print("DETAILED CLASSIFICATION REPORT")
        print("="*60)
        print(classification_report(y_true, y_pred, target_names=class_names))
        
        # Print confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print("\nConfusion Matrix:")
        print(f"{'Predicted':^20}")
        print(" " * 10 + " ".join([f"{name:^8}" for name in class_names]))
        for i, row in enumerate(cm):
            print(f"{class_names[i]:<10} " + " ".join([f"{val:^8}" for val in row]))
    
    def calculate_model_calibration(self, y_true, y_probs, n_bins=10):
        """Calculate model calibration (reliability diagram)"""
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_probs.max(axis=1), bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        calibration_data = []
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.any():
                bin_probs = y_probs[mask]
                bin_true = y_true[mask]
                bin_pred = np.argmax(bin_probs, axis=1)
                
                accuracy = (bin_pred == bin_true).mean()
                confidence = bin_probs.max(axis=1).mean()
                calibration_data.append({
                    'bin': i,
                    'accuracy': accuracy,
                    'confidence': confidence,
                    'count': mask.sum()
                })
        
        return calibration_data


def evaluate_model(model, data_loader, device, return_importances=False):
    """Evaluate model on data loader"""
    model.eval()
    
    all_predictions = []
    all_probabilities = []
    all_targets = []
    all_importances = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            if return_importances:
                outputs, importances = model(inputs, return_importance=True)
                all_importances.append(importances.cpu().numpy())
            else:
                outputs = model(inputs)
            
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.append(predictions.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Concatenate results
    predictions = np.concatenate(all_predictions)
    probabilities = np.concatenate(all_probabilities)
    targets = np.concatenate(all_targets)
    
    if return_importances:
        importances = np.concatenate(all_importances)
        return predictions, probabilities, targets, importances
    
    return predictions, probabilities, targets
