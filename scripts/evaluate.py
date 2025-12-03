#!/usr/bin/env python3
"""
Evaluation script for Protein FT-Transformer
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.ft_transformer import ProteinFTTransformer
from training.dataloader import DataProcessor
from evaluation.metrics import MetricsCalculator, evaluate_model
from evaluation.visualization import (
    plot_confusion_matrix,
    plot_training_history,
    plot_feature_importance,
    plot_roc_curves
)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Protein FT-Transformer')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test data CSV')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for evaluation results')
    
    args = parser.parse_args()
    
    # Load configuration
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model = ProteinFTTransformer.load(args.model_path, device=device)
    model.to(device)
    model.eval()
    
    # Load and preprocess test data
    print(f"Loading test data from {args.test_data}")
    data_processor = DataProcessor(config)
    
    # Load only test data
    df_test = pd.read_csv(args.test_data, low_memory=False)
    target_col = config['data']['target_column']
    
    if target_col not in df_test.columns:
        raise ValueError(f"Target column '{target_col}' not found in test data")
    
    y_test = df_test[target_col].values
    X_test = df_test.drop(columns=[target_col])
    
    # Preprocess using the same preprocessing as training
    # Note: In practice, you should save and load the preprocessor
    # For simplicity, we preprocess here
    data_dict = data_processor.preprocess_data(
        pd.DataFrame(np.zeros((1, len(X_test.columns))), columns=X_test.columns),  # Dummy train
        np.zeros(1),  # Dummy labels
        X_test,
        y_test
    )
    
    X_test_processed = data_dict['X_test']
    y_test_processed = data_dict['y_test']
    
    # Create test dataset and loader
    test_dataset = data_processor.create_datasets(
        {'X_test': X_test_processed, 'y_test': y_test_processed},
        validation_split=0
    )[2]  # Returns (None, None, test_dataset)
    
    test_loader = data_processor.create_dataloaders(
        None, None, test_dataset,
        batch_size=config['training']['batch_size']
    )[2]  # Returns (None, None, test_loader)
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    predictions, probabilities, targets = evaluate_model(
        model, test_loader, device, return_importances=False
    )
    
    # Calculate metrics
    metrics_calculator = MetricsCalculator(model.num_classes)
    metrics = metrics_calculator.calculate_all_metrics(
        targets, predictions, probabilities
    )
    
    # Print results
    print("\n" + "="*60)
    print("TEST SET EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision (Macro): {metrics['precision_macro']:.4f}")
    print(f"  Recall (Macro): {metrics['recall_macro']:.4f}")
    print(f"  F1-Score (Macro): {metrics['f1_macro']:.4f}")
    print(f"  F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
    
    if 'auc_ovr' in metrics:
        print(f"  AUC-ROC (OVR): {metrics['auc_ovr']:.4f}")
        print(f"  AUC-ROC (OVO): {metrics['auc_ovo']:.4f}")
    
    # Detailed report
    if model.num_classes <= 20:
        metrics_calculator.print_detailed_report(
            targets, predictions, data_dict['class_names']
        )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_path = os.path.join(args.output_dir, 'test_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nMetrics saved to: {metrics_path}")
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'true_label': targets,
        'predicted_label': predictions,
        'true_class': data_dict['class_names'][targets],
        'predicted_class': data_dict['class_names'][predictions]
    })
    
    # Add probabilities for each class
    for i, class_name in enumerate(data_dict['class_names']):
        predictions_df[f'prob_{class_name}'] = probabilities[:, i]
    
    predictions_path = os.path.join(args.output_dir, 'test_predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to: {predictions_path}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Confusion matrix
    if model.num_classes <= 20:
        plot_confusion_matrix(
            targets, predictions, data_dict['class_names'],
            save_path=os.path.join(args.output_dir, 'confusion_matrix.png')
        )
    
    # Feature importance (if model supports it)
    try:
        # Get feature importance for a few samples
        sample_batch = next(iter(test_loader))[0][:10].to(device)
        importances = model.get_feature_importance(sample_batch)
        
        plot_feature_importance(
            importances.mean(0).cpu().numpy(),
            feature_names=data_dict['feature_names'],
            top_n=min(20, len(data_dict['feature_names'])),
            save_path=os.path.join(args.output_dir, 'feature_importance.png')
        )
    except:
        print("Feature importance not available or error in calculation")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()
