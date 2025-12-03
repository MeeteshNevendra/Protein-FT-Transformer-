#!/usr/bin/env python3
"""
Training script for Protein FT-Transformer
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.ft_transformer import ProteinFTTransformer
from training.dataloader import DataProcessor
from training.trainer import Trainer
from utils.config_loader import load_config
import warnings
warnings.filterwarnings('ignore')


def set_seed(seed):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='Train Protein FT-Transformer')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--train_data', type=str, 
                       help='Path to training data CSV')
    parser.add_argument('--test_data', type=str,
                       help='Path to test data CSV')
    parser.add_argument('--target', type=str,
                       help='Target column name')
    parser.add_argument('--epochs', type=int,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float,
                       help='Learning rate')
    parser.add_argument('--output_dir', type=str,
                       help='Output directory for models and results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.train_data:
        config['data']['train_path'] = args.train_data
    if args.test_data:
        config['data']['test_path'] = args.test_data
    if args.target:
        config['data']['target_column'] = args.target
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if args.output_dir:
        config['output']['model_save_dir'] = os.path.join(args.output_dir, 'models')
        config['output']['results_save_dir'] = os.path.join(args.output_dir, 'results')
    
    # Set random seed
    set_seed(args.seed)
    config['seed'] = args.seed
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load and preprocess data
    print("\n" + "="*60)
    print("LOADING AND PREPROCESSING DATA")
    print("="*60)
    
    data_processor = DataProcessor(config)
    data_dict = data_processor.load_data(
        config['data']['train_path'],
        config['data']['test_path']
    )
    
    print(f"Training samples: {len(data_dict['X_train'])}")
    print(f"Test samples: {len(data_dict['X_test']) if data_dict['X_test'] is not None else 0}")
    print(f"Number of features: {data_dict['X_train'].shape[1]}")
    print(f"Number of classes: {len(data_dict['class_names'])}")
    print(f"Classes: {data_dict['class_names']}")
    
    # Create datasets and dataloaders
    train_dataset, val_dataset, test_dataset = data_processor.create_datasets(
        data_dict,
        validation_split=config['data']['validation_split']
    )
    
    train_loader, val_loader, test_loader = data_processor.create_dataloaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=config['training']['batch_size']
    )
    
    print(f"\nData loaders created:")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader) if val_loader else 0}")
    print(f"  Test batches: {len(test_loader) if test_loader else 0}")
    
    # Create model
    print("\n" + "="*60)
    print("CREATING MODEL")
    print("="*60)
    
    input_dim = data_dict['X_train'].shape[1]
    num_classes = len(data_dict['class_names'])
    
    model = ProteinFTTransformer(
        input_dim=input_dim,
        num_classes=num_classes,
        d_token=config['model']['d_token'],
        n_head=config['model']['n_head'],
        n_layers=config['model']['n_layers'],
        dropout=config['model']['dropout'],
        ff_dim_factor=config['model']['ff_dim_factor']
    ).to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Summary:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / (1024**2):.2f} MB (FP32)")
    
    # Train model
    trainer = Trainer(model, train_loader, val_loader, config, device)
    history = trainer.train()
    
    # Save final model
    final_model_path = os.path.join(
        config['output']['model_save_dir'],
        'final_model.pth'
    )
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    # Save training history
    history_path = os.path.join(
        config['output']['results_save_dir'],
        'training_history.npy'
    )
    np.save(history_path, history)
    print(f"Training history saved to: {history_path}")
    
    # Save configuration
    config_path = os.path.join(
        config['output']['results_save_dir'],
        'training_config.yaml'
    )
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    print(f"Configuration saved to: {config_path}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("="*60)


if __name__ == "__main__":
    main()
