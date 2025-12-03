#!/usr/bin/env python3
"""
Main entry point for Protein FT-Transformer
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description='Protein FT-Transformer: Multiclass Protein Classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train --config config.yaml
  python main.py evaluate --model saved_models/best_model.pth --data test.csv
  python main.py predict --model saved_models/best_model.pth --input new_data.csv
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--config', type=str, default='config.yaml',
                            help='Configuration file')
    train_parser.add_argument('--train_data', type=str,
                            help='Training data CSV')
    train_parser.add_argument('--test_data', type=str,
                            help='Test data CSV')
    train_parser.add_argument('--epochs', type=int,
                            help='Number of epochs')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    eval_parser.add_argument('--model', type=str, required=True,
                           help='Path to trained model')
    eval_parser.add_argument('--data', type=str, required=True,
                           help='Test data CSV')
    eval_parser.add_argument('--output', type=str, default='evaluation_results',
                           help='Output directory')
    
    # Predict command
    pred_parser = subparsers.add_parser('predict', help='Make predictions')
    pred_parser.add_argument('--model', type=str, required=True,
                           help='Path to trained model')
    pred_parser.add_argument('--input', type=str, required=True,
                           help='Input data CSV')
    pred_parser.add_argument('--output', type=str, default='predictions.csv',
                           help='Output CSV file')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        from scripts.train import main as train_main
        train_args = [
            '--config', args.config,
        ]
        if args.train_data:
            train_args.extend(['--train_data', args.train_data])
        if args.test_data:
            train_args.extend(['--test_data', args.test_data])
        if args.epochs:
            train_args.extend(['--epochs', str(args.epochs)])
        
        # Convert to proper argparse format
        import sys
        sys.argv = [sys.argv[0]] + train_args
        train_main()
    
    elif args.command == 'evaluate':
        from scripts.evaluate import main as eval_main
        import sys
        sys.argv = [
            sys.argv[0],
            '--model_path', args.model,
            '--test_data', args.data,
            '--output_dir', args.output
        ]
        eval_main()
    
    elif args.command == 'predict':
        from scripts.predict import main as pred_main
        import sys
        sys.argv = [
            sys.argv[0],
            '--model_path', args.model,
            '--input_file', args.input,
            '--output_file', args.output
        ]
        pred_main()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
