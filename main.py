"""
Main Training Script for EDA-ResNet50
"""

import os
import sys
import logging
import argparse
from datetime import datetime

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.training.trainer import EDAResNet50Trainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main training function"""
    print("=" * 60)
    print("EDA-ResNet50 Skin Cancer Classification")
    print("=" * 60)
    print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Parse arguments
    parser = argparse.ArgumentParser(description='Train EDA-ResNet50 model')

    parser.add_argument('--data-dir', type=str,
                       default="/root/EDA-ResNet50/training_data_Skin Cancer_Malignant_vs_Benign",
                       help='Path to dataset directory')

    parser.add_argument('--experiments-dir', type=str,
                       default="/root/EDA-ResNet50/experiments",
                       help='Path to experiments directory')

    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')

    parser.add_argument('--batch-size', type=int, default=20,
                       help='Batch size (default: 20)')

    parser.add_argument('--learning-rate', type=float, default=0.0001,
                       help='Initial learning rate (default: 0.0001)')

    parser.add_argument('--test-only', action='store_true',
                       help='Only evaluate on test set (skip training)')

    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with fewer epochs')

    args = parser.parse_args()

    # Print configuration
    print("Configuration:")
    print(f"  Data Directory: {args.data_dir}")
    print(f"  Experiments Directory: {args.experiments_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.learning_rate}")
    print()

    # Create configuration
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'initial_learning_rate': args.learning_rate
    }

    # Quick test configuration
    if args.quick_test:
        config.update({
            'epochs': 2,
            'validation_split': 0.1,
            'early_stopping_patience': 1
        })
        print("Quick test mode: Reduced epochs for testing")
        print()

    # Initialize trainer
    trainer = EDAResNet50Trainer(
        data_dir=args.data_dir,
        experiments_dir=args.experiments_dir,
        config=config
    )

    if args.test_only:
        # Test only mode
        print("Test-only mode: Setting up data and model...")
        trainer.setup_data()
        trainer.setup_model()

        print("Evaluating model...")
        results = trainer.evaluate()
        print()

        print("Evaluation Results:")
        print(f"  Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"  Sensitivity: {results['sensitivity']:.4f}")
        print(f"  Specificity: {results['specificity']:.4f}")

        # Check against paper targets
        paper_targets = {
            'accuracy': 0.9318,
            'sensitivity': 0.94,
            'specificity': 0.925
        }

        print("\nComparison with Paper Targets:")
        metric_key_map = {
            'accuracy': 'test_accuracy',
            'sensitivity': 'sensitivity',
            'specificity': 'specificity'
        }
        for metric, target in paper_targets.items():
            value = results.get(metric_key_map.get(metric, metric), 0)
            diff = value - target
            status = "✓" if value >= target else "✗"
            print(f"  {metric}: {value:.4f} (target: {target:.4f}, diff: {diff:+.4f}) {status}")

    else:
        # Training mode
        print("Starting training...")
        try:
            # Train the model
            training_history = trainer.train()

            print("\nTraining completed successfully!")
            if 'accuracy' in training_history:
                print(f"Final training accuracy: {max(training_history['accuracy']):.4f}")
            else:
                print("Training accuracy history not available.")

            if 'val_accuracy' in training_history:
                print(f"Final validation accuracy: {max(training_history['val_accuracy']):.4f}")
            else:
                print("Validation accuracy not available (no validation split configured).")

            # Evaluate the model
            print("\nEvaluating model on test set...")
            evaluation_results = trainer.evaluate()

            print("Final Evaluation Results:")
            print(f"  Test Accuracy: {evaluation_results['test_accuracy']:.4f}")
            print(f"  Sensitivity: {evaluation_results['sensitivity']:.4f}")
            print(f"  Specificity: {evaluation_results['specificity']:.4f}")

            # Check against paper targets
            paper_targets = {
                'accuracy': 0.9318,
                'sensitivity': 0.94,
                'specificity': 0.925
            }

            print("\nComparison with Paper Targets:")
            metric_key_map = {
                'accuracy': 'test_accuracy',
                'sensitivity': 'sensitivity',
                'specificity': 'specificity'
            }
            paper_achieved = True
            for metric, target in paper_targets.items():
                value = evaluation_results.get(metric_key_map.get(metric, metric), 0)
                diff = value - target
                status = "✓" if value >= target else "✗"
                print(f"  {metric}: {value:.4f} (target: {target:.4f}, diff: {diff:+.4f}) {status}")
                if value < target:
                    paper_achieved = False

            print(f"\nPaper Reproduction Status: {'✓ SUCCESS' if paper_achieved else '✗ PARTIAL'}")

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        except Exception as e:
            print(f"\nError during training: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nScript completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    main()