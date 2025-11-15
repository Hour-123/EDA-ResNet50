"""
Training Script for EDA-ResNet50

This module implements the training logic for the EDA-ResNet50 model
following the exact hyperparameters and training recipe from the paper.
"""

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    ReduceLROnPlateau,
    EarlyStopping,
    TensorBoard,
    CSVLogger
)
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import SkinCancerDataset
from models.eda_resnet50 import create_eda_resnet50, compile_eda_resnet50

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EDAResNet50Trainer:
    """Training class for EDA-ResNet50 model"""

    def __init__(
        self,
        data_dir: str = "/root/EDA-ResNet50/training_data_Skin Cancer_Malignant_vs_Benign",
        experiments_dir: str = "/root/EDA-ResNet50/experiments",
        config: Optional[Dict] = None
    ):
        """Initialize EDA-ResNet50 Trainer"""
        self.data_dir = data_dir
        self.experiments_dir = experiments_dir

        # Create experiments directory structure
        self._create_experiment_dirs()

        # Configuration (paper hyperparameters)
        self.config = self._get_default_config()
        if config:
            self.config.update(config)

        # Initialize components
        self.dataset = None
        self.model = None
        self.train_generator = None
        self.validation_generator = None
        self.test_generator = None
        self.class_weights = None

        logger.info("EDA-ResNet50 Trainer initialized")
        logger.info(f"Configuration: {self.config}")

    def _get_default_config(self) -> Dict:
        """Get default training configuration from the paper"""
        return {
            # Model architecture
            'image_size': (224, 224),
            'batch_size': 20,
            'num_classes': 2,

            # Training hyperparameters (from paper)
            'epochs': 30,
            'initial_learning_rate': 0.0001,
            'min_learning_rate': 1e-6,
            'optimizer': 'adam',
            'loss_function': 'categorical_crossentropy',

            # Learning rate scheduling
            'lr_patience': 2,
            'lr_factor': 0.5,
            'lr_cooldown': 1,

            # Early stopping
            'early_stopping_patience': 8,
            'early_stopping_min_delta': 0.001,

            # Validation split
            'validation_split': 0.0,

            # Data augmentation (paper: no augmentation)
            'augmentation': False,

            # Class weighting
            'use_class_weights': True,
            'class_weight_mode': 'balanced',

            # Model checkpointing
            'save_best_only': True,
            'save_weights_only': False,

            # Random seed for reproducibility
            'random_seed': 42
        }

    def _create_experiment_dirs(self):
        """Create experiment directory structure"""
        dirs = [
            os.path.join(self.experiments_dir, 'models'),
            os.path.join(self.experiments_dir, 'logs'),
            os.path.join(self.experiments_dir, 'plots'),
            os.path.join(self.experiments_dir, 'results'),
            os.path.join(self.experiments_dir, 'checkpoints')
        ]

        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)

        logger.info(f"Experiment directories created in {self.experiments_dir}")

    def setup_data(self):
        """Setup data generators"""
        logger.info("Setting up data generators...")

        # Create dataset
        self.dataset = SkinCancerDataset(
            data_dir=self.data_dir,
            image_size=self.config['image_size'],
            batch_size=self.config['batch_size'],
            random_seed=self.config['random_seed']
        )

        # Create generators
        self.train_generator, self.validation_generator = self.dataset.create_train_generator(
            augmentation=self.config['augmentation'],
            validation_split=self.config['validation_split']
        )

        self.test_generator = self.dataset.create_test_generator()

        # Get class weights if enabled
        if self.config['use_class_weights']:
            self.class_weights = self.dataset.get_class_weights(mode=self.config['class_weight_mode'])

        logger.info("Data generators setup completed")

    def setup_model(self):
        """Setup EDA-ResNet50 model"""
        logger.info("Setting up EDA-ResNet50 model...")

        # Create model
        self.model = create_eda_resnet50(
            num_classes=self.config['num_classes'],
            input_shape=(*self.config['image_size'], 3)
        )

        # Freeze ResNet50 backbone layers for transfer learning
        # Only train the custom attention modules and classifier
        try:
            # Find the EDAResNet50 layer by searching through the model layers
            eda_layer = None
            for layer in self.model.layers:
                if hasattr(layer, 'backbone') and hasattr(layer.backbone, 'resnet50'):
                    eda_layer = layer
                    break

            if eda_layer:
                backbone_model = eda_layer.backbone.resnet50
                backbone_model.trainable = True

                unfrozen_prefixes = ('conv4_block', 'conv5_block')
                for layer in backbone_model.layers:
                    if layer.name.startswith(unfrozen_prefixes):
                        layer.trainable = True
                    else:
                        layer.trainable = False

                logger.info("ResNet50 backbone: conv4/conv5 blocks unfrozen for fine-tuning")

                # Print trainable summary
                trainable_count = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
                total_count = sum([tf.keras.backend.count_params(w) for w in self.model.weights])
                logger.info(f"Trainable parameters: {trainable_count:,} / {total_count:,} ({100*trainable_count/total_count:.2f}%)")
            else:
                logger.warning("EDAResNet50 layer not found, all layers will be trainable")

        except Exception as e:
            logger.warning(f"Could not freeze ResNet50 layers: {e}")
            logger.warning("All layers will be trainable (may cause the gradient warning)")

        # Compile model with paper hyperparameters
        self.model = compile_eda_resnet50(
            self.model,
            learning_rate=self.config['initial_learning_rate']
        )

        # Print model summary
        logger.info("Model architecture:")
        self.model.summary()

        logger.info("Model setup completed")

    def create_callbacks(self, has_validation: bool = True) -> List:
        """Create training callbacks as specified in the paper

        Args:
            has_validation: Whether validation metrics are available for monitoring
        """
        callbacks = []

        # Experiment timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"eda_resnet50_{timestamp}"

        # 1. Model Checkpoint - Save best model
        checkpoint_path = os.path.join(
            self.experiments_dir, 'checkpoints',
            f"{experiment_name}_best.h5"
        )

        if has_validation:
            checkpoint_callback = ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_accuracy',
                save_best_only=self.config['save_best_only'],
                save_weights_only=self.config['save_weights_only'],
                mode='max',
                verbose=1
            )
            callbacks.append(checkpoint_callback)

        # 2. Learning Rate Scheduler - ReduceLROnPlateau (as per paper)
        if has_validation:
            lr_scheduler = ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.config['lr_factor'],
                patience=self.config['lr_patience'],
                min_lr=self.config['min_learning_rate'],
                cooldown=self.config['lr_cooldown'],
                verbose=1,
                mode='min'
            )
            callbacks.append(lr_scheduler)

        # 3. Early Stopping
        if has_validation:
            early_stopping = EarlyStopping(
                monitor='val_accuracy',
                patience=self.config['early_stopping_patience'],
                min_delta=self.config['early_stopping_min_delta'],
                mode='max',
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stopping)

        # 4. TensorBoard logging
        log_dir = os.path.join(self.experiments_dir, 'logs', experiment_name)
        tensorboard_callback = TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch'
        )
        callbacks.append(tensorboard_callback)

        # 5. CSV Logger for training metrics
        csv_path = os.path.join(
            self.experiments_dir, 'results',
            f"{experiment_name}_training_log.csv"
        )
        csv_logger = CSVLogger(
            filename=csv_path,
            append=False
        )
        callbacks.append(csv_logger)

        logger.info(f"Callbacks created for experiment: {experiment_name}")
        return callbacks

    def train(self) -> Dict:
        """Train the EDA-ResNet50 model"""
        logger.info("Starting EDA-ResNet50 training...")
        logger.info(f"Training configuration: {self.config}")

        # Setup components
        self.setup_data()
        self.setup_model()

        # Create callbacks
        has_validation = self.validation_generator is not None
        callbacks = self.create_callbacks(has_validation=has_validation)

        # Calculate number of steps per epoch
        steps_per_epoch = len(self.train_generator)
        validation_steps = len(self.validation_generator) if self.validation_generator else None

        logger.info(f"Training for {self.config['epochs']} epochs")
        logger.info(f"Steps per epoch: {steps_per_epoch}")
        logger.info(f"Validation steps: {validation_steps}")

        # Start training
        start_time = time.time()

        self.history = self.model.fit(
            self.train_generator,
            epochs=self.config['epochs'],
            validation_data=self.validation_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            class_weight=self.class_weights,
            steps_per_epoch=steps_per_epoch,
            verbose=1
        )

        training_time = time.time() - start_time

        logger.info(f"Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

        # Save final model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        models_dir = os.path.join(self.experiments_dir, 'models')

        final_model_path = os.path.join(
            models_dir,
            f"eda_resnet50_final_{timestamp}.keras"
        )
        self.model.save(final_model_path)
        logger.info(f"Final model saved to {final_model_path}")

        weights_path = os.path.join(
            models_dir,
            f"eda_resnet50_weights_{timestamp}.h5"
        )
        self.model.save_weights(weights_path)
        logger.info(f"Model weights saved to {weights_path}")

        return self.history.history

    def evaluate(self) -> Dict:
        """Evaluate the trained model on test set"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        logger.info("Evaluating model on test set...")

        # Reset test generator to start from beginning
        self.test_generator.reset()

        # Evaluate on test set
        test_results = self.model.evaluate(
            self.test_generator,
            verbose=1,
            return_dict=True
        )

        # Get predictions for detailed metrics
        self.test_generator.reset()
        y_pred = self.model.predict(self.test_generator)
        y_true = self.test_generator.classes

        # Calculate additional metrics
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Calculate confusion matrix based metrics
        from sklearn.metrics import confusion_matrix, classification_report

        cm = confusion_matrix(y_true, y_pred_classes)
        tn, fp, fn, tp = cm.ravel()

        # Calculate sensitivity (recall for positive class)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Calculate specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # Calculate precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        # Calculate F1-score
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

        evaluation_results = {
            'test_loss': test_results['loss'],
            'test_accuracy': test_results['accuracy'],
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1_score': f1_score,
            'confusion_matrix': cm.tolist(),
            'classification_report': classification_report(y_true, y_pred_classes,
                                                             target_names=self.dataset.class_names)
        }

        logger.info("Evaluation Results:")
        logger.info(f"  Test Accuracy: {evaluation_results['test_accuracy']:.4f}")
        logger.info(f"  Sensitivity: {evaluation_results['sensitivity']:.4f}")
        logger.info(f"  Specificity: {evaluation_results['specificity']:.4f}")
        logger.info(f"  Precision: {evaluation_results['precision']:.4f}")
        logger.info(f"  F1 Score: {evaluation_results['f1_score']:.4f}")

        # Save evaluation results
        import json
        results_path = os.path.join(
            self.experiments_dir, 'results',
            f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for key, value in evaluation_results.items():
                if isinstance(value, np.ndarray):
                    serializable_results[key] = value.tolist()
                else:
                    serializable_results[key] = value

            json.dump(serializable_results, f, indent=2)

        logger.info(f"Evaluation results saved to {results_path}")

        return evaluation_results


if __name__ == "__main__":
    print("EDA-ResNet50 Training Script")
    print("=" * 50)

    try:
        # Example usage
        trainer = EDAResNet50Trainer()

        # Train and evaluate
        history = trainer.train()
        results = trainer.evaluate()

        print("\nTraining completed successfully!")
        print(f"Final training accuracy: {max(history['accuracy']):.4f}")
        print(f"Final validation accuracy: {max(history['val_accuracy']):.4f}")
        print(f"Test accuracy: {results['test_accuracy']:.4f}")

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()