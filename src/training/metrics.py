"""
Custom Metrics for EDA-ResNet50 Training

This module implements custom metrics for monitoring during training,
specifically sensitivity and specificity as mentioned in the paper.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import os
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ArgmaxRecallMetric(tf.keras.metrics.Recall):
    """
    Recall metric that operates on one-hot encoded targets by using argmax to
    recover class indices before delegating to the built-in Recall metric.
    """

    def __init__(self, class_id=1, name='argmax_recall', **kwargs):
        super().__init__(class_id=class_id, name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_labels = tf.argmax(y_true, axis=-1)
        y_pred_labels = tf.argmax(y_pred, axis=-1)
        return super().update_state(y_true_labels, y_pred_labels, sample_weight)


class SensitivitySpecificityMetrics(Callback):
    """
    Custom callback to calculate and log sensitivity and specificity during training
    """

    def __init__(self, validation_data=None, log_dir=None, experiment_name=None):
        """Initialize metrics callback"""
        super(SensitivitySpecificityMetrics, self).__init__()
        self.validation_data = validation_data
        self.log_dir = log_dir
        self.experiment_name = experiment_name

        # Metrics storage
        self.metrics_history = {
            'epoch': [],
            'sensitivity': [],
            'specificity': []
        }

        # Paper targets for comparison
        self.paper_targets = {
            'accuracy': 0.9318,
            'sensitivity': 0.94,
            'specificity': 0.925
        }

    def on_epoch_end(self, epoch, logs=None):
        """Calculate metrics at end of each epoch"""
        if self.validation_data is None:
            return

        # Get predictions on validation set
        y_true = []
        y_pred = []

        # Reset validation generator
        self.validation_data.reset()

        # Collect predictions (sample to avoid too much computation)
        max_batches = min(10, len(self.validation_data))

        for i in range(max_batches):
            x_batch, y_batch = next(self.validation_data)
            batch_pred = self.model.predict(x_batch, verbose=0)

            y_true.extend(np.argmax(y_batch, axis=1))
            y_pred.extend(np.argmax(batch_pred, axis=1))

        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Calculate metrics
        sensitivity, specificity = self._calculate_sensitivity_specificity(y_true, y_pred)

        # Store metrics
        self.metrics_history['epoch'].append(epoch + 1)
        self.metrics_history['sensitivity'].append(sensitivity)
        self.metrics_history['specificity'].append(specificity)

        # Log metrics
        logger.info(f"Epoch {epoch + 1} - Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")

        # Compare with paper targets
        self._compare_with_paper_targets(sensitivity, specificity)

        # Save metrics to file
        self._save_metrics()

    def _calculate_sensitivity_specificity(self, y_true, y_pred):
        """Calculate sensitivity and specificity"""
        # True positives, false negatives, true negatives, false positives
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred != 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred != 1))

        # Calculate metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        return sensitivity, specificity

    def _compare_with_paper_targets(self, sensitivity, specificity):
        """Compare current metrics with paper targets"""
        sens_diff = sensitivity - self.paper_targets['sensitivity']
        spec_diff = specificity - self.paper_targets['specificity']

        if sensitivity >= self.paper_targets['sensitivity']:
            logger.info(f"   Sensitivity {sensitivity:.4f} meets/exceeds paper target {self.paper_targets['sensitivity']}")
        else:
            logger.info(f"   Sensitivity {sensitivity:.4f} below paper target {self.paper_targets['sensitivity']} (diff: {sens_diff:.4f})")

        if specificity >= self.paper_targets['specificity']:
            logger.info(f"   Specificity {specificity:.4f} meets/exceeds paper target {self.paper_targets['specificity']}")
        else:
            logger.info(f"   Specificity {specificity:.4f} below paper target {self.paper_targets['specificity']} (diff: {spec_diff:.4f})")

    def _save_metrics(self):
        """Save metrics history to file"""
        if self.log_dir and self.experiment_name:
            metrics_path = os.path.join(
                self.log_dir,
                f"{self.experiment_name}_metrics.json"
            )

            with open(metrics_path, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)


if __name__ == "__main__":
    print("Metrics module test")
    print("=" * 30)

    # Test callback with dummy data
    class DummyModel:
        def predict(self, x, verbose=0):
            # Return random predictions for testing
            return np.random.random((len(x), 2))

    # Create dummy data
    dummy_val_data = type('DummyGenerator', (), {})()
    dummy_val_data.reset = lambda: None
    dummy_val_data.__len__ = lambda: 5

    def dummy_next(self):
        # Return dummy batch
        x = np.random.random((32, 224, 224, 3))
        y = np.random.random((32, 2))
        y = np.argmax(y, axis=1)
        y_onehot = np.zeros((32, 2))
        y_onehot[np.arange(32), y] = 1
        return x, y_onehot

    dummy_val_data.next = dummy_next

    # Test callback
    callback = SensitivitySpecificityMetrics(
        validation_data=dummy_val_data,
        log_dir="/tmp",
        experiment_name="test"
    )

    # Simulate epoch end
    callback.model = DummyModel()
    callback.on_epoch_end(0)

    print("Metrics module test completed successfully!")