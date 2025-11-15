"""
EDA-ResNet50: Complete Model Architecture

This module implements the complete EDA-ResNet50 model that combines:
1. ResNet50 Backbone (pre-trained on ImageNet)
2. MFR Module (Multi-Scale Feature Representation)
3. Efficient Module (Efficient Channel Attention)
4. DA Module (Dual Attention)
5. Classification Head (GAP + Dense + Softmax)

The model is designed for skin cancer classification (benign vs malignant).
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model

# Import custom modules
try:
    # Try relative imports (when imported as part of package)
    from .backbone import ResNet50FeatureExtractor
    from .mfr_module import MFRModule
    from .efficient_module import EfficientModuleSimplified
    from .da_module import DualAttentionModule
    from ..training.metrics import ArgmaxRecallMetric
except (ImportError, ValueError):
    try:
        # Fallback when accessed as models.*
        from models.backbone import ResNet50FeatureExtractor
        from models.mfr_module import MFRModule
        from models.efficient_module import EfficientModuleSimplified
        from models.da_module import DualAttentionModule
        from training.metrics import ArgmaxRecallMetric
    except ImportError:
        # Final fallback for running from project root
        from src.models.backbone import ResNet50FeatureExtractor
        from src.models.mfr_module import MFRModule
        from src.models.efficient_module import EfficientModuleSimplified
        from src.models.da_module import DualAttentionModule
        from src.training.metrics import ArgmaxRecallMetric


class EDAResNet50(Model):
    """
    Complete EDA-ResNet50 Model Architecture

    Architecture flow:
    Input (224x224x3) � ResNet50 Backbone � MFR Module � Efficient Module � DA Module � GAP � Dense(2) � Softmax
    """

    def __init__(self, num_classes=2, input_shape=(224, 224, 3), **kwargs):
        """
        Initialize EDA-ResNet50 Model

        Args:
            num_classes (int): Number of output classes (default: 2 for binary classification)
            input_shape (tuple): Input image shape (default: (224, 224, 3))
            **kwargs: Additional model arguments
        """
        super(EDAResNet50, self).__init__(**kwargs)
        self.num_classes = num_classes
        self._input_shape = input_shape

        # ===== 1. Backbone: ResNet50 =====
        # Extract features from conv5_block1_out layer (typical shape: 7x7x2048)
        self.backbone = ResNet50FeatureExtractor(
            output_layer_name='conv5_block1_out',
            pretrained_weights='imagenet'
        )

        # ===== 2. MFR Module (Multi-Scale Feature Representation) =====
        # Takes ResNet50 output and applies multi-scale feature fusion
        self.mfr_module = MFRModule(filters=64)

        # ===== 3. Efficient Module (Efficient Channel Attention) =====
        # Applies lightweight channel attention mechanism
        self.efficient_module = EfficientModuleSimplified(kernel_size=3)

        # ===== 4. DA Module (Dual Attention) =====
        # Combines channel and spatial attention mechanisms
        self.da_module = DualAttentionModule()

        # ===== 5. Classification Head =====
        # Global Average Pooling + Dense + Softmax
        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.classifier = layers.Dense(
            units=num_classes,
            activation='softmax',
            kernel_initializer='he_normal'
        )

    def call(self, inputs, training=None):
        """
        Forward pass through EDA-ResNet50

        Args:
            inputs: Input tensor of shape (batch_size, 224, 224, 3)
            training: Whether in training mode

        Returns:
            Classification probabilities of shape (batch_size, num_classes)
        """
        # ===== 1. Backbone Feature Extraction =====
        # Input: (batch_size, 224, 224, 3)
        # Output: (batch_size, 7, 7, 2048) - typical ResNet50 conv5_block1_out output
        backbone_features = self.backbone(inputs, training=training)

        # ===== 2. Multi-Scale Feature Representation =====
        # Input: (batch_size, 7, 7, 2048)
        # Output: (batch_size, 7, 7, 64) - MFR module with 64 filters
        mfr_features = self.mfr_module(backbone_features, training=training)

        # ===== 3. Efficient Channel Attention =====
        # Input: (batch_size, 7, 7, 64)
        # Output: (batch_size, 7, 7, 64) - with channel attention applied
        efficient_features = self.efficient_module(mfr_features, training=training)

        # ===== 4. Dual Attention Mechanism =====
        # Input: (batch_size, 7, 7, 64)
        # Output: (batch_size, 7, 7, 64) - with dual attention applied
        da_features = self.da_module(efficient_features, training=training)

        # ===== 5. Classification Head =====
        # Global Average Pooling
        # Input: (batch_size, 7, 7, 64)
        # Output: (batch_size, 64)
        pooled_features = self.global_avg_pool(da_features)

        # Final classification
        # Input: (batch_size, 64)
        # Output: (batch_size, num_classes)
        logits = self.classifier(pooled_features)

        return logits

    def model_summary(self):
        """Print model architecture summary"""
        # Build the model to show summary
        dummy_input = layers.Input(shape=self.input_shape)
        model = Model(inputs=[dummy_input], outputs=self.call(dummy_input))
        model.summary()

    def get_config(self):
        """Get model configuration for serialization"""
        config = super(EDAResNet50, self).get_config()
        config.update({
            'num_classes': self.num_classes,
            'input_shape': self.input_shape,
        })
        return config


class EDAResNet50Alternative(EDAResNet50):
    """
    Alternative EDA-ResNet50 Implementation with different feature extraction point

    Uses conv4_block6_out as output layer for different spatial resolution
    """

    def __init__(self, num_classes=2, input_shape=(224, 224, 3), **kwargs):
        """
        Initialize Alternative EDA-ResNet50 Model

        Args:
            num_classes (int): Number of output classes (default: 2)
            input_shape (tuple): Input image shape (default: (224, 224, 3))
            **kwargs: Additional model arguments
        """
        super(EDAResNet50Alternative, self).__init__(num_classes, input_shape, **kwargs)

        # Override backbone with different output layer
        self.backbone = ResNet50FeatureExtractor(
            output_layer_name='conv4_block6_out',  # Higher spatial resolution
            pretrained_weights='imagenet'
        )

    def call(self, inputs, training=None):
        """
        Forward pass through Alternative EDA-ResNet50

        Args:
            inputs: Input tensor of shape (batch_size, 224, 224, 3)
            training: Whether in training mode

        Returns:
            Classification probabilities of shape (batch_size, num_classes)
        """
        # Alternative flow with higher spatial resolution features
        backbone_features = self.backbone(inputs, training=training)
        mfr_features = self.mfr_module(backbone_features, training=training)
        efficient_features = self.efficient_module(mfr_features, training=training)
        da_features = self.da_module(efficient_features, training=training)
        pooled_features = self.global_avg_pool(da_features)
        logits = self.classifier(pooled_features)

        return logits


def create_eda_resnet50(num_classes=2, input_shape=(224, 224, 3), alternative=False):
    """
    Factory function to create EDA-ResNet50 model

    Args:
        num_classes (int): Number of output classes (default: 2)
        input_shape (tuple): Input image shape (default: (224, 224, 3))
        alternative (bool): Whether to use alternative architecture (default: False)

    Returns:
        EDA-ResNet50 model as Keras Model
    """
    if alternative:
        model = EDAResNet50Alternative(num_classes=num_classes, input_shape=input_shape)
    else:
        model = EDAResNet50(num_classes=num_classes, input_shape=input_shape)

    # Build the model
    inputs = layers.Input(shape=input_shape)
    outputs = model(inputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="EDA-ResNet50")


def compile_eda_resnet50(model, learning_rate=0.0001):
    """
    Compile EDA-ResNet50 model with appropriate optimizer and loss

    Args:
        model: EDA-ResNet50 model to compile
        learning_rate (float): Learning rate for optimizer (default: 0.0001)

    Returns:
        Compiled model
    """
    # Optimizer: Adam with specified learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Loss: Categorical Cross-Entropy (as per paper)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    # Metrics: Accuracy, Sensitivity, Specificity
    metrics = [
        'accuracy',
        ArgmaxRecallMetric(class_id=1, name='sensitivity'),
        ArgmaxRecallMetric(class_id=0, name='specificity'),
    ]

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=metrics
    )

    return model



if __name__ == "__main__":
    # Example usage and testing
    print("Testing EDA-ResNet50 Complete Model...")

    # Create test input
    input_shape = (224, 224, 3)
    test_input = tf.random.normal((2, *input_shape))

    # Test EDA-ResNet50
    print("Creating EDA-ResNet50 model...")
    eda_model = EDAResNet50(num_classes=2, input_shape=input_shape)

    # Test forward pass
    output = eda_model(test_input, training=True)
    print(f"EDA-ResNet50:")
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output probabilities: {output[0]}")

    # Test factory function
    print("\nCreating model using factory function...")
    factory_model = create_eda_resnet50(num_classes=2, input_shape=input_shape)
    factory_output = factory_model(test_input)
    print(f"Factory model output shape: {factory_output.shape}")

    # Test model compilation
    print("\nCompiling model...")
    compiled_model = compile_eda_resnet50(factory_model, learning_rate=0.0001)
    print("Model compiled successfully!")

    # Test alternative version
    # print("\nTesting alternative architecture...")
    # alt_model = create_eda_resnet50(alternative=True)
    # alt_output = alt_model(test_input)
    # print(f"Alternative model output shape: {alt_output.shape}")

    print("\nEDA-ResNet50 complete model test completed successfully!")

    # Print model summary (optional - can be very long)
    # print("\nModel Summary:")
    # factory_model.summary()