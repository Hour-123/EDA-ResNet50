"""
Multi-Scale Feature Representation (MFR) Module

This module implements the Multi-Scale Feature Representation (MFR) module
as described in the EDA-ResNet50 paper for skin cancer classification.

The MFR module captures multi-scale features through parallel convolutional
branches with different receptive fields and feature fusion strategies.
"""

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K


class Swish(layers.Layer):
    """Swish activation function: x * sigmoid(x)"""
    def __init__(self, **kwargs):
        super(Swish, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs * tf.keras.activations.sigmoid(inputs)

    def get_config(self):
        config = super(Swish, self).get_config()
        return config


class MFRModule(layers.Layer):
    """
    Multi-Scale Feature Representation (MFR) Module

    Implementation based on the algorithm described in the paper:
    1. Input: Feature map X with shape (H, W, C)
    2. Three parallel branches with different conv operations
    3. Multi-level concatenation and fusion
    4. Output: Multi-scale fused features
    """

    def __init__(self, filters=64, **kwargs):
        """
        Initialize MFR Module

        Args:
            filters (int): Number of filters for convolution operations (default: 64)
            **kwargs: Additional layer arguments
        """
        super(MFRModule, self).__init__(**kwargs)
        self.filters = filters
        self.swish = Swish()

        # Branch 1: Single 1x1 convolution
        self.branch1_conv = layers.Conv2D(
            filters, (1, 1),
            padding='same',
            kernel_initializer='he_normal'
        )
        self.branch1_bn = layers.BatchNormalization()

        # Branch 2: Two 3x3 convolutions (each with filters/2)
        branch2_filters = filters // 2
        self.branch2_conv1 = layers.Conv2D(
            branch2_filters, (3, 3),
            padding='same',
            kernel_initializer='he_normal'
        )
        self.branch2_bn1 = layers.BatchNormalization()

        self.branch2_conv2 = layers.Conv2D(
            branch2_filters, (3, 3),
            padding='same',
            kernel_initializer='he_normal'
        )
        self.branch2_bn2 = layers.BatchNormalization()

        # Branch 3: Single 3x3 convolution
        self.branch3_conv = layers.Conv2D(
            filters, (3, 3),
            padding='same',
            kernel_initializer='he_normal'
        )
        self.branch3_bn = layers.BatchNormalization()

        # Fusion: 3x3 convolution on concatenated branch2 and branch3
        self.fusion_conv = layers.Conv2D(
            filters, (3, 3),
            padding='same',
            kernel_initializer='he_normal'
        )
        self.fusion_bn = layers.BatchNormalization()

        # Output: 1x1 convolution
        self.output_conv = layers.Conv2D(
            filters, (1, 1),
            padding='same',
            kernel_initializer='he_normal'
        )
        self.output_bn = layers.BatchNormalization()

    def call(self, inputs, training=None):
        """
        Forward pass of MFR Module

        Args:
            inputs: Input tensor of shape (batch_size, H, W, C)
            training: Whether in training mode (for batch normalization)

        Returns:
            Output tensor with multi-scale fused features
        """
        # Branch 1: 1x1 convolution
        branch1 = self.branch1_conv(inputs)
        branch1 = self.branch1_bn(branch1, training=training)
        branch1 = self.swish(branch1)

        # Branch 2: Two sequential 3x3 convolutions
        branch2 = self.branch2_conv1(inputs)
        branch2 = self.branch2_bn1(branch2, training=training)
        branch2 = self.swish(branch2)

        branch2 = self.branch2_conv2(branch2)
        branch2 = self.branch2_bn2(branch2, training=training)
        branch2 = self.swish(branch2)

        # Branch 3: Single 3x3 convolution
        branch3 = self.branch3_conv(inputs)
        branch3 = self.branch3_bn(branch3, training=training)
        branch3 = self.swish(branch3)

        # Concatenation 1: Branch 2 + Branch 3
        concat1 = layers.Concatenate(axis=-1)([branch2, branch3])

        # Fusion: 3x3 convolution on concatenated features
        fusion = self.fusion_conv(concat1)
        fusion = self.fusion_bn(fusion, training=training)
        fusion = self.swish(fusion)

        # Concatenation 2: Fusion + Branch 2 + Branch 3
        concat2 = layers.Concatenate(axis=-1)([fusion, branch2, branch3])

        # Final concatenation: Previous result + Branch 1
        final_concat = layers.Concatenate(axis=-1)([concat2, branch1])

        # Output: 1x1 convolution
        output = self.output_conv(final_concat)
        output = self.output_bn(output, training=training)
        output = self.swish(output)

        return output

    def get_config(self):
        """Get layer configuration for serialization"""
        config = super(MFRModule, self).get_config()
        config.update({
            'filters': self.filters,
        })
        return config

    def build(self, input_shape):
        """Build the layer with input shape"""
        super(MFRModule, self).build(input_shape)


def create_mfr_module(filters=64, input_shape=None):
    """
    Factory function to create MFR Module

    Args:
        filters (int): Number of filters (default: 64)
        input_shape: Input shape tuple (optional)

    Returns:
        MFR Module as Keras Model (if input_shape provided) or Layer
    """
    if input_shape is not None:
        inputs = layers.Input(shape=input_shape)
        outputs = MFRModule(filters)(inputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs)
    else:
        return MFRModule(filters)


if __name__ == "__main__":
    # Example usage and testing
    print("Testing MFR Module...")

    # Create test input
    input_shape = (224, 224, 512)  # Typical ResNet50 output shape
    test_input = tf.random.normal((2, *input_shape))

    # Create MFR module
    mfr = MFRModule(filters=64)

    # Test forward pass
    output = mfr(test_input, training=True)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")

    # Test with different input sizes
    test_input2 = tf.random.normal((1, 112, 112, 256))
    mfr2 = MFRModule(filters=32)
    output2 = mfr2(test_input2, training=False)
    print(f"Input shape: {test_input2.shape}")
    print(f"Output shape: {output2.shape}")

    print("MFR Module test completed successfully!")