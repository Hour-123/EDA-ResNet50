"""
Dual Attention (DA) Module

This module implements the Dual Attention mechanism as described in the EDA-ResNet50 paper.

The DA module combines both channel attention and spatial attention mechanisms
in parallel branches and fuses their outputs through element-wise addition.
Updated algorithm includes bottleneck structures and separate spatial branches.
"""

import tensorflow as tf
from tensorflow.keras import layers


class Swish(layers.Layer):
    """Swish activation function: x * sigmoid(x)"""
    def __init__(self, **kwargs):
        super(Swish, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs * tf.keras.activations.sigmoid(inputs)

    def get_config(self):
        config = super(Swish, self).get_config()
        return config


class DualAttentionModule(layers.Layer):
    """
    Dual Attention Module

    Implementation based on the updated algorithm described in the paper:
    1. Input: Feature map X ∈ R^(H×W×C)
    2. Channel Attention: GAP → Conv1x1(C/4, Swish) → Conv1x1(C, Sigmoid) → element-wise multiplication
    3. Spatial Attention: Two separate branches (Max/Avg) → Conv1x1(C/4, Swish) → Conv1x1(C) → average → Sigmoid → multiplication
    4. Output: Element-wise addition of both branches
    """

    def __init__(self, **kwargs):
        """
        Initialize Dual Attention Module

        Args:
            **kwargs: Additional layer arguments
        """
        super(DualAttentionModule, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Build the layer with input shape

        Args:
            input_shape: Shape of input tensor (batch_size, H, W, C)
        """
        input_channels = input_shape[-1]
        reduced_channels = max(input_channels // 4, 1)  # C/4 bottleneck as per algorithm

        # ===== Channel Attention Branch =====
        # 2.b: Bottleneck structure for channel attention: C → C/4 → C

        # Layer 1: C → C/4 with Swish activation
        self.channel_gap = layers.GlobalAveragePooling2D(keepdims=True)
        self.channel_conv1 = layers.Conv2D(
            filters=reduced_channels,  # C/4
            kernel_size=(1, 1),
            padding='same',
            activation='swish',
            kernel_initializer='he_normal'
        )

        # Layer 2: C/4 → C with Sigmoid activation
        self.channel_conv2 = layers.Conv2D(
            filters=input_channels,  # Restore to C
            kernel_size=(1, 1),
            padding='same',
            activation='sigmoid',
            kernel_initializer='he_normal'
        )

        # ===== Spatial Attention Branch - Max Pooling Path =====
        # Two separate convolutional networks for max and avg pooling (no Sigmoid at end)

        # Max branch - Layer 1: 1 → C/4 with Swish
        self.spatial_max_conv1 = layers.Conv2D(
            filters=reduced_channels,  # C/4
            kernel_size=(1, 1),
            padding='same',
            activation='swish',
            kernel_initializer='he_normal'
        )

        # Max branch - Layer 2: C/4 → C (NO activation - as per algorithm)
        self.spatial_max_conv2 = layers.Conv2D(
            filters=input_channels,  # Restore to C
            kernel_size=(1, 1),
            padding='same',
            kernel_initializer='he_normal'  # No activation - will be applied later
        )

        # ===== Spatial Attention Branch - Average Pooling Path =====

        # Avg branch - Layer 1: 1 → C/4 with Swish
        self.spatial_avg_conv1 = layers.Conv2D(
            filters=reduced_channels,  # C/4
            kernel_size=(1, 1),
            padding='same',
            activation='swish',
            kernel_initializer='he_normal'
        )

        # Avg branch - Layer 2: C/4 → C (NO activation - as per algorithm)
        self.spatial_avg_conv2 = layers.Conv2D(
            filters=input_channels,  # Restore to C
            kernel_size=(1, 1),
            padding='same',
            kernel_initializer='he_normal'  # No activation - will be applied later
        )

        # Final Sigmoid for spatial attention (applied after averaging the two branches)
        self.spatial_sigmoid = layers.Activation('sigmoid')

        super(DualAttentionModule, self).build(input_shape)

    def call(self, inputs, training=None):
        """
        Forward pass following exact updated algorithm

        Args:
            inputs: Input tensor of shape (batch_size, H, W, C)
            training: Whether in training mode

        Returns:
            Output tensor with dual attention applied
        """
        X = inputs

        # ===== Step 2: Channel Attention =====

        # 2.a: Squeeze - Global Average Pooling for channel descriptor
        # z ∈ R^C
        channel_descriptor = self.channel_gap(X)  # (batch_size, 1, 1, C)

        # 2.b: Excite - Bottleneck structure with two 1x1 convolutions
        # Layer 1: C → C/4 with Swish activation
        channel_features = self.channel_conv1(channel_descriptor, training=training)
        # Layer 2: C/4 → C with Sigmoid activation to get final channel weights w
        w = self.channel_conv2(channel_features, training=training)  # (batch_size, 1, 1, C)

        # 2.c: Apply channel attention: X_CA = w · X (element-wise multiplication)
        X_CA = X * w

        # ===== Step 3: Spatial Attention =====

        # 3.a: Pool - Max pooling and Average pooling along channel axis
        # Generate two 2D feature maps: M_max and M_avg
        M_max = tf.reduce_max(X, axis=-1, keepdims=True)  # (batch_size, H, W, 1)
        M_avg = tf.reduce_mean(X, axis=-1, keepdims=True)  # (batch_size, H, W, 1)

        # 3.b: Feature extraction - Two separate branches with identical structure
        # NOTE: No Sigmoid at the end of each branch (as per algorithm)

        # Max branch output: M'_{max} = Conv_{1x1, C}(Swish(Conv_{1x1, C/4}(M_{max})))
        M_max_features = self.spatial_max_conv1(M_max, training=training)  # C/4, Swish
        M_prime_max = self.spatial_max_conv2(M_max_features, training=training)  # C, no activation

        # Avg branch output: M'_{avg} = Conv_{1x1, C}(Swish(Conv_{1x1, C/4}(M_{avg})))
        M_avg_features = self.spatial_avg_conv1(M_avg, training=training)  # C/4, Swish
        M_prime_avg = self.spatial_avg_conv2(M_avg_features, training=training)  # C, no activation

        # 3.c: Generate spatial attention map using formula (14)
        # A_{spatial} = σ((M'_{max} + M'_{avg}) / 2)
        combined_spatial = (M_prime_max + M_prime_avg) / 2.0  # Average the two branch outputs
        A_spatial = self.spatial_sigmoid(combined_spatial)  # Apply final sigmoid to get spatial weights

        # 3.d: Apply spatial attention: X_SA = A_{spatial} · X (element-wise multiplication)
        X_SA = X * A_spatial

        # ===== Step 4: Output =====

        # Element-wise addition of both attention final outputs
        # X_{out} = X_{CA} + X_{SA}
        X_out = X_CA + X_SA

        return X_out

    def get_config(self):
        """Get layer configuration for serialization"""
        config = super(DualAttentionModule, self).get_config()
        return config


def create_dual_attention_module(input_shape=None):
    """
    Factory function to create Dual Attention Module

    Args:
        input_shape: Input shape tuple (optional)

    Returns:
        Dual Attention Module as Keras Model (if input_shape provided) or Layer
    """
    if input_shape is not None:
        inputs = layers.Input(shape=input_shape)
        outputs = DualAttentionModule()(inputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs)
    else:
        return DualAttentionModule()


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Dual Attention Module (Updated Algorithm)...")

    # Create test input
    input_shape = (224, 224, 64)  # C=64 as mentioned in algorithm
    test_input = tf.random.normal((2, *input_shape))

    # Test Dual Attention Module
    da = DualAttentionModule()
    output = da(test_input, training=True)
    print(f"Dual Attention Module:")
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")

    # Verify output shape matches input shape
    assert test_input.shape == output.shape, "Output shape should match input shape"

    # Test with different input sizes
    test_input2 = tf.random.normal((1, 112, 112, 32))
    da2 = DualAttentionModule()
    output2 = da2(test_input2, training=False)
    print(f"\nDifferent size test:")
    print(f"Input shape: {test_input2.shape}")
    print(f"Output shape: {output2.shape}")

    print("\nDual Attention Module test completed successfully!")