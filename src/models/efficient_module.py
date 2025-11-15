"""
Efficient Channel Attention Module

This module implements the Efficient Channel Attention mechanism
as described in the EDA-ResNet50 paper.

The module uses global average pooling followed by 1D convolution
to capture channel-wise attention in a lightweight manner.
"""

import tensorflow as tf
from tensorflow.keras import layers


class EfficientModule(layers.Layer):
    """
    Efficient Channel Attention Module

    Implementation based on the algorithm described in the paper:
    1. Squeeze: Global Average Pooling to compress spatial information
    2. Channel Dependencies: 1D convolution to capture local channel dependencies
    3. Excitation: Sigmoid activation to generate channel weights
    4. Rescale: Element-wise multiplication to reweight original features
    """

    def __init__(self, reduction_ratio=16, kernel_size=3, **kwargs):
        """
        Initialize Efficient Channel Attention Module

        Args:
            reduction_ratio (int): Reduction ratio for channel compression (default: 16)
            kernel_size (int): Kernel size for 1D convolution (default: 3)
            **kwargs: Additional layer arguments
        """
        super(EfficientModule, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size

        # Global Average Pooling
        self.gap = layers.GlobalAveragePooling2D(keepdims=True)

        # 1D Convolution for channel attention
        self.conv1d = None  # Will be initialized in build()

        # Sigmoid activation
        self.sigmoid = layers.Activation('sigmoid')

    def build(self, input_shape):
        """
        Build the layer with input shape

        Args:
            input_shape: Shape of input tensor (batch_size, H, W, C)
        """
        input_channels = input_shape[-1]
        reduced_channels = max(input_channels // self.reduction_ratio, 1)

        # Create 1D convolution layer
        self.conv1d = layers.Conv1D(
            filters=reduced_channels,
            kernel_size=self.kernel_size,
            padding='same',
            activation='relu',
            kernel_initializer='he_normal'
        )

        # Create final 1D convolution to restore channels
        self.conv1d_restore = layers.Conv1D(
            filters=input_channels,
            kernel_size=1,
            padding='same',
            activation='relu',
            kernel_initializer='he_normal'
        )

        super(EfficientModule, self).build(input_shape)

    def call(self, inputs, training=None):
        """
        Forward pass of Efficient Module

        Args:
            inputs: Input tensor of shape (batch_size, H, W, C)
            training: Whether in training mode

        Returns:
            Output tensor with channel attention applied
        """
        # Step 1: Squeeze - Global Average Pooling
        # Input: (batch_size, H, W, C)
        # Output: (batch_size, 1, 1, C)
        squeezed = self.gap(inputs)

        # Reshape for 1D convolution: (batch_size, C, 1)
        # We need to transpose dimensions to make channels the sequence dimension
        squeezed_reshaped = tf.transpose(squeezed, [0, 3, 1, 2])  # (batch_size, C, 1, 1)
        squeezed_reshaped = tf.squeeze(squeezed_reshaped, axis=-1)  # (batch_size, C, 1)

        # Step 2 & 3: Channel Dependencies & Excitation
        # Apply 1D convolution to capture local channel dependencies
        attention = self.conv1d(squeezed_reshaped, training=training)

        # Restore channel dimension
        attention = self.conv1d_restore(attention, training=training)

        # Apply sigmoid to generate attention weights
        attention = self.sigmoid(attention)

        # Reshape back to (batch_size, 1, 1, C) for broadcasting
        # After conv operations: attention.shape = (batch_size, C, 1)
        attention = tf.squeeze(attention, axis=-1)       # (batch_size, C)
        attention = tf.reshape(attention, [-1, 1, 1, tf.shape(attention)[-1]])  # (batch_size, 1, 1, C)
        # Now ready for broadcasting with inputs (batch_size, H, W, C)

        # Step 4: Rescale - Element-wise multiplication
        # Broadcast attention weights to match input spatial dimensions
        output = inputs * attention

        return output

    def get_config(self):
        """Get layer configuration for serialization"""
        config = super(EfficientModule, self).get_config()
        config.update({
            'reduction_ratio': self.reduction_ratio,
            'kernel_size': self.kernel_size,
        })
        return config


class EfficientModuleSimplified(layers.Layer):
    """
    Simplified version of Efficient Module following the exact algorithm from README

    This version directly follows the 6-step algorithm
    """

    def __init__(self, kernel_size=3, **kwargs):
        """
        Initialize Simplified Efficient Module

        Args:
            kernel_size (int): Kernel size for 1D convolution (default: 3)
            **kwargs: Additional layer arguments
        """
        super(EfficientModuleSimplified, self).__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        """
        Build the layer with input shape

        Args:
            input_shape: Shape of input tensor (batch_size, H, W, C)
        """
        input_channels = input_shape[-1]

        # Global Average Pooling
        self.gap = layers.GlobalAveragePooling2D()

        # 1D Convolution for capturing local channel dependencies
        # Use 1 filter to produce single attention value per position
        self.conv1d = layers.Conv1D(
            filters=1,  # Single filter to produce single output value
            kernel_size=self.kernel_size,
            padding='same',
            activation='relu',
            kernel_initializer='he_normal'
        )

        # Sigmoid activation
        self.sigmoid = layers.Activation('sigmoid')

        super(EfficientModuleSimplified, self).build(input_shape)

    def call(self, inputs, training=None):
        """
        Forward pass following the exact 6-step algorithm

        Args:
            inputs: Input tensor of shape (batch_size, H, W, C)
            training: Whether in training mode

        Returns:
            Output tensor with channel attention applied
        """
        
        X = inputs

        
        # Global Average Pooling compresses spatial information
        z = self.gap(X)  # Shape: (batch_size, C)

        # Reshape for 1D convolution: (batch_size, C, 1)
        z_expanded = tf.expand_dims(z, axis=-1)  # Shape: (batch_size, C, 1)

        # Step 3: Channel Dependencies - Conv1D(z; k)
        # 1D convolution captures local dependencies across channels
        # Input: (batch_size, C, 1) → Output: (batch_size, C, 1)
        z_conv = self.conv1d(z_expanded, training=training)

        # Remove the last dimension (feature dimension)
        # Shape: (batch_size, C, 1) → (batch_size, C)
        z_conv = tf.squeeze(z_conv, axis=-1)  # Shape: (batch_size, C)

        
        # Generate channel attention weights
        w = self.sigmoid(z_conv)  # Shape: (batch_size, C)

        # Step 5: Rescale - X � w
        # Reshape weights for broadcasting: (batch_size, 1, 1, C)
        w_reshaped = tf.reshape(w, [-1, 1, 1, tf.shape(w)[-1]])

        # Element-wise multiplication with broadcasting
        X_att = X * w_reshaped

        # Step 6: Output
        return X_att

    def get_config(self):
        """Get layer configuration for serialization"""
        config = super(EfficientModuleSimplified, self).get_config()
        config.update({
            'kernel_size': self.kernel_size,
        })
        return config


def create_efficient_module(kernel_size=3, simplified=True, input_shape=None):
    """
    Factory function to create Efficient Module

    Args:
        kernel_size (int): Kernel size for 1D convolution (default: 3)
        simplified (bool): Whether to use simplified version (default: True)
        input_shape: Input shape tuple (optional)

    Returns:
        Efficient Module as Keras Model (if input_shape provided) or Layer
    """
    if simplified:
        module_class = EfficientModuleSimplified
    else:
        module_class = EfficientModule

    if input_shape is not None:
        inputs = layers.Input(shape=input_shape)
        outputs = module_class(kernel_size=kernel_size)(inputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs)
    else:
        return module_class(kernel_size=kernel_size)


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Efficient Module...")

    # Create test input
    input_shape = (224, 224, 64)
    test_input = tf.random.normal((2, *input_shape))

    # Test simplified version
    efficient_simple = EfficientModuleSimplified(kernel_size=3)
    output_simple = efficient_simple(test_input, training=True)
    print(f"Simplified version:")
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output_simple.shape}")

    # Test full version
    # efficient_full = EfficientModule(kernel_size=3)
    # output_full = efficient_full(test_input, training=True)
    # print(f"\nFull version:")
    # print(f"Input shape: {test_input.shape}")
    # print(f"Output shape: {output_full.shape}")

    print("\nEfficient Module test completed successfully!")