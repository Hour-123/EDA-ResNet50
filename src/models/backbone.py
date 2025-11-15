"""
ResNet50 Backbone for EDA-ResNet50

This module implements the ResNet50 backbone used in the EDA-ResNet50 model.
The backbone uses pre-trained ImageNet weights and removes the top classification layers.
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50


class ResNet50Backbone(layers.Layer):
    """
    ResNet50 Backbone for EDA-ResNet50

    Uses pre-trained ResNet50 weights from ImageNet and removes the top classification layers.
    Outputs feature maps that will be fed into the custom attention modules.
    """

    def __init__(self, include_top=False, pretrained_weights='imagenet', pooling=None, **kwargs):
        """
        Initialize ResNet50 Backbone

        Args:
            include_top (bool): Whether to include the fully-connected layer at the top (default: False)
            pretrained_weights (str): Pre-trained weights to use (default: 'imagenet')
            pooling (str): Optional pooling mode for feature extraction (default: None)
            **kwargs: Additional layer arguments
        """
        super(ResNet50Backbone, self).__init__(**kwargs)
        self.include_top = include_top
        self.pretrained_weights = pretrained_weights  # Changed from 'weights' to avoid conflict
        self.pooling = pooling

        # Create ResNet50 model
        self.resnet50 = ResNet50(
            include_top=include_top,
            weights=pretrained_weights,
            pooling=pooling,
            input_shape=(224, 224, 3)
        )

    def call(self, inputs, training=None):
        """
        Forward pass through ResNet50 backbone

        Args:
            inputs: Input tensor of shape (batch_size, 224, 224, 3)
            training: Whether in training mode

        Returns:
            Feature maps from ResNet50 (typically shape: batch_size, 7, 7, 2048)
        """
        return self.resnet50(inputs, training=training)

    def get_config(self):
        """Get layer configuration for serialization"""
        config = super(ResNet50Backbone, self).get_config()
        config.update({
            'include_top': self.include_top,
            'pretrained_weights': self.pretrained_weights,
            'pooling': self.pooling,
        })
        return config


class ResNet50FeatureExtractor(layers.Layer):
    """
    ResNet50 Feature Extractor with specific output layer

    Allows extracting features from specific layers of ResNet50
    for better compatibility with custom attention modules.
    """

    def __init__(self, output_layer_name='conv5_block1_out', pretrained_weights='imagenet', **kwargs):
        """
        Initialize ResNet50 Feature Extractor

        Args:
            output_layer_name (str): Name of the layer to extract features from
            pretrained_weights (str): Pre-trained weights to use (default: 'imagenet')
            **kwargs: Additional layer arguments
        """
        super(ResNet50FeatureExtractor, self).__init__(**kwargs)
        self.output_layer_name = output_layer_name
        self.pretrained_weights = pretrained_weights  # Changed from 'weights'

        # Create ResNet50 model
        self.resnet50 = ResNet50(
            include_top=False,
            weights=pretrained_weights,
            input_shape=(224, 224, 3)
        )

        # Extract specific layer output
        self.feature_extractor = tf.keras.Model(
            inputs=self.resnet50.inputs,
            outputs=self.resnet50.get_layer(output_layer_name).output,
            name=f"resnet50_{output_layer_name}"
        )

    def call(self, inputs, training=None):
        """
        Forward pass through ResNet50 feature extractor

        Args:
            inputs: Input tensor of shape (batch_size, 224, 224, 3)
            training: Whether in training mode

        Returns:
            Feature maps from specified ResNet50 layer
        """
        return self.feature_extractor(inputs, training=training)

    def get_config(self):
        """Get layer configuration for serialization"""
        config = super(ResNet50FeatureExtractor, self).get_config()
        config.update({
            'output_layer_name': self.output_layer_name,
            'pretrained_weights': self.pretrained_weights,
        })
        return config


def create_resnet50_backbone(output_layer_name='conv5_block1_out', input_shape=(224, 224, 3)):
    """
    Factory function to create ResNet50 backbone

    Args:
        output_layer_name (str): Name of the layer to extract features from
        input_shape (tuple): Input shape tuple

    Returns:
        ResNet50 backbone as Keras Model
    """
    backbone = ResNet50FeatureExtractor(output_layer_name=output_layer_name)

    inputs = layers.Input(shape=input_shape)
    outputs = backbone(inputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="resnet50_backbone")


if __name__ == "__main__":
    # Example usage and testing
    print("Testing ResNet50 Backbone...")

    # Create test input
    input_shape = (224, 224, 3)
    test_input = tf.random.normal((2, *input_shape))

    # Test ResNet50 Backbone
    backbone = ResNet50Backbone()
    output = backbone(test_input, training=False)
    print(f"ResNet50 Backbone:")
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")

    # Test ResNet50 Feature Extractor
    feature_extractor = ResNet50FeatureExtractor(output_layer_name='conv5_block1_out')
    output_features = feature_extractor(test_input, training=False)
    print(f"\nResNet50 Feature Extractor:")
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output_features.shape}")

    # Test factory function
    backbone_model = create_resnet50_backbone()
    output_model = backbone_model(test_input)
    print(f"\nFactory Function Model:")
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output_model.shape}")

    print("\nResNet50 Backbone test completed successfully!")