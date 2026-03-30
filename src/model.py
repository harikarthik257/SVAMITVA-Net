import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

def create_model(num_classes=1, encoder_name='resnet34', encoder_weights='imagenet'):
    """
    Creates a Unet architecture with a ResNet34 encoder.
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
    )
    return model

if __name__ == "__main__":
    model = create_model()
    print("Model created successfully.")
