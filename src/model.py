import torch
import segmentation_models_pytorch as smp

def create_model(num_classes=5, encoder_name='resnet34', encoder_weights=None):
    """
    Creates a Unet architecture with a ResNet34 encoder.

    Defaults (in_channels=3, classes=5, encoder_weights=None) match the
    trained checkpoint in weights/svamitva_refined_v2_e5.pth, confirmed by
    loading it with strict=True (zero missing/unexpected keys).
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
    )
    return model

def load_trained_model(weights_path, device='cpu'):
    """
    Creates the model and loads the trained SVAMITVA-Net checkpoint onto it.
    """
    model = create_model()
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model

if __name__ == "__main__":
    model = create_model()
    print("Model created successfully.")
