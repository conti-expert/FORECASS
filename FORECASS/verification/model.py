import segmentation_models_pytorch as smp


def get_error_detector(backbone="resnet101", encoder_weights=None):
    return smp.MAnet(
        encoder_name=backbone,
        encoder_weights=encoder_weights,
        in_channels=4,
        classes=1,
        activation=None,
    )
