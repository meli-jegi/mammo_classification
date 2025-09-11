import torch
import torch.nn as nn
import torchvision.models as models
import timm
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet50, ResNet50_Weights,
    densenet121, DenseNet121_Weights,
    squeezenet1_0, SqueezeNet1_0_Weights,
    mobilenet_v2, MobileNet_V2_Weights
)

# TIMM ViT & Hybrid Models
timm_vit_hybrid_models = [
    'vit_base_patch16_224',
    'vit_large_patch16_224',
    'deit_small_patch16_224',
    'deit_base_distilled_patch16_224',
    'swin_tiny_patch4_window7_224',
    'swin_base_patch4_window7_224',
    'maxvit_tiny_tf_224',
    'eva_giant_patch14_224',
    'beit_base_patch16_224',
    'convit_base',
    'swinconv_base_patch4_window7_224'
]

# Model Loader 
def get_model(name, num_classes=2, pretrained=True, tune_level='full'):
    name = name.lower()
    dropout_rate = 0.5

    # Torchvision CNNs 
    if name == 'resnet18':
        model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name == 'resnet50':
        model = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name == 'densenet121':
        model = densenet121(weights=DenseNet121_Weights.DEFAULT if pretrained else None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    elif name == 'squeezenet1_0':
        weights = SqueezeNet1_0_Weights.DEFAULT if pretrained else None
        model = squeezenet1_0(weights=weights)
        # SqueezeNet classifier
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1))
        model.num_classes = num_classes

    elif name == 'mobilenet_v2':
        try:
            weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
            model = mobilenet_v2(weights=weights)
        except TypeError:  # older torchvision
            model = mobilenet_v2(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)


    # TIMM CNNs 
    elif name in [
        'efficientnet_b0','efficientnet_b4',
        'convnext_tiny','convnext_base',
        'regnety_032','repvgg_b1',
        'efficientnetv2_rw_t', 'efficientnetv2_s', 'efficientnetv2_m', 'efficientnetv2_l', 'efficientnet_b7',
        'mobilevit_s', 'mobilevit_xs', 'mobilevit_xxs', 'mobilevitv2_050', 'mobilevitv2_100', 'mobilevitv2_150'
    ]:
        model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes, drop_rate=dropout_rate)

    # TIMM ViT & Hybrid Models 
    elif name in timm_vit_hybrid_models:
        model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes, drop_rate=dropout_rate)

    else:
        raise ValueError(f"Unsupported model: {name}")

    # Freezing / Tuning 
    if pretrained:
        for param in model.parameters():
            param.requires_grad = False

        if tune_level == 'full':
            for param in model.parameters():
                param.requires_grad = True
        elif tune_level == 'partial':
            # Only classifier/fc/head is trainable
            if hasattr(model, 'fc'):
                for param in model.fc.parameters():
                    param.requires_grad = True
            elif hasattr(model, 'classifier'):
                for param in model.classifier.parameters():
                    param.requires_grad = True
            elif hasattr(model, 'head'):
                for param in model.head.parameters():
                    param.requires_grad = True

            if name == 'squeezenet1_0':
                for param in model.classifier.parameters():
                    param.requires_grad = True

    return model
