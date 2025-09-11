import matplotlib
matplotlib.use('Agg')  
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (confusion_matrix, classification_report, roc_auc_score, 
                           roc_curve, precision_recall_curve, average_precision_score,
                           precision_score, recall_score, f1_score, accuracy_score)
import numpy as np
import cv2
from collections import Counter 
from model_final_new import get_model
from dataloader_final import get_dataloaders
import torch.nn.functional as F
from torchvision import transforms
from sklearn.metrics import auc
from sklearn.metrics import ConfusionMatrixDisplay
import json
import random 
import time
import torch
from sklearn.manifold import TSNE
import logging
import sys
import pandas as pd

# GradCAM 
class GradCAM:
    def __init__(self, model, target_layer=None, verbose=False):
        self.model = model
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        # Auto-detect last Conv2d if not provided
        if target_layer is None:
            name, target_layer = find_last_conv_layer(model, verbose=verbose)
            print(f"[Grad-CAM] Using target layer: {name}")
        self.target_layer = target_layer
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.hook_handles.append(
            self.target_layer.register_forward_hook(forward_hook)
        )
        self.hook_handles.append(
            self.target_layer.register_full_backward_hook(backward_hook)
        )

    def generate(self, input_tensor, class_idx=None):
        # Ensure input requires grad
        input_tensor = input_tensor.to(next(self.model.parameters()).device)
        input_tensor.requires_grad = True        
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # target class score
        loss = output[0, class_idx]
        # backward to get gradients
        loss.backward()
        gradients = self.gradients[0]      
        activations = self.activations[0]  #[C, H, W]
        # global-average-pool gradients
        weights = gradients.mean(dim=(1, 2))
        # weighted sum of activations
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32).to(input_tensor.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # normalize heatmap
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = cam.detach().cpu().numpy()

        return cam

    def clear_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

def generate_gradcam_for_test(model, dataloader, device, save_dir, target_layer=None):
    model.eval()
    os.makedirs(os.path.join(save_dir, 'correct'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'incorrect'), exist_ok=True)
    # Initialize Grad-CAM
    gradcam = GradCAM(model, target_layer=target_layer, verbose=True)
    correct_count = 0
    incorrect_count = 0
    for i, (inputs, labels, filenames) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Get predictions (without grad for efficiency)
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        # Process each image in the batch
        for j in range(inputs.size(0)):
            input_tensor = inputs[j].unsqueeze(0)  # Add batch dimension
            pred = preds[j].item()
            true = labels[j].item()
            # Create filename
            if isinstance(filenames[j], str):
                base_filename = os.path.splitext(os.path.basename(filenames[j]))[0]
            else:
                base_filename = f"image_{i}_{j}"
            
            filename = f"{base_filename}_pred{pred}_true{true}.png"
            # Generate CAM (requires gradients)
            cam = gradcam.generate(input_tensor, class_idx=pred)
            # Determine save folder
            if pred == true:
                folder = 'correct'
                correct_count += 1
            else:
                folder = 'incorrect'
                incorrect_count += 1
                
            save_path = os.path.join(save_dir, folder, filename)
            # Save Grad-CAM overlay
            save_gradcam_on_image(
                input_tensor.detach().cpu(),
                cam,
                save_path,
                original_path=filenames[j]  # full-size mammogram path
            )
        print(f"Processed batch {i+1}/{len(dataloader)}")
    gradcam.clear_hooks()
    print(f"Grad-CAM heatmaps saved:")
    print(f"  - Correct predictions: {correct_count} images in {save_dir}/correct")
    print(f"  - Incorrect predictions: {incorrect_count} images in {save_dir}/incorrect")

def save_gradcam_on_image(image_tensor, cam, save_path, original_path=None):
    if original_path is not None and os.path.exists(original_path):
        # Load original full-size mammogram
        orig_img = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2RGB)

        # Resize CAM to original image size
        cam_resized = cv2.resize(cam, (orig_img.shape[1], orig_img.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # overlay
        superimposed_img = heatmap * 0.4 + orig_img * 0.6
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

        cv2.imwrite(save_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
    
    else:
        # tensor-sized overlay
        image = image_tensor.squeeze(0).detach().cpu().numpy()
        if image.shape[0] == 1:
            image = np.repeat(image, 3, axis=0)
        image = np.transpose(image, (1, 2, 0))
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        image = np.uint8(255 * image)

        cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        superimposed_img = heatmap * 0.4 + image * 0.6
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
        cv2.imwrite(save_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))

def find_last_conv_layer(model, verbose=False):
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            conv_layers.append((name, module))

    if not conv_layers:
        raise ValueError("No Conv2d layers found in this model.")

    if verbose:
        print("\n[Grad-CAM] Available Conv2d layers:")
        for i, (name, _) in enumerate(conv_layers):
            print(f"  {i+1}. {name}")
        print(f"[Grad-CAM] Auto-selecting last conv layer: {conv_layers[-1][0]}\n")
    return conv_layers[-1]  # (name, module)

def get_last_conv_layer(model, model_name):
    model_name = model_name.lower()
    
    if model_name in ['resnet18', 'resnet50']:
        # ResNet: use the last conv layer in layer4
        return model.layer4[-1].conv2
    
    elif model_name in ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b4', 'efficientnet_b7']:
        # TIMM EfficientNet
        if hasattr(model, 'conv_head') and model.conv_head is not None:
            return model.conv_head
        elif hasattr(model, 'blocks') and len(model.blocks) > 0:
            # Search in the last few blocks for conv layers
            for block_idx in reversed(range(len(model.blocks))):
                block = model.blocks[block_idx]
                conv_layers = []
                for name, module in block.named_modules():
                    if isinstance(module, torch.nn.Conv2d):
                        conv_layers.append((name, module))
                if conv_layers:
                    print(f"[EfficientNet] Using conv layer from block {block_idx}: {conv_layers[-1][0]}")
                    return conv_layers[-1][1]
        
        # search entire model
        return _auto_detect_last_conv(model, model_name)
    
    elif model_name == 'densenet121':
        # DenseNet: search in features module
        conv_layers = []
        for name, module in model.features.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                conv_layers.append((name, module))
        if conv_layers:
            print(f"[DenseNet] Using: {conv_layers[-1][0]}")
            return conv_layers[-1][1]
        else:
            raise ValueError("No Conv2d layers found in DenseNet features")
    
    elif model_name in ['alexnet', 'vgg16']:
        # AlexNet/VGG: search features in reverse order
        conv_layers = []
        for i, layer in enumerate(model.features):
            if isinstance(layer, nn.Conv2d):
                conv_layers.append((f"features.{i}", layer))
        if conv_layers:
            print(f"[{model_name.upper()}] Using: {conv_layers[-1][0]}")
            return conv_layers[-1][1]
        else:
            raise ValueError(f"No Conv2d layers found in {model_name} features")
    
    elif model_name == 'mobilenet_v2':
        # MobileNetV2: search in features
        conv_layers = []
        for name, module in model.features.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                conv_layers.append((name, module))
        if conv_layers:
            print(f"[MobileNetV2] Using: {conv_layers[-1][0]}")
            return conv_layers[-1][1]
        else:
            raise ValueError("No Conv2d layers found in MobileNet features")
    
    elif model_name == 'squeezenet1_0':
        # ModifiedSqueezeNet: search in features + handle classifier conv
        conv_layers = []
        # Check features first
        for name, module in model.features.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                conv_layers.append((f"features.{name}", module))
        
        # Check classifier for conv layers 
        for name, module in model.classifier.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                conv_layers.append((f"classifier.{name}", module))
        
        if conv_layers:
            print(f"[SqueezeNet] Using: {conv_layers[-1][0]}")
            return conv_layers[-1][1]
        else:
            raise ValueError("No Conv2d layers found in SqueezeNet")
    
    else:
        # Unknown model: auto-detect
        return _auto_detect_last_conv(model, model_name)

def _auto_detect_last_conv(model, model_name):
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            conv_layers.append((name, module))
    
    if conv_layers:
        print(f"[{model_name}] Auto-detected {len(conv_layers)} conv layers, using: {conv_layers[-1][0]}")
        return conv_layers[-1][1]
    else:
        raise ValueError(f"No Conv2d layers found in model: {model_name}")

def extract_features(model, dataloader, device, num_samples=2000):
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for i, (images, targets, _) in enumerate(dataloader):
            images = images.to(device)
            
            # Forward pass up to the penultimate layer
            feats = model.forward_features(images) if hasattr(model, "forward_features") else model(images)
            
            # Flatten conv features if necessary
            if feats.ndim > 2:  
                feats = torch.flatten(feats, 1)

            features.append(feats.cpu().numpy())
            labels.append(targets.numpy())

            if len(np.concatenate(features)) >= num_samples:
                break

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels

def save_roc_pr_curves(all_labels, all_probs, class_names, model_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    try:
        if len(class_names) == 2:  # Binary classification
            # For binary classification, use probabilities of positive class
            if all_probs.shape[1] == 2:
                y_scores = all_probs[:, 1]  # Probability of positive class
            else:
                y_scores = all_probs[:, 0]  # If only one column
            
            # Calculate ROC AUC
            roc_auc = roc_auc_score(all_labels, y_scores)
            print(f"ROC-AUC Score: {roc_auc:.4f}")

            # Save AUC score to txt
            with open(os.path.join(save_dir, f"{model_name}_roc_auc.txt"), "w") as f:
                f.write(f"ROC-AUC Score: {roc_auc:.4f}\n")

            # ROC Curve
            fpr, tpr, roc_thresholds = roc_curve(all_labels, y_scores)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color="darkorange", lw=2, 
                     label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve - {model_name}")
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{model_name}_roc_curve.png"), dpi=300, bbox_inches="tight")
            plt.close()

            # Precision-Recall Curve
            precision, recall, pr_thresholds = precision_recall_curve(all_labels, y_scores)
            avg_precision = average_precision_score(all_labels, y_scores)
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color="blue", lw=2, 
                     label=f"PR curve (AP = {avg_precision:.3f})")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"Precision-Recall Curve - {model_name}")
            plt.legend(loc="lower left")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{model_name}_pr_curve.png"), dpi=300, bbox_inches="tight")
            plt.close()
            
            return roc_auc, avg_precision

        else:  # Multi-class
            # One-hot encode labels for multi-class ROC
            from sklearn.preprocessing import label_binarize
            y_bin = label_binarize(all_labels, classes=range(len(class_names)))
            
            # Calculate macro-average ROC AUC
            roc_auc = roc_auc_score(y_bin, all_probs, average="macro", multi_class="ovr")
            print(f"ROC-AUC Score (macro): {roc_auc:.4f}")

            # Save AUC score to txt
            with open(os.path.join(save_dir, f"{model_name}_roc_auc.txt"), "w") as f:
                f.write(f"ROC-AUC Score (macro): {roc_auc:.4f}\n")

            # Plot ROC curves for each class
            plt.figure(figsize=(10, 8))
            for i in range(len(class_names)):
                fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
                roc_auc_class = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, 
                         label=f'{class_names[i]} (AUC = {roc_auc_class:.3f})')
            
            plt.plot([0, 1], [0, 1], 'k--', lw=2, label="Random")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'Multi-class ROC Curves - {model_name}')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{model_name}_roc_curve.png"), dpi=300, bbox_inches="tight")
            plt.close()
            
            # Calculate and save average precision for each class
            avg_precisions = []
            plt.figure(figsize=(10, 8))
            for i in range(len(class_names)):
                precision, recall, _ = precision_recall_curve(y_bin[:, i], all_probs[:, i])
                avg_precision = average_precision_score(y_bin[:, i], all_probs[:, i])
                avg_precisions.append(avg_precision)
                plt.plot(recall, precision, lw=2,
                         label=f'{class_names[i]} (AP = {avg_precision:.3f})')
            
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Multi-class Precision-Recall Curves - {model_name}')
            plt.legend(loc="lower left")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{model_name}_pr_curve.png"), dpi=300, bbox_inches="tight")
            plt.close()
            
            return roc_auc, np.mean(avg_precisions)

    except Exception as e:
        print(f"Error generating ROC/PR curves: {e}")
        return None, None

def evaluate_model(model, dataloader, device, class_names, model_name, save_dir):
    if len(class_names) == 2:
        # Binary classification 
        return evaluate_medical_binary_model(model, dataloader, device, 
                                           class_names, model_name, save_dir)
    else:
        # Multi-class classification - macro averaging
        return evaluate_multiclass_model(model, dataloader, device, 
                                        class_names, model_name, save_dir)

def evaluate_medical_binary_model(model, dataloader, device, class_names, model_name, save_dir):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels, _ in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Determine which class is malignant
    if 'malignant' in [name.lower() for name in class_names]:
        malignant_idx = [name.lower() for name in class_names].index('malignant')
    elif 'cancer' in [name.lower() for name in class_names]:
        malignant_idx = [name.lower() for name in class_names].index('cancer')
    else:
        # Assume malignant is class 1 (common convention)
        malignant_idx = 1
    
    benign_idx = 1 - malignant_idx
    # Overall accuracy
    accuracy = accuracy_score(all_labels, all_preds)

    # Use macro averaging for balanced reporting
    # gives equal weight to both classes regardless of class distribution
    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)

    # Sensitivity = Recall for malignant class (ability to detect cancer)
    sensitivity = recall_per_class[malignant_idx]
    
    # Specificity = Recall for benign class (ability to correctly identify benign)
    specificity = recall_per_class[benign_idx]
    
    # Positive Predictive Value = Precision for malignant class
    ppv = precision_per_class[malignant_idx]
    
    # Negative Predictive Value = Precision for benign class  
    npv = precision_per_class[benign_idx]

    # Confusion matrix 
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    # False Positive Rate (1 - Specificity)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # False Negative Rate (1 - Sensitivity) 
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    # Print results in medical context
    print(f"\n{model_name} Medical Binary Classification Results:")
    print("=" * 60)
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"\nMacro-averaged Metrics (Equal weight to both classes):")
    print(f"  Precision: {precision_macro:.4f}")
    print(f"  Recall: {recall_macro:.4f}")
    print(f"  F1-Score: {f1_macro:.4f}")
    
    print(f"\nMedical Metrics:")
    print(f"  Sensitivity (Malignant Recall): {sensitivity:.4f}")
    print(f"  Specificity (Benign Recall): {specificity:.4f}")
    print(f"  PPV (Malignant Precision): {ppv:.4f}")
    print(f"  NPV (Benign Precision): {npv:.4f}")
    print(f"  False Positive Rate: {fpr:.4f}")
    print(f"  False Negative Rate: {fnr:.4f}")

    print(f"\nPer-Class Detailed Metrics:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}:")
        print(f"    Precision: {precision_per_class[i]:.4f}")
        print(f"    Recall: {recall_per_class[i]:.4f}")
        print(f"    F1-Score: {f1_per_class[i]:.4f}")

    print(f"\nConfusion Matrix Components:")
    print(f"  True Negatives (Benign→Benign): {tn}")
    print(f"  False Positives (Benign→Malignant): {fp}")
    print(f"  False Negatives (Malignant→Benign): {fn}")
    print(f"  True Positives (Malignant→Malignant): {tp}")

    # Classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, 
                                 digits=4, zero_division=0)
    print(f"\nDetailed Classification Report:")
    print(report)

    # Save comprehensive results for thesis
    os.makedirs(save_dir, exist_ok=True)
    
    # Save detailed medical report
    with open(os.path.join(save_dir, f"{model_name}_medical_evaluation.txt"), "w") as f:
        f.write(f"{model_name} Medical Binary Classification Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Dataset: Mammography Binary Classification (Benign vs Malignant)\n")
        f.write(f"Total samples: {len(all_labels)}\n")
        f.write(f"Class distribution: {class_names[0]}: {np.sum(all_labels == 0)}, {class_names[1]}: {np.sum(all_labels == 1)}\n\n")
        
        f.write(f"OVERALL PERFORMANCE:\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        
        f.write(f"MACRO-AVERAGED METRICS (Recommended for medical applications):\n")
        f.write(f"Precision: {precision_macro:.4f}\n")
        f.write(f"Recall: {recall_macro:.4f}\n")
        f.write(f"F1-Score: {f1_macro:.4f}\n\n")
        
        f.write(f"CLINICAL PERFORMANCE METRICS:\n")
        f.write(f"Sensitivity (True Positive Rate): {sensitivity:.4f}\n")
        f.write(f"  - Ability to correctly identify malignant cases\n")
        f.write(f"  - Higher values mean fewer missed cancers\n")
        f.write(f"Specificity (True Negative Rate): {specificity:.4f}\n")
        f.write(f"  - Ability to correctly identify benign cases\n")
        f.write(f"  - Higher values mean fewer false alarms\n")
        f.write(f"Positive Predictive Value: {ppv:.4f}\n")
        f.write(f"  - When model predicts malignant, probability it's correct\n")
        f.write(f"Negative Predictive Value: {npv:.4f}\n")
        f.write(f"  - When model predicts benign, probability it's correct\n")
        f.write(f"False Positive Rate: {fpr:.4f}\n")
        f.write(f"False Negative Rate: {fnr:.4f}\n\n")
        
        f.write(f"PER-CLASS PERFORMANCE:\n")
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name}:\n")
            f.write(f"  Precision: {precision_per_class[i]:.4f}\n")
            f.write(f"  Recall: {recall_per_class[i]:.4f}\n")
            f.write(f"  F1-Score: {f1_per_class[i]:.4f}\n")
            f.write(f"  Support: {np.sum(all_labels == i)} samples\n\n")
        
        f.write(f"CONFUSION MATRIX:\n")
        f.write(f"                 Predicted\n")
        f.write(f"               Ben   Mal\n")
        f.write(f"Actual Benign  {tn:3d}  {fp:3d}\n")
        f.write(f"       Malign  {fn:3d}  {tp:3d}\n\n")
        
        f.write("DETAILED CLASSIFICATION REPORT:\n")
        f.write(report)

    # Create medical-focused confusion matrix visualization
    plt.figure(figsize=(8, 6))
    labels_cm = ['Benign', 'Malignant']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels_cm, yticklabels=labels_cm,
                cbar_kws={'label': 'Count'})
    
    # Add medical interpretation
    plt.title(f'Confusion Matrix - {model_name}\n' + 
             f'Sensitivity: {sensitivity:.3f}, Specificity: {specificity:.3f}')
    plt.ylabel('Actual Diagnosis')
    plt.xlabel('Predicted Diagnosis')
    
    # Add text annotations for medical context
    plt.text(0.5, -0.1, f'False Positive Rate: {fpr:.3f}', 
             transform=plt.gca().transAxes, ha='center')
    plt.text(0.5, -0.15, f'False Negative Rate: {fnr:.3f}', 
             transform=plt.gca().transAxes, ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_medical_confusion_matrix.png"), 
                dpi=300, bbox_inches="tight")
    plt.close()

    # ROC and PR curves
    roc_auc, avg_precision = save_roc_pr_curves(all_labels, all_probs, class_names, model_name, save_dir)

    # Return results with medical focus
    results = {
        'model_name': model_name,
        'accuracy': float(accuracy),
        'precision': float(precision_macro),  # Macro averaging
        'recall': float(recall_macro),        # Macro averaging  
        'f1_score': float(f1_macro),         # Macro averaging
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'ppv': float(ppv),
        'npv': float(npv),
        'false_positive_rate': float(fpr),
        'false_negative_rate': float(fnr),
        'benign_precision': float(precision_per_class[benign_idx]),
        'benign_recall': float(recall_per_class[benign_idx]),
        'malignant_precision': float(precision_per_class[malignant_idx]),
        'malignant_recall': float(recall_per_class[malignant_idx]),
        'roc_auc': float(roc_auc) if roc_auc is not None else None,
        'avg_precision': float(avg_precision) if avg_precision is not None else None,
        'confusion_matrix': cm.tolist(),
        'class_names': class_names,
        'malignant_class_index': malignant_idx
    }

    # Save as JSON for easy data analysis
    with open(os.path.join(save_dir, f"{model_name}_medical_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results

def evaluate_multiclass_model(model, dataloader, device, class_names, model_name, save_dir):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels, _ in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate metrics with macro averaging
    accuracy = accuracy_score(all_labels, all_preds)
    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print(f"\n{model_name} Multi-Class Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (macro): {precision_macro:.4f}")
    print(f"Recall (macro): {recall_macro:.4f}")
    print(f"F1-Score (macro): {f1_macro:.4f}")

    # Classification Report
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    print("\nClassification Report:")
    print(report)

    # Save classification report
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"{model_name}_classification_report.txt"), "w") as f:
        f.write(f"{model_name} Multi-Class Results:\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision (macro): {precision_macro:.4f}\n")
        f.write(f"Recall (macro): {recall_macro:.4f}\n")
        f.write(f"F1-Score (macro): {f1_macro:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_confusion_matrix.png"), 
                dpi=300, bbox_inches="tight")
    plt.close()

    # ROC and PR curves
    roc_auc, avg_precision = save_roc_pr_curves(all_labels, all_probs, class_names, model_name, save_dir)

    # Save comprehensive results
    results = {
        'model_name': model_name,
        'accuracy': float(accuracy),
        'precision': float(precision_macro),
        'recall': float(recall_macro),
        'f1_score': float(f1_macro),
        'roc_auc': float(roc_auc) if roc_auc is not None else None,
        'avg_precision': float(avg_precision) if avg_precision is not None else None,
        'confusion_matrix': cm.tolist(),
        'class_names': class_names
    }

    with open(os.path.join(save_dir, f"{model_name}_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results

def plot_tsne(model, dataloader, device, class_names, model_name, save_dir, num_samples=1000):
    """
    Generate t-SNE visualization of learned features
    """
    try:
        print(f"Generating t-SNE visualization for {model_name}...")
        features, labels = extract_features(model, dataloader, device, num_samples)
        
        # Limit samples for t-SNE performance
        if len(features) > num_samples:
            indices = np.random.choice(len(features), num_samples, replace=False)
            features = features[indices]
            labels = labels[indices]
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
        features_2d = tsne.fit_transform(features)
        
        # Plot
        plt.figure(figsize=(10, 8))
        colors = plt.cm.Set1(np.linspace(0, 1, len(class_names)))
        
        for i, class_name in enumerate(class_names):
            mask = labels == i
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                       c=[colors[i]], label=class_name, alpha=0.7, s=20)
        
        plt.title(f't-SNE Visualization - {model_name}')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{model_name}_tsne.png"), 
                    dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"t-SNE visualization saved for {model_name}")
        
    except Exception as e:
        print(f"Error generating t-SNE for {model_name}: {e}")

def setup_logging(save_dir, log_filename="evaluation.log"):
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, log_filename)
    
    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_path

def main():
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--model_dirs', type=str, nargs='+', required=True,
                        help='List of model directories containing trained models')
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Path to dataset')
    parser.add_argument('--img_size', type=int, default=224, 
                        help='Input image size')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--save_dir', type=str, default='./evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of classes')
    parser.add_argument('--device', type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for evaluation')
    parser.add_argument('--generate_gradcam', action='store_true',
                        help='Generate Grad-CAM visualizations')
    parser.add_argument('--generate_tsne', action='store_true',
                        help='Generate t-SNE visualizations')
    
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.save_dir)
    logging.info("Starting evaluation session")
    logging.info(f"Model directories: {args.model_dirs}")
    logging.info(f"Device: {args.device}")

    # Load dataset
    dataloaders, dataset_sizes, class_names = get_dataloaders(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size
    )
    device = torch.device(args.device)

    all_results = []

    # Evaluate each model
    for model_dir in args.model_dirs:
        if not os.path.exists(model_dir):
            logging.warning(f"Model directory not found: {model_dir}")
            continue

        # Extract model name from directory
        model_name = os.path.basename(model_dir)
        
        # Load model info
        model_info_path = os.path.join(model_dir, f"{model_name}_model_info.json")
        if os.path.exists(model_info_path):
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)
        else:
            logging.warning(f"Model info not found for {model_name}, using defaults")
            model_info = {
                'model_name': model_name,
                'num_classes': args.num_classes,
                'img_size': args.img_size,
                'class_names': class_names
            }

        # Load trained model
        model_path = os.path.join(model_dir, f"{model_name}_best.pth")
        if not os.path.exists(model_path):
            logging.warning(f"Model weights not found: {model_path}")
            continue

        logging.info(f"Evaluating model: {model_name}")
        
        try:
            # Initialize model
            model = get_model(model_name, num_classes=model_info['num_classes']).to(device)
            # Load trained weights
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint) 
            # Create save directory for this model
            model_save_dir = os.path.join(args.save_dir, model_name)    
            # Evaluate on test set
            if 'test' in dataloaders and dataloaders['test'] is not None:
                results = evaluate_model(model, dataloaders['test'], device, 
                                       class_names, model_name, model_save_dir)
                all_results.append(results)
            else:
                logging.warning(f"No test dataloader available, using validation set")
                results = evaluate_model(model, dataloaders['val'], device, 
                                       class_names, model_name, model_save_dir)
                all_results.append(results)
            
            # Grad-CAM visualizations 
            if args.generate_gradcam:
                logging.info(f"Generating Grad-CAM for {model_name}")
                gradcam_dir = os.path.join(model_save_dir, 'gradcam')
                test_loader = dataloaders['test'] if 'test' in dataloaders else dataloaders['val']
                try:
                    generate_gradcam_for_test(model, test_loader, device, gradcam_dir)
                except Exception as e:
                    logging.error(f"Error generating Grad-CAM for {model_name}: {e}")
            
            # t-SNE visualization 
            if args.generate_tsne:
                test_loader = dataloaders['test'] if 'test' in dataloaders else dataloaders['val']
                plot_tsne(model, test_loader, device, class_names, model_name, model_save_dir)
            
            logging.info(f"Evaluation completed for {model_name}")
            
        except Exception as e:
            logging.error(f"Error evaluating {model_name}: {e}")
            continue

    # Generate comparison report
    if all_results:
        logging.info("Generating comparison report...")
        generate_comparison_report(all_results, args.save_dir)
    
    logging.info("All evaluations completed!")

def generate_comparison_report(all_results, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    # comparison DataFrame
    comparison_data = []
    for result in all_results:
        data_row = {
            'Model': result['model_name'],
            'Accuracy': result['accuracy'],
            'Precision': result['precision'],
            'Recall': result['recall'],
            'F1-Score': result['f1_score'],
            'ROC-AUC': result.get('roc_auc', 'N/A'),
            'Avg Precision': result.get('avg_precision', 'N/A')
        }
        
        # Add medical-specific metrics if available
        if 'sensitivity' in result:
            data_row.update({
                'Sensitivity': result['sensitivity'],
                'Specificity': result['specificity'],
                'PPV': result['ppv'],
                'NPV': result['npv'],
                'FPR': result['false_positive_rate'],
                'FNR': result['false_negative_rate']
            })
        
        comparison_data.append(data_row)
    
    df = pd.DataFrame(comparison_data)
    
    # Save as CSV
    df.to_csv(os.path.join(save_dir, 'model_comparison.csv'), index=False)
    
    # Save as formatted text report
    with open(os.path.join(save_dir, 'comparison_report.txt'), 'w') as f:
        f.write("MODEL COMPARISON REPORT - MEDICAL BINARY CLASSIFICATION\n")
        f.write("=" * 60 + "\n\n")
        f.write(df.to_string(index=False, float_format='%.4f'))
        f.write("\n\n")
        
        # Find best model for each metric
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        if 'Sensitivity' in df.columns:
            metrics.extend(['Sensitivity', 'Specificity', 'PPV', 'NPV'])
            
        f.write("BEST MODELS BY METRIC:\n")
        f.write("-" * 30 + "\n")
        for metric in metrics:
            if metric in df.columns and df[metric].dtype in ['float64', 'int64']:
                try:
                    best_idx = df[metric].idxmax()
                    best_model = df.loc[best_idx, 'Model']
                    best_value = df.loc[best_idx, metric]
                    f.write(f"{metric}: {best_model} ({best_value:.4f})\n")
                except:
                    continue
    
    # Generate comparison plots
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    medical_metrics = []
    
    if 'Sensitivity' in df.columns:
        medical_metrics = ['Sensitivity', 'Specificity', 'PPV', 'NPV']
        
    # Main performance metrics plot
    if len(metrics_to_plot) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison - Core Metrics', fontsize=16)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(df)))
        
        for i, metric in enumerate(metrics_to_plot):
            if metric in df.columns:
                ax = axes[i // 2, i % 2]
                bars = ax.bar(df['Model'], df[metric], color=colors)
                ax.set_title(f'{metric} Comparison')
                ax.set_ylabel(metric)
                ax.set_ylim(0, 1)
                
                # Add value labels on bars
                for bar, value in zip(bars, df[metric]):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=9)
                
                # Rotate x-axis labels if needed
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'model_comparison_core.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    if len(medical_metrics) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison - Medical Metrics', fontsize=16)
        
        for i, metric in enumerate(medical_metrics):
            if metric in df.columns:
                ax = axes[i // 2, i % 2]
                bars = ax.bar(df['Model'], df[metric], color=colors)
                ax.set_title(f'{metric} Comparison')
                ax.set_ylabel(metric)
                ax.set_ylim(0, 1)
                
                # Add value labels on bars
                for bar, value in zip(bars, df[metric]):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=9)
                
                # Rotate x-axis labels if needed
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'model_comparison_medical.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Comparison report saved to {save_dir}")
    print("\nModel Rankings (by F1-Score):")
    if 'F1-Score' in df.columns:
        df_sorted = df.sort_values('F1-Score', ascending=False)
        for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
            base_info = f"{i}. {row['Model']}: F1={row['F1-Score']:.4f}, Acc={row['Accuracy']:.4f}"
            if 'Sensitivity' in row:
                medical_info = f", Sens={row['Sensitivity']:.4f}, Spec={row['Specificity']:.4f}"
                print(base_info + medical_info)
            else:
                print(base_info)

if __name__ == "__main__":
    main()