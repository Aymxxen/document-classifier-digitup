import torch
import torch.nn.functional as F
import numpy as np
import cv2


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hooks pour intercepter les données au passage
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx=None):
        # 1. Forward pass
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        # 2. Backward pass (calcul des gradients pour la classe cible)
        self.model.zero_grad()
        target = output[0][class_idx]
        target.backward()

        # 3. Génération de la heatmap
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]

        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=0).cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0)  # ReLU

        # Normalisation entre 0 et 1
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)

        return heatmap, class_idx


def overlay_heatmap(heatmap, original_image, alpha=0.4):
    """Superpose la heatmap sur l'image PIL originale"""
    from PIL import Image
    import numpy as np

    img_np = np.array(original_image)

    # Redimensionner la heatmap à la taille de l'image
    heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))

    # Coloriser la heatmap (Bleu -> Rouge)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Fusionner
    overlay = cv2.addWeighted(img_np, 1 - alpha, heatmap_color, alpha, 0)
    return Image.fromarray(overlay)