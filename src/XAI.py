import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import cv2
from captum.attr import IntegratedGradients

def _unnormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    t = tensor.clone()
    for ch, m, s in zip(t, mean, std):
        ch.mul_(s).add_(m)
    return t

@torch.no_grad()
def _softmax_top1(logits: torch.Tensor):
    probs = F.softmax(logits, dim=1)
    score, idx = torch.topk(probs, 1)
    return score.item(), idx.item()

def attribution_maps(
    input_image: torch.Tensor,            # shape (1, C, H, W), normalized
    model: torch.nn.Module,
    true_label_idx: int,
    device: str,
    index_to_class: dict[int, str],
    output_dir: str = "xai_outputs",
    alpha: float = 0.5,
    threshold_percentile: float = 99.0,
    blur_radius: int = 5,
):
    """
    Saves an SVG showing original image + IG-based attribution overlay.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    # forward (keep grads for IG)
    input_image = input_image.to(device)
    logits = model(input_image)
    pred_score, pred_idx = _softmax_top1(logits)
    predicted_class_name = index_to_class.get(pred_idx, str(pred_idx))
    actual_label_name = index_to_class.get(true_label_idx, str(true_label_idx))

    ig = IntegratedGradients(model)
    target_label = torch.tensor([true_label_idx], device=device)
    attributions, _ = ig.attribute(
        inputs=input_image, target=target_label, return_convergence_delta=True
    )
    attr = attributions.squeeze(0).abs().sum(dim=0).detach().cpu().numpy()
    attr = attr / (attr.max() + 1e-12)

    colors = [(1, 0.5, 0), (0, 1, 0), (0, 0, 1)]  # Orange→Green→Blue
    cm = LinearSegmentedColormap.from_list('orange_green_blue', colors, N=100)

    heat_rgba = (cm(attr) * 255).astype(np.uint8)
    heat_rgba[..., 3] = (attr * 255).astype(np.uint8)

    if blur_radius and blur_radius > 0:
        k = blur_radius if blur_radius % 2 == 1 else blur_radius + 1
        heat_rgba = cv2.GaussianBlur(heat_rgba, (k, k), 0)

    orig = _unnormalize(input_image[0].detach().cpu()).clamp(0, 1).numpy()
    orig = (np.transpose(orig, (1, 2, 0)) * 255).astype(np.uint8)

    thr = np.percentile(attr, threshold_percentile)
    mask = (attr > thr)[..., None]
    overlay_rgb = np.where(mask, heat_rgba[..., :3], 0)

    import cv2 as _cv2
    blended = _cv2.addWeighted(overlay_rgb.astype('uint8'), alpha, orig.astype('uint8'), 1 - alpha, 0)

    h, w = orig.shape[:2]
    aspect = w / h if h else 1.0
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(orig)
    ax1.set_title(f"Actual: {actual_label_name}")
    ax1.axis("off"); ax1.set_aspect(aspect)

    ax2.imshow(blended)
    ax2.set_title(f"IG Overlay — Pred: {predicted_class_name} (p≈{pred_score:.2f})")
    ax2.axis("off"); ax2.set_aspect(aspect)

    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    plt.colorbar(sm, fraction=0.046, pad=0.04, ax=ax2, orientation="horizontal")
    plt.tight_layout()

    svg_path = os.path.join(output_dir, f'visualization_{true_label_idx}.svg')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    plt.close(fig)
    return svg_path, pred_idx, pred_score
