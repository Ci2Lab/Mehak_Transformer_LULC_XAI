import timm
import torch
from typing import Tuple, Dict, List

def load_vit_base_model(
    model_name: str = "vit_base_patch16_224",
    pretrained: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_classes: int = 10,
) -> Tuple[torch.nn.Module, int, Dict[int, str], Dict[str, int]]:
    """
    Create a ViT model with the correct classifier head size for EuroSAT.
    Returns:
      model: the timm model on the chosen device
      total_params: trainable parameter count
      index_to_class: {idx: name} mapping
      class_to_index: {name: idx} mapping
    """
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    model.eval()
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    classes: List[str] = [
        "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
        "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
    ]
    if len(classes) != num_classes:
        classes = [f"class_{i}" for i in range(num_classes)]

    index_to_class = {i: c for i, c in enumerate(classes)}
    class_to_index = {c: i for i, c in enumerate(classes)}
    return model, total_params, index_to_class, class_to_index
