import os, json, argparse, time, random
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import EuroSAT as TVEuroSAT

from src.reproducibility import set_seed
from src.preprocess import get_transforms
from src.eurosat_dataset import EuroSAT
from src.model_loader import load_vit_base_model
from src.training import train_and_evaluate
from src.model_utils import calculate_confusion_matrix, calculate_f1_score, calculate_balanced_accuracy
from src.plot_helpers import plot_history

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="./data")
    ap.add_argument("--out_dir", type=str, default="runs/seed42")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--early_stop", type=int, default=5)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--model", type=str, default="vit_base_patch16_224")
    ap.add_argument("--num_classes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def main():
    args = parse_args()
    ensure_dir(args.out_dir)
    set_seed(args.seed)

    device = args.device
    print("Device:", device)

    # Data
    train_tf, test_tf = get_transforms(args.img_size)
    full_ds = TVEuroSAT(root=args.root, download=True, transform=None)
    n_total = len(full_ds)
    n_train = int(0.8 * n_total)
    n_test = n_total - n_train
    train_raw, test_raw = random_split(full_ds, [n_train, n_test], generator=torch.Generator().manual_seed(args.seed))

    train_ds = EuroSAT(train_raw, transform=train_tf)
    test_ds  = EuroSAT(test_raw, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Model
    model, n_params, index_to_class, class_to_index = load_vit_base_model(
        model_name=args.model, pretrained=True, device=device, num_classes=args.num_classes
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # Train & evaluate
    best_path = os.path.join(args.out_dir, "best_model.pth")
    history, final_metrics, total_time = train_and_evaluate(
        model, train_loader, test_loader, criterion, optimizer, device,
        n_epochs=args.epochs, early_stopping_threshold=args.early_stop, final_model_path=best_path
    )

    # Save history, metrics
    with open(os.path.join(args.out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(args.out_dir, "final_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)

    # Plots
    plot_history(history, out_png=os.path.join(args.out_dir, "training_curves.png"))

    # Confusion Matrix
    cm = calculate_confusion_matrix(model, test_loader, device)
    import numpy as np, matplotlib.pyplot as plt
    fig = plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("True")
    tick_marks = np.arange(len(index_to_class))
    plt.xticks(tick_marks, [index_to_class[i] for i in range(len(index_to_class))], rotation=45, ha="right")
    plt.yticks(tick_marks, [index_to_class[i] for i in range(len(index_to_class))])
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "confusion_matrix.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Sample XAI for a few random test images
    from src.XAI import attribution_maps
    ensure_dir(os.path.join(args.out_dir, "xai"))
    for _ in range(3):
        idx = random.randrange(len(test_ds))
        pil_img, label_idx = test_ds.dataset[idx]
        x = test_tf(pil_img).unsqueeze(0)
        svg_path, pred_idx, pred_score = attribution_maps(
            input_image=x, model=model, true_label_idx=label_idx, device=device,
            index_to_class=index_to_class, output_dir=os.path.join(args.out_dir, "xai"),
            alpha=0.6, threshold_percentile=99.0, blur_radius=5
        )
        print("XAI saved:", svg_path)

    # Print summary
    print("Params:", n_params)
    print("Final metrics:", final_metrics)
    print("Total time (s):", total_time)

if __name__ == "__main__":
    main()
