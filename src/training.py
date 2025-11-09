import time
import copy
import torch
from model_utils import (
    train, evaluate, calculate_f1_score,
    calculate_confusion_matrix, calculate_balanced_accuracy
)

def train_and_evaluate(
    model, train_loader, test_loader, criterion, optimizer, device,
    n_epochs: int = 20,
    early_stopping_threshold: int = 5,
    final_model_path: str = "best_model.pth",
):
    best_state = copy.deepcopy(model.state_dict())
    best_metric = float("inf")  # monitor test loss
    best_epoch = -1

    history = []
    total_elapsed_time = 0.0
    patience = 0

    for epoch in range(n_epochs):
        start_time = time.time()
        print(f"\nEpoch {epoch+1}/{n_epochs}")

        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        f1 = calculate_f1_score(model, test_loader, device)
        bal_acc = calculate_balanced_accuracy(model, test_loader, device)

        # log
        rec = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "f1_score": float(f1),
            "balanced_accuracy": float(bal_acc),
        }
        history.append(rec)
        print(f"F1: {f1:.4f} | Balanced Acc: {bal_acc:.4f}")

        # early stopping on test loss
        if test_loss < best_metric:
            best_metric = test_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= early_stopping_threshold:
                print(f"Early stopping at epoch {epoch+1} (best epoch: {best_epoch}).")
                break

        total_elapsed_time += time.time() - start_time

    # restore & save best
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved best model from epoch {best_epoch} to {final_model_path}")

    final_metrics = history[best_epoch - 1] if best_epoch > 0 else history[-1]
    return history, final_metrics, total_elapsed_time
