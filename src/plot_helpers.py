import matplotlib.pyplot as plt

def plot_history(history, out_png="training_curves.png"):
    epochs = [r["epoch"] for r in history]
    train_loss = [r["train_loss"] for r in history]
    test_loss = [r["test_loss"] for r in history]
    train_acc = [r["train_accuracy"] for r in history]
    test_acc = [r["test_accuracy"] for r in history]

    # Loss
    plt.figure(figsize=(6,4))
    plt.plot(epochs, train_loss, label="train")
    plt.plot(epochs, test_loss, label="test")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss vs Epoch")
    plt.legend(); plt.tight_layout(); plt.savefig(out_png.replace(".png","_loss.png"), dpi=200)

    # Accuracy
    plt.figure(figsize=(6,4))
    plt.plot(epochs, train_acc, label="train")
    plt.plot(epochs, test_acc, label="test")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.title("Accuracy vs Epoch")
    plt.legend(); plt.tight_layout(); plt.savefig(out_png.replace(".png","_acc.png"), dpi=200)
