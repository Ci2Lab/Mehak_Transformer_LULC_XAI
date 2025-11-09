import torch
from sklearn.metrics import f1_score, confusion_matrix, balanced_accuracy_score
from tqdm.auto import tqdm  # works in notebooks & scripts

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, running_total_correct = 0.0, 0

    for inputs, labels in tqdm(dataloader, desc="Train"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_total_correct += torch.sum(preds == labels).item()

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_accuracy = 100.0 * running_total_correct / len(dataloader.dataset)
    print(f"Train Loss: {epoch_loss:.4f}; Accuracy: {epoch_accuracy:.2f}")
    return epoch_loss, epoch_accuracy

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss, running_total_correct = 0.0, 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Eval"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            running_total_correct += torch.sum(preds == labels).item()

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_accuracy = 100.0 * running_total_correct / len(dataloader.dataset)
    print(f"Test  Loss: {epoch_loss:.4f}; Accuracy: {epoch_accuracy:.2f}")
    return epoch_loss, epoch_accuracy

def calculate_f1_score(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    return f1_score(all_labels, all_preds, average='weighted')

def calculate_confusion_matrix(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    return confusion_matrix(all_labels, all_preds)

def calculate_balanced_accuracy(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    return balanced_accuracy_score(all_labels, all_preds)
