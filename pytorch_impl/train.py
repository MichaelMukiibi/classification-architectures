import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

def get_args():
    parser = argparse.ArgumentParser(description="Classification Benchmark API")
    
    # Core CLI Parameters
    parser.add_argument(
        "--model", 
        type=str, 
        default="resnet", 
        choices=["resnet", "densenet", "swin"],
        help="Target model architecture for training"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="cifar10", 
        choices=["cifar10"],
        help="Target dataset for evaluation"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=10,
        help="Number of full training epochs"
    )
    
    # Optional Hyperparameters (Defaults Maintained)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    
    return parser.parse_args()

def build_model(model_name):
    if model_name == "resnet":
        model = models.resnet50(num_classes=10)
    elif model_name == "densenet":
        model = models.densenet121(num_classes=10)
    elif model_name == "swin":
        # Swin Transformer tiny architecture
        model = models.swin_t(num_classes=10, weights=None)
    return model

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Model: {args.model}")

    # Swin expects 224x224 images; upscaling CIFAR-10 for unified architecture benchmarking
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model(args.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100.0 * correct / total
        print(f"Epoch [{epoch+1}/{args.epochs}] -> Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%")

    # Evaluation loop
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    print(f"\nFinal Validation Accuracy for {args.model}: {100.0 * correct / total:.2f}%")

if __name__ == "__main__":
    main()