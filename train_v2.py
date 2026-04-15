import os
import time
import torch
import csv 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse

from tqdm import tqdm
from sae_weightcompress import SAEInjection
from models import TeacherModel, StudentModel
from data import get_dataloaders


def train_one_epoch(model, optimizer, scheduler, dataloader, device, temperature=4.0, kl_weight=1.0, cls_weight=1.0, sae_weight=1.0, progress_bar=None):
    """
    Train one epoch.
    """
    model.train()
    
    total_loss = 0.0
    total_sae_loss = 0.0
    total_cls_loss = 0.0
    total_kl_loss = 0.0
    total_correct = 0
    total_samples = 0

    classification_criterion = nn.CrossEntropyLoss()
    distill_criterion = nn.KLDivLoss(reduction="batchmean")

    for data, target in dataloader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        teacher_logits, student_logits, sae_loss = model(data)
        sae_loss = sae_loss if sae_loss is not None else torch.tensor(0.0, dtype=torch.float, device=device)

        cls_loss = classification_criterion(student_logits, target)
        kl_loss = distill_criterion(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1)
        ) * (temperature ** 2)  

        loss = kl_weight * kl_loss + cls_weight * cls_loss + sae_weight * sae_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_sae_loss += sae_loss.item()
        total_cls_loss += cls_loss.item()
        total_kl_loss += kl_loss.item()

        _, preds = torch.max(student_logits, dim=1)
        total_correct += (preds == target).sum().item()
        total_samples += data.size(0)

        if progress_bar:
            progress_bar.update(1)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    avg_sae_loss = total_sae_loss / total_samples
    avg_cls_loss = total_cls_loss / total_samples
    avg_kl_loss = total_kl_loss / total_samples

    return avg_loss, accuracy, avg_sae_loss, avg_cls_loss, avg_kl_loss


def eval_one_epoch(model, dataloader, device, temperature=4.0, kl_weight=1.0, cls_weight=1.0, sae_weight=1.0):
    """
    Evaluate one epoch.
    """
    model.eval()
        
    total_loss = 0.0
    total_sae_loss = 0.0
    total_cls_loss = 0.0
    total_kl_loss = 0.0
    total_correct = 0
    total_samples = 0

    classification_criterion = nn.CrossEntropyLoss()
    distill_criterion = nn.KLDivLoss(reduction="batchmean")

    with torch.no_grad():
        with tqdm(total=len(dataloader), desc="Evaluating", unit="batch") as pbar:
            for data, target in dataloader:
                data, target = data.to(device), target.to(device)

                teacher_logits, student_logits, sae_loss = model(data)
                sae_loss = sae_loss if sae_loss is not None else torch.tensor(0.0, dtype=torch.float, device=device)

                cls_loss = classification_criterion(student_logits, target)
                kl_loss = distill_criterion(
                    F.log_softmax(student_logits / temperature, dim=1),
                    F.softmax(teacher_logits / temperature, dim=1)
                ) * (temperature ** 2)

                loss = kl_weight * kl_loss + cls_weight * cls_loss + sae_weight * sae_loss

                total_loss += loss.item()
                total_sae_loss += sae_loss.item()
                total_cls_loss += cls_loss.item()
                total_kl_loss += kl_loss.item()

                _, preds = torch.max(student_logits, dim=1)
                total_correct += (preds == target).sum().item()
                total_samples += data.size(0)

                pbar.update(1)

    avg_loss = total_loss / total_samples
    avg_sae_loss = total_sae_loss / total_samples
    avg_cls_loss = total_cls_loss / total_samples
    avg_kl_loss = total_kl_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy, avg_sae_loss, avg_cls_loss, avg_kl_loss


def main():
    torch.set_seed(42)
    parser = argparse.ArgumentParser(description="SAE Injection Knowledge Distillation")
    
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('--batch-size', type=int, default=64, help='Input batch size for training.')
    parser.add_argument('--dataset', type=str, default='CIFAR100', help='Dataset to use.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--teacher-model', type=str, default='resnet50', help='Teacher model architecture.')
    parser.add_argument('--student-model', type=str, default='resnet18', help='Student model architecture.')
    parser.add_argument('--teacher-weights', type=str, default=None, help='Path to teacher weights.')
    
    parser.add_argument('--cls-weight', type=float, default=1.0, help='Weight for the classification loss.')
    parser.add_argument('--kl-weight', type=float, default=1.0, help='Weight for the KL loss.')
    parser.add_argument('--sae-weight', type=float, default=1.0, help='Weight for the SAE loss.')
    parser.add_argument('--temperature', type=float, default=4.0, help='Temperature for KL loss.')

    parser.add_argument('--save-dir', type=str, default='logs', help='Directory to save logs.')
    parser.add_argument('--model-dir', type=str, default='weights', help='Directory to save model weights.')
    parser.add_argument('--experiment-name', type=str, default='sae_injection', help='Name of csv log.')
    
    parser.add_argument('--teacher-layers', nargs='+', required=True,
                        help='Space-separated list of teacher layer names')
    parser.add_argument('--teacher-channels', nargs='+', type=int, required=True,
                        help='Space-separated list of channel counts matching --teacher-layers')
    parser.add_argument('--student-layers', nargs='+', required=True,
                        help='Space-separated list of student layer names')
    parser.add_argument('--student-channels', nargs='+', type=int, required=True,
                        help='Space-separated list of channel counts matching --student-layers')
    parser.add_argument('--sparsity', type=float, default=1e-4, help='Weight of L1 regularizer in the SAE.')
    
    # NEW: Weight compression arguments
    parser.add_argument('--teacher-weight-layers', nargs='+', default=None,
                        help='Space-separated list of teacher layers whose weights to compress (e.g., layer1[0].conv1)')
    parser.add_argument('--student-weight-layers', nargs='+', default=None,
                        help='Space-separated list of student layers to inject compressed weights (e.g., layer1[0].conv1)')

    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if len(args.teacher_layers) != len(args.teacher_channels):
        parser.error("Number of --teacher-layers must match number of --teacher-channels")
    if len(args.student_layers) != len(args.student_channels):
        parser.error("Number of --student-layers must match number of --student-channels")
    
    # Validate weight compression arguments
    if (args.teacher_weight_layers is None) != (args.student_weight_layers is None):
        parser.error("Both --teacher-weight-layers and --student-weight-layers must be provided together or omitted together")
    
    if args.teacher_weight_layers is not None and len(args.teacher_weight_layers) != len(args.student_weight_layers):
        parser.error("Number of --teacher-weight-layers must match number of --student-weight-layers")

    train_loader, eval_loader, num_classes = get_dataloaders(
        dataset=args.dataset, batch_size=args.batch_size, jitter=False, edge=False, noise=False)

    teacher = TeacherModel(model_name=args.teacher_model, num_classes=num_classes, weights_path=args.teacher_weights).to(device)
    student = StudentModel(model_name=args.student_model, num_classes=num_classes).to(device)
    
    for param in teacher.parameters():
        param.requires_grad = False

    model = SAEInjection(
        teacher,
        student,
        teacher_layer_names=args.teacher_layers,
        teacher_channels=args.teacher_channels,
        student_layer_names=args.student_layers,
        student_channels=args.student_channels,
        sparsity=args.sparsity,
        teacher_weight_layers=args.teacher_weight_layers,
        student_weight_layers=args.student_weight_layers,
    ).to(device)
    
    # Optimizer includes student parameters, SAE adapters, and weight compressors
    optimizer_params = list(student.parameters()) + list(model.sae_adapters.parameters())
    if model.weight_compressors:
        optimizer_params += list(model.weight_compressors.parameters())
    
    optimizer = optim.Adam(optimizer_params, lr=args.lr)

    total_params = sum(p.numel() for group in optimizer.param_groups for p in group['params'])
    print(f"Total parameters passed to optimizer: {total_params*1e-6:.2f}M")

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    csv_path = os.path.join(args.save_dir, f'{args.experiment_name}.csv')
    best_acc = 0.0
    best_epoch = 0

    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "train_sae_loss", "train_cls_loss", "train_kl_loss", "train_acc", 
                         "val_loss", "val_sae_loss", "val_cls_loss", "val_kl_loss", "val_acc", "lr"])

        for epoch in range(1, args.epochs + 1):
            start = time.perf_counter()

            with tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{args.epochs}", unit="batch") as pbar:
                train_loss, train_acc, train_sae_loss, train_cls_loss, train_kl_loss = train_one_epoch(
                    model, optimizer, None, train_loader, device,
                    temperature=args.temperature, kl_weight=args.kl_weight, cls_weight=args.cls_weight, 
                    sae_weight=args.sae_weight, progress_bar=pbar
                )

            eval_loss, eval_acc, eval_sae_loss, eval_cls_loss, eval_kl_loss = eval_one_epoch(
                model, eval_loader, device,
                temperature=args.temperature, kl_weight=args.kl_weight, cls_weight=args.cls_weight, 
                sae_weight=args.sae_weight
            )

            current_lr = optimizer.param_groups[0]['lr']
            end = time.perf_counter()

            writer.writerow([
                epoch, train_loss, train_sae_loss, train_cls_loss, train_kl_loss, train_acc,
                eval_loss, eval_sae_loss, eval_cls_loss, eval_kl_loss, eval_acc, current_lr
            ])

            if eval_acc > best_acc:
                best_acc = eval_acc
                best_epoch = epoch

            print(f"\nEpoch {epoch}/{args.epochs} ({end-start:.2f}s) | LR: {current_lr:.6f}")
            print(f"Training Loss: {train_loss:.4f} | Training SAE Loss: {train_sae_loss:.8f} | Training Accuracy: {train_acc*100:.2f}%")
            print(f"Evaluation Loss: {eval_loss:.4f} | Evaluation SAE Loss: {eval_sae_loss:.8f} | Evaluation Accuracy: {eval_acc*100:.2f}%")
    
    model_path = os.path.join(args.model_dir, f'{args.experiment_name}.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path} at epoch #{best_epoch}.")


if __name__ == '__main__':
    main()