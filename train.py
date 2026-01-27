import os
import sys
import time
import torch
import csv 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse

from sae_injection_v2 import SAEInjection
from models import TeacherModel, StudentModel
from data import get_dataloaders
from utils import set_seed


def train_one_epoch(model, teacher, optimizer, dataloader, device, temperature=4.0, kl_weight=1.0, cls_weight=1.0, sae_weight=1.0, epoch=1, total_epochs=1):
    """
    Train one epoch with detailed logging using vanilla KD loss.
    """
    model.train()
    teacher.eval()  # Teacher always in eval mode
    
    total_loss = 0.0
    total_sae_loss = 0.0
    total_cls_loss = 0.0
    total_kl_loss = 0.0
    total_correct = 0
    total_samples = 0

    classification_criterion = nn.CrossEntropyLoss()
    distill_criterion = nn.KLDivLoss(reduction="batchmean")

    num_batches = len(dataloader)
    print(f"\nEpoch {epoch}/{total_epochs} [TRAIN] - Processing {num_batches} batches...", flush=True)
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # Get teacher logits (full teacher forward pass)
        with torch.no_grad():
            teacher_logits = teacher(data)

        # Forward pass through hybrid model (student logits via SAE injection)
        student_logits, sae_loss = model(data)
        sae_loss = sae_loss if sae_loss is not None else torch.tensor(0.0, dtype=torch.float, device=device)

        # Classification loss (CE with ground truth)
        cls_loss = classification_criterion(student_logits, target)
        
        # Knowledge distillation loss (KL divergence with teacher)
        kl_loss = distill_criterion(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1)
        ) * (temperature ** 2)
        
        # Total loss: weighted combination of all losses
        loss = cls_weight * cls_loss + kl_weight * kl_loss + sae_weight * sae_loss

        loss.backward()
        optimizer.step()

        # Accumulate metrics
        total_loss += loss.item() * data.size(0)
        total_sae_loss += sae_loss.item() * data.size(0)
        total_cls_loss += cls_loss.item() * data.size(0)
        total_kl_loss += kl_loss.item() * data.size(0)

        _, preds = torch.max(student_logits, dim=1)
        total_correct += (preds == target).sum().item()
        total_samples += data.size(0)

        # Print progress every 10 batches
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
            running_acc = 100.0 * total_correct / total_samples
            print(f"  Batch [{batch_idx+1}/{num_batches}] | "
                  f"Loss: {loss.item():.4f} | "
                  f"CLS: {cls_loss.item():.4f} | "
                  f"KL: {kl_loss.item():.4f} | "
                  f"SAE: {sae_loss.item():.6f} | "
                  f"Acc: {running_acc:.2f}%", flush=True)

    # Calculate epoch averages
    avg_loss = total_loss / total_samples
    avg_sae_loss = total_sae_loss / total_samples
    avg_cls_loss = total_cls_loss / total_samples
    avg_kl_loss = total_kl_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy, avg_sae_loss, avg_cls_loss, avg_kl_loss


def eval_one_epoch(model, teacher, dataloader, device, temperature=4.0, kl_weight=1.0, cls_weight=1.0, sae_weight=1.0, epoch=1, total_epochs=1):
    """
    Evaluate one epoch with detailed logging using vanilla KD loss.
    """
    model.eval()
    teacher.eval()
        
    total_loss = 0.0
    total_sae_loss = 0.0
    total_cls_loss = 0.0
    total_kl_loss = 0.0
    total_correct = 0
    total_samples = 0

    classification_criterion = nn.CrossEntropyLoss()
    distill_criterion = nn.KLDivLoss(reduction="batchmean")

    num_batches = len(dataloader)
    print(f"\nEpoch {epoch}/{total_epochs} [EVAL] - Processing {num_batches} batches...", flush=True)
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)

            # Get teacher logits (full teacher forward pass)
            teacher_logits = teacher(data)

            # Forward pass through hybrid model (student logits via SAE injection)
            student_logits, sae_loss = model(data)
            sae_loss = sae_loss if sae_loss is not None else torch.tensor(0.0, dtype=torch.float, device=device)

            # Classification loss (CE with ground truth)
            cls_loss = classification_criterion(student_logits, target)
            
            # Knowledge distillation loss (KL divergence with teacher)
            kl_loss = distill_criterion(
                F.log_softmax(student_logits / temperature, dim=1),
                F.softmax(teacher_logits / temperature, dim=1)
            ) * (temperature ** 2)
            
            # Total loss
            loss = cls_weight * cls_loss + kl_weight * kl_loss + sae_weight * sae_loss

            # Accumulate metrics
            total_loss += loss.item() * data.size(0)
            total_sae_loss += sae_loss.item() * data.size(0)
            total_cls_loss += cls_loss.item() * data.size(0)
            total_kl_loss += kl_loss.item() * data.size(0)

            _, preds = torch.max(student_logits, dim=1)
            total_correct += (preds == target).sum().item()
            total_samples += data.size(0)

            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                running_acc = 100.0 * total_correct / total_samples
                print(f"  Batch [{batch_idx+1}/{num_batches}] | "
                      f"Loss: {loss.item():.4f} | "
                      f"CLS: {cls_loss.item():.4f} | "
                      f"KL: {kl_loss.item():.4f} | "
                      f"SAE: {sae_loss.item():.6f} | "
                      f"Acc: {running_acc:.2f}%", flush=True)

    # Calculate epoch averages
    avg_loss = total_loss / total_samples
    avg_sae_loss = total_sae_loss / total_samples
    avg_cls_loss = total_cls_loss / total_samples
    avg_kl_loss = total_kl_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy, avg_sae_loss, avg_cls_loss, avg_kl_loss


def main():
    set_seed(42)
    parser = argparse.ArgumentParser(description="SAE Injection Knowledge Distillation")
    
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('--batch-size', type=int, default=64, help='Input batch size for training.')
    parser.add_argument('--dataset', type=str, default='CIFAR100', help='Dataset to use.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--teacher-model', type=str, default='resnet50', help='Teacher model architecture.')
    parser.add_argument('--student-model', type=str, default='resnet18', help='Student model architecture.')
    parser.add_argument('--teacher-weights', type=str, default=None, help='Path to teacher weights.')
    
    parser.add_argument('--cls-weight', type=float, default=1.0, help='Weight for the classification loss.')
    parser.add_argument('--kl-weight', type=float, default=1.0, help='Weight for the KL divergence loss.')
    parser.add_argument('--sae-weight', type=float, default=1.0, help='Weight for the SAE loss.')
    parser.add_argument('--temperature', type=float, default=4.0, help='Temperature for KL divergence loss.')

    parser.add_argument('--save-dir', type=str, default='logs', help='Directory to save logs.')
    parser.add_argument('--model-dir', type=str, default='weights', help='Directory to save model weights.')
    parser.add_argument('--experiment-name', type=str, default='sae_injection', help='Name of csv log.')
    
    # Teacher and student layer arguments
    parser.add_argument('--teacher-layer', type=str, required=True,
                        help='Teacher hint layer name (e.g., layer3)')
    parser.add_argument('--teacher-channels', type=int, required=True,
                        help='Channel count at teacher hint layer')
    parser.add_argument('--student-layers', nargs='+', type=str, required=True,
                        help='Space-separated list of student layer names to include (e.g., layer1 layer3 fc)')
    parser.add_argument('--student-channels', type=int, required=True,
                        help='Channel count at first student layer (for SAE input)')
    parser.add_argument('--sparsity', type=float, default=1e-4, help='Weight of L1 regularizer in the SAE.')
    parser.add_argument('--freeze-teacher', action='store_true', default=False,
                        help='Freeze teacher trunk parameters (default: False)')

    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*80}", flush=True)
    print(f"{'SAE INJECTION TRAINING WITH VANILLA KD':^80}", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Dataset: {args.dataset}", flush=True)
    print(f"Teacher: {args.teacher_model} | Student: {args.student_model}", flush=True)
    print(f"Hint Layer: {args.teacher_layer} ({args.teacher_channels} channels)", flush=True)
    print(f"Student Layers: {args.student_layers}", flush=True)
    print(f"Student Input Channels (SAE): {args.student_channels}", flush=True)
    print(f"Freeze Teacher Trunk: {args.freeze_teacher}", flush=True)
    print(f"Batch Size: {args.batch_size} | Epochs: {args.epochs} | LR: {args.lr}", flush=True)
    print(f"Temperature: {args.temperature}", flush=True)
    print(f"Loss Weights - CLS: {args.cls_weight}, KL: {args.kl_weight}, SAE: {args.sae_weight}", flush=True)
    print(f"{'='*80}\n", flush=True)

    # Get data loaders
    train_loader, eval_loader, num_classes = get_dataloaders(
        dataset=args.dataset, batch_size=args.batch_size, jitter=False, edge=False, noise=False)

    # Load teacher and student models
    teacher = TeacherModel(model_name=args.teacher_model, num_classes=num_classes, weights_path=args.teacher_weights).to(device)
    student = StudentModel(model_name=args.student_model, num_classes=num_classes).to(device)
    
    # Freeze full teacher model (for vanilla KD loss computation)
    for param in teacher.parameters():
        param.requires_grad = False
    teacher.eval()

    # Create SAE Injection hybrid model
    if args.freeze_teacher:
        print(f"Creating SAE Injection model with frozen teacher trunk...", flush=True)
    else:
        print(f"Creating SAE Injection model with trainable teacher trunk...", flush=True)

    model = SAEInjection(
        teacher=teacher,
        student=student,
        teacher_layer_name=args.teacher_layer,
        teacher_channels=args.teacher_channels,
        student_layer_names=args.student_layers,  
        student_channels=args.student_channels,
        sparsity=args.sparsity,
        freeze_teacher=args.freeze_teacher
    ).to(device)
    
    # Optimizer - only train hybrid model parameters (includes SAE and student trunk, optionally teacher trunk)
    optimizer = optim.Adam(model.hybrid_model.parameters(), lr=args.lr)

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.hybrid_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.hybrid_model.parameters())
    teacher_params = sum(p.numel() for p in teacher.parameters())
    print(f"Teacher parameters: {teacher_params/1e6:.2f}M (frozen)", flush=True)
    print(f"Hybrid model trainable parameters: {trainable_params/1e6:.2f}M / {total_params/1e6:.2f}M total", flush=True)
    print(f"{'='*80}\n", flush=True)

    # Create directories for logs and weights
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    csv_path = os.path.join(args.save_dir, f'{args.experiment_name}.csv')
    best_acc = 0.0
    best_epoch = 0

    # Training loop
    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "epoch", "train_loss", "train_sae_loss", "train_cls_loss", "train_kl_loss", "train_acc", 
            "val_loss", "val_sae_loss", "val_cls_loss", "val_kl_loss", "val_acc", "lr"
        ])

        for epoch in range(1, args.epochs + 1):
            start_time = time.perf_counter()

            # Training
            train_loss, train_acc, train_sae_loss, train_cls_loss, train_kl_loss = train_one_epoch(
                model=model,
                teacher=teacher,
                optimizer=optimizer,
                dataloader=train_loader,
                device=device,
                temperature=args.temperature,
                kl_weight=args.kl_weight,
                cls_weight=args.cls_weight,
                sae_weight=args.sae_weight,
                epoch=epoch,
                total_epochs=args.epochs
            )

            # Evaluation
            eval_loss, eval_acc, eval_sae_loss, eval_cls_loss, eval_kl_loss = eval_one_epoch(
                model=model,
                teacher=teacher,
                dataloader=eval_loader,
                device=device,
                temperature=args.temperature,
                kl_weight=args.kl_weight,
                cls_weight=args.cls_weight,
                sae_weight=args.sae_weight,
                epoch=epoch,
                total_epochs=args.epochs
            )

            end_time = time.perf_counter()
            epoch_time = end_time - start_time
            current_lr = optimizer.param_groups[0]['lr']

            # Log to CSV
            writer.writerow([
                epoch, train_loss, train_sae_loss, train_cls_loss, train_kl_loss, train_acc,
                eval_loss, eval_sae_loss, eval_cls_loss, eval_kl_loss, eval_acc, current_lr
            ])
            file.flush()  # Ensure data is written immediately

            # Update best model tracking
            if eval_acc > best_acc:
                best_acc = eval_acc
                best_epoch = epoch
                # Save best model
                best_model_path = os.path.join(args.model_dir, f'{args.experiment_name}_best.pth')
                torch.save(model.state_dict(), best_model_path)

            # Print epoch summary
            print(f"\n{'='*80}", flush=True)
            print(f"EPOCH {epoch}/{args.epochs} SUMMARY ({epoch_time:.2f}s) | LR: {current_lr:.6f}", flush=True)
            print(f"{'='*80}", flush=True)
            print(f"TRAIN | Loss: {train_loss:.4f} | CLS: {train_cls_loss:.4f} | KL: {train_kl_loss:.4f} | SAE: {train_sae_loss:.6f} | Acc: {train_acc*100:.2f}%", flush=True)
            print(f"EVAL  | Loss: {eval_loss:.4f} | CLS: {eval_cls_loss:.4f} | KL: {eval_kl_loss:.4f} | SAE: {eval_sae_loss:.6f} | Acc: {eval_acc*100:.2f}%", flush=True)
            print(f"{'='*80}", flush=True)
            print(f"Best Accuracy: {best_acc*100:.2f}% (Epoch {best_epoch})", flush=True)
            print(f"{'='*80}\n", flush=True)
    
    # Save final model
    final_model_path = os.path.join(args.model_dir, f'{args.experiment_name}_final.pth')
    torch.save(model.state_dict(), final_model_path)
    
    print(f"\n{'='*80}", flush=True)
    print(f"{'TRAINING COMPLETE':^80}", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"Best Validation Accuracy: {best_acc*100:.2f}% (Epoch {best_epoch})", flush=True)
    print(f"Final model saved: {final_model_path}", flush=True)
    print(f"Best model saved: {best_model_path}", flush=True)
    print(f"Training log saved: {csv_path}", flush=True)
    print(f"{'='*80}\n", flush=True)


if __name__ == '__main__':
    main()