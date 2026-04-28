import os
import sys
import time
import torch
import csv
import torch.nn as nn
import torch.optim as optim
import argparse

from models import TeacherModel
from data import get_dataloaders
from utils import set_seed


def train_one_epoch(model, optimizer, dataloader, criterion, device, epoch, total_epochs):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    num_batches = len(dataloader)
    print(f"\nEpoch {epoch}/{total_epochs} [TRAIN] - Processing {num_batches} batches...", flush=True)
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        
        # Accumulate metrics
        total_loss += loss.item() * data.size(0)
        _, preds = torch.max(outputs, dim=1)
        total_correct += (preds == target).sum().item()
        total_samples += data.size(0)
        
        # Print progress every 10 batches
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
            running_acc = 100.0 * total_correct / total_samples
            print(f"  Batch [{batch_idx+1}/{num_batches}] | "
                  f"Loss: {loss.item():.4f} | "
                  f"Acc: {running_acc:.2f}%", flush=True)
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    
    return avg_loss, accuracy


def eval_one_epoch(model, dataloader, criterion, device, epoch, total_epochs):
    """Evaluate for one epoch."""
    model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    num_batches = len(dataloader)
    print(f"\nEpoch {epoch}/{total_epochs} [EVAL] - Processing {num_batches} batches...", flush=True)
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            outputs = model(data)
            loss = criterion(outputs, target)
            
            # Accumulate metrics
            total_loss += loss.item() * data.size(0)
            _, preds = torch.max(outputs, dim=1)
            total_correct += (preds == target).sum().item()
            total_samples += data.size(0)
            
            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                running_acc = 100.0 * total_correct / total_samples
                print(f"  Batch [{batch_idx+1}/{num_batches}] | "
                      f"Loss: {loss.item():.4f} | "
                      f"Acc: {running_acc:.2f}%", flush=True)
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    
    return avg_loss, accuracy




def main():
    parser = argparse.ArgumentParser(description="Fine-tune Teacher Model")
    
    # Model and dataset arguments
    parser.add_argument('--model', type=str, default='resnet50', 
                        help='Model architecture (e.g., resnet18, resnet50, vgg16)')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='Load ImageNet pretrained weights (default: random init)')
    parser.add_argument('--dataset', type=str, default='CIFAR100', 
                        help='Dataset to use (CIFAR10, CIFAR100, IMAGENETTE, FOOD101, CUSTOM)')
    parser.add_argument('--data-root', type=str, default='./data', 
                        help='Root directory for dataset')
    parser.add_argument('--model-family', type=str, default='auto',
                        choices=['auto', 'cnn', 'vit'],
                        help='Backbone family. auto infers from model name.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=100, 
                        help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=128, 
                        help='Input batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, 
                        help='Initial learning rate')
    
    # Learning rate scheduler arguments
    parser.add_argument('--use-scheduler', action='store_true', default=False,
                        help='Use learning rate scheduler')
    parser.add_argument('--step-size', type=int, default=30, 
                        help='Period of learning rate decay (epochs)')
    parser.add_argument('--lr-decay', type=float, default=0.1, 
                        help='Multiplicative factor of learning rate decay (gamma)')
    
    # Data augmentation
    parser.add_argument('--jitter', action='store_true', default=False,
                        help='Apply color jitter augmentation')
    parser.add_argument('--edge', action='store_true', default=False,
                        help='Apply edge detection transform')
    parser.add_argument('--noise', action='store_true', default=False,
                        help='Add noise to images')
    
    # Save arguments
    parser.add_argument('--save-dir', type=str, default='teacher_weights', 
                        help='Directory to save model checkpoints')
    parser.add_argument('--log-dir', type=str, default='logs', 
                        help='Directory to save training logs')
    parser.add_argument('--experiment-name', type=str, default='teacher_finetune', 
                        help='Name for this experiment (used in filenames)')
    
    args = parser.parse_args()
    set_seed(args.seed)

    if args.model_family == 'auto':
        args.model_family = 'vit' if args.model.lower().startswith('vit') else 'cnn'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*80}", flush=True)
    print(f"{'TEACHER MODEL FINE-TUNING':^80}", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Model: {args.model} ({'ImageNet pretrained' if args.pretrained else 'random init'})", flush=True)
    print(f"Model family: {args.model_family}", flush=True)
    print(f"Dataset: {args.dataset}", flush=True)
    print(f"Batch Size: {args.batch_size} | Epochs: {args.epochs}", flush=True)
    print(f"Learning Rate: {args.lr}", flush=True)
    print(f"Seed: {args.seed}", flush=True)
    if args.use_scheduler:
        print(f"LR Scheduler: StepLR (step_size={args.step_size}, gamma={args.lr_decay})", flush=True)
    print(f"Augmentations - Jitter: {args.jitter}, Edge: {args.edge}, Noise: {args.noise}", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    # Get data loaders
    train_loader, eval_loader, num_classes = get_dataloaders(
        dataset=args.dataset,
        batch_size=args.batch_size,
        root=args.data_root,
        jitter=args.jitter,
        edge=args.edge,
        noise=args.noise,
        model_family=args.model_family,
    )
    
    print(f"Dataset loaded: {len(train_loader.dataset)} train samples, "
          f"{len(eval_loader.dataset)} test samples, {num_classes} classes\n", flush=True)
    
    # Initialize model (pretrained ImageNet weights only if --pretrained flag is set)
    model = TeacherModel(
        model_name=args.model,
        num_classes=num_classes,
        weights_path=None,
        pretrained=args.pretrained,
        model_family=args.model_family,
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {trainable_params/1e6:.2f}M trainable / {total_params/1e6:.2f}M total\n", flush=True)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    scheduler = None
    if args.use_scheduler:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.step_size,
            gamma=args.lr_decay
        )
        print(f"Using StepLR scheduler with step_size={args.step_size}, gamma={args.lr_decay}\n", flush=True)
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    csv_path = os.path.join(args.log_dir, f'{args.experiment_name}.csv')
    best_acc = 0.0
    best_epoch = 0
    
    # Training loop
    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])
        
        for epoch in range(1, args.epochs + 1):
            start_time = time.perf_counter()
            
            # Training
            train_loss, train_acc = train_one_epoch(
                model=model,
                optimizer=optimizer,
                dataloader=train_loader,
                criterion=criterion,
                device=device,
                epoch=epoch,
                total_epochs=args.epochs
            )
            
            # Evaluation
            eval_loss, eval_acc = eval_one_epoch(
                model=model,
                dataloader=eval_loader,
                criterion=criterion,
                device=device,
                epoch=epoch,
                total_epochs=args.epochs
            )
            
            # Step scheduler
            if scheduler is not None:
                scheduler.step()
            
            end_time = time.perf_counter()
            epoch_time = end_time - start_time
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log to CSV
            writer.writerow([
                epoch, train_loss, train_acc, eval_loss, eval_acc, current_lr
            ])
            file.flush()
            
            # Save best model
            if eval_acc > best_acc:
                best_acc = eval_acc
                best_epoch = epoch
                best_model_path = os.path.join(args.save_dir, f'{args.experiment_name}_best.pth')
                torch.save(model.model.state_dict(), best_model_path)
            
            # Print epoch summary
            print(f"\n{'='*80}", flush=True)
            print(f"EPOCH {epoch}/{args.epochs} SUMMARY ({epoch_time:.2f}s) | LR: {current_lr:.6f}", flush=True)
            print(f"{'='*80}", flush=True)
            print(f"TRAIN | Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}%", flush=True)
            print(f"EVAL  | Loss: {eval_loss:.4f} | Acc: {eval_acc*100:.2f}%", flush=True)
            print(f"{'='*80}", flush=True)
            print(f"Best Accuracy: {best_acc*100:.2f}% (Epoch {best_epoch})", flush=True)
            print(f"{'='*80}\n", flush=True)
    
    # Save final model
    final_model_path = os.path.join(args.save_dir, f'{args.experiment_name}_final.pth')
    torch.save(model.model.state_dict(), final_model_path)
    
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
