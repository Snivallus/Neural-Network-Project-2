import matplotlib as mpl
mpl.use('Agg')  # Use non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")  # Changed to darkgrid style
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm as tqdm
import torchvision
import torchvision.transforms as transforms
import argparse

# Model imports with modified paths
from models.vgg import VGG_A
from models.vgg import VGG_A_BatchNorm

# Constants configuration
DEVICE_IDS = [0,1,2,3]
WORKERS_COUNT = 4
SAMPLE_SIZE = 128  # Renamed batch_size

RESULTS_DIR = 'vgg_results'
IMAGE_DIR = os.path.join(RESULTS_DIR, 'visualization')
MODEL_DIR = os.path.join(RESULTS_DIR, 'saved_models')
LOSS_DIR = RESULTS_DIR

# Ensure output directories exist
for path in [IMAGE_DIR, MODEL_DIR, LOSS_DIR]:
    os.makedirs(path, exist_ok=True)

# Device configuration
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def configure_arguments():
    """Setup command-line arguments for training"""
    parser = argparse.ArgumentParser(description='VGG Training Analysis')
    parser.add_argument('--epoch_count', type=int, default=20,
                        help='Number of training cycles (default: 20)')
    parser.add_argument('--learning_rates', type=str, 
                        default='2e-3,1e-3,5e-4,1e-4',
                        help='Comma-separated learning rates (default: 2e-3,1e-3,5e-4,1e-4)')
    parser.add_argument('--seed_value', type=int, default=42,
                        help='Randomization seed (default: 42)')
    parser.add_argument('--initial_skip', type=int, default=25,
                        help='Initial steps to skip in landscape (default: 25)')
    parser.add_argument('--plot_density', type=int, default=1,
                        help='Data point sampling density for plots (default: 1)')
    # Add model saving option
    parser.add_argument('--save_models', action='store_true',
                        help='Save best models during training')
    args = parser.parse_args()
    
    # Convert learning rates to floats
    args.learning_rates = [float(lr) for lr in args.learning_rates.split(',')]
    return args

def evaluate_model(model, data_loader, device):
    """Measure model accuracy on provided dataset"""
    model.eval()
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total_samples += labels.size(0)
            correct_predictions += predicted.eq(labels).sum().item()
    return 100.0 * correct_predictions / total_samples

def initialize_randomness(seed=0, device='cpu'):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def run_training(model, optimizer, loss_fn, train_loader, val_loader, 
                 scheduler=None, epochs=100, best_model_path=None, save_models=False):
    """Execute training process and record metrics"""
    model.to(device)
    epoch_losses = [np.nan] * epochs
    train_acc_history = [np.nan] * epochs
    val_acc_history = [np.nan] * epochs
    best_val_acc = 0
    best_epoch = 0
    best_model_state = None  # Store best model weights

    batch_count = len(train_loader)
    step_losses = []  # Loss for each training step
    
    for epoch in tqdm(range(epochs), desc='Training'):
        if scheduler:
            scheduler.step()
        model.train()

        batch_losses = []  # Losses within current epoch
        epoch_losses[epoch] = 0

        for data in train_loader:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            
            batch_losses.append(loss.item())
            epoch_losses[epoch] += loss.item()
            
            loss.backward()
            optimizer.step()

        step_losses.append(batch_losses)
        epoch_losses[epoch] /= batch_count
        
        # Calculate accuracy metrics
        train_acc = evaluate_model(model, train_loader, device)
        train_acc_history[epoch] = train_acc
        
        if val_loader:
            val_acc = evaluate_model(model, val_loader, device)
            val_acc_history[epoch] = val_acc
            
            # Check if current model is the best
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                
                # Capture best model state
                if save_models:
                    best_model_state = {
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_accuracy': val_acc,
                        'loss': epoch_losses[epoch]
                    }
                
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Loss={epoch_losses[epoch]:.4f}, "
                  f"Train Acc={train_acc:.2f}%, "
                  f"Val Acc={val_acc:.2f}%, "
                  f"Best Val={best_val_acc:.2f}% @ epoch {best_epoch+1}")
        else:
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Loss={epoch_losses[epoch]:.4f}, "
                  f"Train Acc={train_acc:.2f}%")

    # Save best model at end of training
    if save_models and best_model_state and best_model_path:
        torch.save(best_model_state, best_model_path)
        print(f"Saved best model to: {best_model_path} (Val Acc: {best_val_acc:.2f}%)")

    return step_losses

def calculate_boundaries(loss_records):
    """Compute min and max loss boundaries across models"""
    if not loss_records:
        return [], []
    
    try:
        # Flatten all loss records
        all_losses = []
        for model_data in loss_records:
            model_losses = []
            for epoch_data in model_data:
                if isinstance(epoch_data, list):
                    model_losses.extend(epoch_data)
            all_losses.append(model_losses)
        
        # Find minimum length
        min_length = min(len(losses) for losses in all_losses) if all_losses else 0
        
        min_values = []
        max_values = []
        
        # Calculate min/max for each step
        for step_idx in range(min_length):
            step_values = []
            for model_losses in all_losses:
                if step_idx < len(model_losses):
                    step_values.append(model_losses[step_idx])
            
            if step_values:
                min_values.append(min(step_values))
                max_values.append(max(step_values))
        
        return min_values, max_values
    except Exception as e:
        print(f"Boundary calculation error: {e}")
        return [], []

def execute():
    """Main training and analysis workflow"""
    args = configure_arguments()
    epochs = args.epoch_count
    lr_values = args.learning_rates
    skip_steps = args.initial_skip
    save_models = args.save_models  # Get save models flag
    
    print(f"{'='*60}")
    print(f"Commencing Training Analysis")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Training cycles: {epochs}")
    print(f"Learning rates: {lr_values}")
    print(f"Save models: {'Enabled' if save_models else 'Disabled'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(device.index)}")
    
    # Data preparation
    print("Preparing datasets...")
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_data = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=SAMPLE_SIZE, shuffle=True, num_workers=WORKERS_COUNT)

    test_data = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=100, shuffle=False, num_workers=WORKERS_COUNT)

    # Verify data loading
    sample_batch = next(iter(train_loader))
    inputs, labels = sample_batch
    print(f"Input shape: {inputs.shape}, Label shape: {labels.shape}")
    print(f"Value range: [{inputs.min():.4f}, {inputs.max():.4f}]")
    print(f"Sample labels: {labels[:10].tolist()}")
    
    # Training process
    vanilla_vgg_losses = []
    norm_vgg_losses = []

    # Train vanilla VGG models
    for lr in lr_values:
        print(f"\n{'='*60}\nTraining Vanilla VGG (LR={lr})\n{'='*60}")
        initialize_randomness(seed=args.seed_value, device=device)
        model = VGG_A()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Create model save path
        model_path = os.path.join(MODEL_DIR, f'vanilla_vgg_lr_{lr}_best.pth') if save_models else None
        
        losses = run_training(
            model, optimizer, criterion, 
            train_loader, test_loader, 
            epochs=epochs,
            best_model_path=model_path,
            save_models=save_models
        )
        vanilla_vgg_losses.append(losses)
        
        # Flatten and save losses
        all_losses = [loss for epoch in losses for loss in epoch]
        np.savetxt(os.path.join(LOSS_DIR, f'vanilla_vgg_lr_{lr}.txt'), 
                  all_losses, fmt='%.6f')

    # Train VGG with normalization
    for lr in lr_values:
        print(f"\n{'='*60}\nTraining VGG with Normalization (LR={lr})\n{'='*60}")
        initialize_randomness(seed=args.seed_value, device=device)
        model_bn = VGG_A_BatchNorm()
        optimizer_bn = torch.optim.Adam(model_bn.parameters(), lr=lr)
        criterion_bn = nn.CrossEntropyLoss()
        
        # Create model save path
        model_path = os.path.join(MODEL_DIR, f'vgg_bn_lr_{lr}_best.pth') if save_models else None
        
        losses_bn = run_training(
            model_bn, optimizer_bn, criterion_bn, 
            train_loader, test_loader, 
            epochs=epochs,
            best_model_path=model_path,
            save_models=save_models
        )
        norm_vgg_losses.append(losses_bn)
        
        # Flatten and save losses
        all_losses_bn = [loss for epoch in losses_bn for loss in epoch]
        np.savetxt(os.path.join(LOSS_DIR, f'norm_vgg_lr_{lr}.txt'), 
                  all_losses_bn, fmt='%.6f')

    # Calculate loss boundaries
    vanilla_min, vanilla_max = calculate_boundaries(vanilla_vgg_losses)
    norm_min, norm_max = calculate_boundaries(norm_vgg_losses)

    # Generate visualization
    print(f"{'='*60}")
    print(f"Creating loss landscape visualization...")
    print(f"{'='*60}")
    
    try:
        # Process data for plotting
        vanilla_min = np.array(vanilla_min, dtype=float).flatten()[skip_steps:]
        vanilla_max = np.array(vanilla_max, dtype=float).flatten()[skip_steps:]
        norm_min = np.array(norm_min, dtype=float).flatten()[skip_steps:]
        norm_max = np.array(norm_max, dtype=float).flatten()[skip_steps:]
        
        # Apply sampling
        sample_rate = args.plot_density
        steps = np.arange(len(vanilla_min))[::sample_rate]
        vanilla_min = vanilla_min[::sample_rate]
        vanilla_max = vanilla_max[::sample_rate]
        norm_min = norm_min[::sample_rate]
        norm_max = norm_max[::sample_rate]
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        if len(lr_values) > 1:
            plt.fill_between(steps, vanilla_min, vanilla_max, 
                            alpha=0.4, color="#EC0D0D", label='Vanilla VGG')
            plt.fill_between(steps, norm_min, norm_max, 
                            alpha=0.4, color="#0B31EA", label='VGG with Normalization')
        else:
            plt.plot(steps, vanilla_min, 'forestgreen', label='Vanilla VGG')
            plt.plot(steps, norm_min, 'darkviolet', label='VGG with Normalization')
        
        plt.title('Loss Landscape Comparison', fontsize=16)
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Loss Value', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.2)
        
        output_path = os.path.join(IMAGE_DIR, "landscape_comparison.pdf")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved: {output_path}")
        plt.close()
    except Exception as e:
        print(f"Visualization error: {e}")

    print(f"{'='*60}")
    print("Training analysis completed")
    print(f"Results available in: {IMAGE_DIR}")
    if save_models:
        print(f"Saved models available in: {MODEL_DIR}")
    print(f"{'='*60}")

if __name__ == "__main__":
    execute()