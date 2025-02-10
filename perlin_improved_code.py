import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import numpy as np
import copy

def perlin_noise_2d(width, height, freq_x, freq_y, device='cpu'):
    """
    Generate a 2D Perlin noise array using vectorized PyTorch operations.

    Parameters:
    - width (int): Width of the output noise image.
    - height (int): Height of the output noise image.
    - freq_x (int): Frequency of the noise along the x-axis.
    - freq_y (int): Frequency of the noise along the y-axis.
    - device (str): Device to perform computations on ('cpu' or 'cuda').

    Returns:
    - torch.Tensor: A 2D tensor of Perlin noise values in the range [-1, 1].
    """

    # Define the number of grid points along x and y axes
    grid_x = freq_x + 1
    grid_y = freq_y + 1

    # Generate random gradient vectors at each grid point
    theta = torch.rand(grid_x, grid_y, device=device) * 2 * np.pi
    gradients = torch.stack((torch.cos(theta), torch.sin(theta)), dim=2)

    # Create coordinate grids
    x = torch.linspace(0, freq_x, width, device=device).repeat(height, 1)
    y = torch.linspace(0, freq_y, height, device=device).unsqueeze(1).repeat(1, width)

    # Compute grid indices and fractional offsets
    xi = x.floor().long()
    yi = y.floor().long()
    xf = x - xi
    yf = y - yi

    xi = xi % freq_x
    yi = yi % freq_y

    # Gather gradient vectors for the four corners
    g00 = gradients[xi,     yi]
    g10 = gradients[(xi + 1) % freq_x, yi]
    g01 = gradients[xi,     (yi + 1) % freq_y]
    g11 = gradients[(xi + 1) % freq_x, (yi + 1) % freq_y]

    # Compute the dot products between gradients and distance vectors
    dot00 = (g00[..., 0] * xf + g00[..., 1] * yf)
    dot10 = (g10[..., 0] * (xf - 1) + g10[..., 1] * yf)
    dot01 = (g01[..., 0] * xf + g01[..., 1] * (yf - 1))
    dot11 = (g11[..., 0] * (xf - 1) + g11[..., 1] * (yf - 1))

    # Compute fade curves for smooth interpolation
    u = fade_torch(xf)
    v = fade_torch(yf)

    # Perform linear interpolation
    nx0 = lerp(dot00, dot10, u)
    nx1 = lerp(dot01, dot11, u)
    nxy = lerp(nx0, nx1, v)

    return nxy

def fade_torch(t):
    """Perlin's fade function for smooth interpolation."""
    return t * t * t * (t * (t * 6 - 15) + 10)

def lerp(a, b, t):
    """Linear interpolation between a and b."""
    return a + t * (b - a)


def initialize_network_with_perlin(model, N=8, M=8, K=500, batch_size=256, num_epochs=10, image_size=32):
    """
    Initialize network weights using Perlin noise classification.

    Args:
        model (nn.Module): Neural network model to initialize.
        N (int): Maximum value for n parameter in Perlin noise.
        M (int): Maximum value for m parameter in Perlin noise.
        K (int): Number of instances per category.
        batch_size (int): Batch size for training.
        num_epochs (int): Number of epochs for training.
        image_size (int): Size of the Perlin noise images.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create Perlin noise dataset
    perlin_dataset = ColorPerlinNoiseDataset(N=N, M=M, K=K, image_size=image_size)
    perlin_loader = DataLoader(perlin_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Define loss function and optimizer
    num_classes = N * M * 4
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, epochs=num_epochs, steps_per_epoch=len(perlin_loader))

    # Enable mixed-precision training if possible
    scaler = torch.amp.GradScaler() if device.type == 'cuda' else None

    model = model.to(device)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(perlin_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(enabled=(scaler is not None), device_type='cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            scheduler.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar.set_postfix(loss=running_loss / (total / batch_size), accuracy=100. * correct / total)

        epoch_loss = running_loss / len(perlin_loader)
        accuracy = 100. * correct / total
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')

    return model

def train_cifar10(model, batch_size=128, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epoch_losses = []
    epoch_accuracies = []

    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.2010]),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    model = model.to(device)
    model.train()
    import matplotlib.pyplot as plt
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(trainloader, desc=f'Training Epoch {epoch+1}/{num_epochs}', leave=False)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar.set_postfix(loss=running_loss / (total / batch_size), accuracy=100. * correct / total)

        scheduler.step()
        accuracy = 100. * correct / total
        epoch_loss = running_loss / len(trainloader)

        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(accuracy)

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(trainloader):.3f}, Accuracy: {accuracy:.2f}%')
        # Plot training metrics
    plt.figure(figsize=(10, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), epoch_accuracies, 'b-', label='Accuracy')
    plt.title('Training Accuracy vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), epoch_losses, 'r-', label='Loss')
    plt.title('Training Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

def test_cifar10(model, batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.2010]),
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True)

    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(testloader, desc='Testing', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    print(f'Test Loss: {running_loss / len(testloader):.3f}, Accuracy: {accuracy:.2f}%')
    return accuracy


def train_dtd(model, data_dir='./dtd/images', batch_size=128, num_epochs=10, image_size=224):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # DTD-specific transforms
    transform = transforms.Compose([
        transforms.Resize(image_size),  
        transforms.CenterCrop(image_size),  
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create DTD dataset
    trainset = torchvision.datasets.DTD(root='./data', 
                                      split='train',
                                      download=True, 
                                      transform=transform)
    
    trainloader = DataLoader(trainset, 
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=4,
                           pin_memory=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                         lr=0.00784,  
                         momentum=0.82239,
                         weight_decay=0.00015486)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    model = model.to(device)
    model.train()
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(trainloader, 
                          desc=f'Training Epoch {epoch+1}/{num_epochs}',
                          leave=False)
        
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix(
                loss=running_loss / (total / batch_size),
                accuracy=100. * correct / total
            )
        
        # Step the scheduler
        scheduler.step()
        
        # Calculate epoch statistics
        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100. * correct / total
        
        # Print epoch results
        print(f'Epoch {epoch + 1}/{num_epochs}: Loss: {epoch_loss:.3f} Accuracy: {epoch_acc:.2f}%')
        
            
    print(f'Training completed. Best accuracy: {best_acc:.2f}%')
    return model


def test_dtd(model, data_dir='./dtd/images', batch_size=128, image_size=224):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # DTD-specific transforms
    transform = transforms.Compose([
        transforms.Resize(image_size),  
        transforms.CenterCrop(image_size),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create DTD dataset
    testset = torchvision.datasets.DTD(root='./data', 
                                     split='test',
                                     download=True, 
                                     transform=transform)
    
    testloader = DataLoader(testset, 
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=4,
                          pin_memory=True)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(testloader, desc='Testing', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    print(f'Test Loss: {running_loss / len(testloader):.3f}, Accuracy: {accuracy:.2f}%')
    return accuracy


def train_cifar100(model, batch_size=128, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.2010]),
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0168005, momentum=0.814941, weight_decay=0.00012611989)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    model = model.to(device)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(trainloader, desc=f'Training Epoch {epoch+1}/{num_epochs}', leave=False)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar.set_postfix(loss=running_loss / (total / batch_size), accuracy=100. * correct / total)

        scheduler.step()
        accuracy = 100. * correct / total
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(trainloader):.3f}, Accuracy: {accuracy:.2f}%')


def test_cifar100(model, batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.2010]),
    ])

    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True)

    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(testloader, desc='Testing', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    print(f'Test Loss: {running_loss / len(testloader):.3f}, Accuracy: {accuracy:.2f}%')
    return accuracy



def rgb_noise_label_generator(n, m, c, N, M):
    """
    Generate label incorporating color complexity.
    Label = (n-1)*M*4 + (m-1)*4 + (c-1)
    """
    return (n - 1) * M * 4 + (m - 1) * 4 + (c - 1)

def generate_rgb_perlin(width, height, n, m, color_level, device='cpu'):
    """
    Generate RGB Perlin noise with specified color complexity level.
    
    Parameters:
    - width, height: Image dimensions
    - n, m: Grid parameters
    - color_level: 1 to 4, controlling RGB relationship
    - device: 'cpu' or 'cuda'
    """
    # Generate base Perlin noise
    base_noise = perlin_noise_2d(width, height, n, m, device)
    
    # Different color complexity levels
    if color_level == 1:
        # Level 1: Grayscale-like (all channels highly correlated)
        noise = base_noise.unsqueeze(0).repeat(3, 1, 1)
    
    elif color_level == 2:
        # Level 2: Slight color variation
        r = base_noise
        g = perlin_noise_2d(width, height, n, m, device) * 0.3 + base_noise * 0.7
        b = perlin_noise_2d(width, height, n, m, device) * 0.3 + base_noise * 0.7
        noise = torch.stack([r, g, b])
    
    elif color_level == 3:
        # Level 3: Moderate color independence
        r = base_noise  
        g = perlin_noise_2d(width, height, n, m, device) * 0.6 + base_noise * 0.4
        b = perlin_noise_2d(width, height, n, m, device) * 0.6 + base_noise * 0.4
        noise = torch.stack([r, g, b])
    
    else:  # level 4
        # Level 4: Fully independent RGB channels
        r = base_noise
        g = perlin_noise_2d(width, height, n, m, device)
        b = perlin_noise_2d(width, height, n, m, device)
        noise = torch.stack([r, g, b])
    
    # Normalize to [0, 1]
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    return noise



class ColorPerlinNoiseDataset(Dataset):
    def __init__(self, N, M, K, image_size=32):
        """
        Initialize the Color Perlin Noise Dataset.
        
        Args:
            N (int): Maximum value for n parameter
            M (int): Maximum value for m parameter
            K (int): Number of instances per category
            image_size (int): Size of the output images
        """
        self.N = N
        self.M = M
        self.K = K
        self.image_size = image_size
        self.total_samples = N * M * 4 * K  # 4 color complexity levels
        self.num_categories = N * M * 4
        self.data = None
        self.labels = None
        
        self._generate_dataset()
    
    def _generate_dataset(self):
        """Generate the complete dataset of colored Perlin noise samples."""
        save_path = f'color_perlin_dataset_N{self.N}_M{self.M}_K{self.K}.pt'
        
        if os.path.exists(save_path):
            print(f"Loading pre-generated dataset from {save_path}")
            saved_data = torch.load(save_path)
            self.data = saved_data['data']
            self.labels = saved_data['labels']
        else:
            print(f"Generating new dataset with image size {self.image_size}...")
            self.data = torch.zeros((self.total_samples, 3, self.image_size, self.image_size), 
                                  dtype=torch.float32)
            self.labels = torch.zeros((self.total_samples,), dtype=torch.long)
            
            idx = 0
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            for n in range(1, self.N + 1):
                for m in range(1, self.M + 1):
                    for c in range(1, 5):  
                        label = rgb_noise_label_generator(n, m, c, self.N, self.M)
                        print(f"Generating samples for n={n}, m={m}, color_level={c}, label={label}")
                        
                        for _ in range(self.K):
                            noise = generate_rgb_perlin(
                                self.image_size, self.image_size, 
                                n, m, c, device
                            )
                            self.data[idx] = noise.cpu()
                            self.labels[idx] = label
                            idx += 1
                            
            print(f"Saving dataset to {save_path}")
            torch.save({'data': self.data, 'labels': self.labels}, save_path)
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]



def main():
    # Set seed for reproducibility
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # Class counts for different datasets
    cifar10_class_count = 10
    dtd_class_count = 47
    cifar100_class_count = 100
    omniglot_class_count = 1623

    # Parameters for Perlin noise intialization
    N = 8
    M = 8
    K = 500
    num_epochs = 5
    num_classes = N * M * 4

    # Model 1: Perlin initialization
    model_perlin = torchvision.models.resnet50(num_classes=num_classes)
    model_perlin.to('cuda')
    
    # Initialize with Perlin noise
    model_perlin = initialize_network_with_perlin(model_perlin, N=N, M=M, K=K, num_epochs=num_epochs, image_size=32)
    
    # These models saved to saved_perlin_models folder
    # torch.save(model_perlin.state_dict(), 'color_perlin_init_8_8_500.pth')
    # model_perlin.load_state_dict(torch.load('perlin_icolor_perlin_init_8_8_500nit.pth'))

    # Model 2: Random initialization

    # Save initial Perlin-initialized model state for different dataset trainings
    model_perlin_cifar = copy.deepcopy(model_perlin)
    model_perlin_dtd = copy.deepcopy(model_perlin)
    model_perlin_cifar10 = copy.deepcopy(model_perlin)

    # Change to CIFAR-100 classes
    model_perlin_cifar.fc = nn.Linear(model_perlin_cifar.fc.in_features, cifar100_class_count)
    model_perlin_cifar.to('cuda')

    # Change to DTD classes
    model_perlin_dtd.fc = nn.Linear(model_perlin_dtd.fc.in_features, dtd_class_count)
    model_perlin_dtd.to('cuda')

    # Change to CIFAR-10 classes
    model_perlin_cifar10.fc = nn.Linear(model_perlin_cifar10.fc.in_features, cifar10_class_count)
    model_perlin_cifar10.to('cuda')
    
    # To control base method before original paper
    # model_he = torchvision.models.resnet50(num_classes=cifar100_class_count)  
    # model_he.to('cuda')
    
    print("Training on CIFAR-100")
    train_cifar100(model_perlin_cifar, num_epochs=num_epochs)
    test_cifar100(model_perlin_cifar)

    print("Training on DTD")
    train_dtd(model_perlin_dtd, num_epochs=num_epochs)
    test_dtd(model_perlin_dtd)

    print("Training on CIFAR-10")
    train_cifar10(model_perlin_cifar10, num_epochs=num_epochs)
    test_cifar10(model_perlin_cifar10)


if __name__ == "__main__":
    main()