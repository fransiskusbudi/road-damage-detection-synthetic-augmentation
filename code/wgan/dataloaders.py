from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

def get_mnist_dataloaders(batch_size=128):
    """MNIST dataloader with (32, 32) sized images."""
    # Resize images so they are a power of 2
    all_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    # Get train and test data
    train_data = datasets.MNIST('data', train=True, download=True,
                                transform=all_transforms)
    test_data = datasets.MNIST('data', train=False,
                               transform=all_transforms)
    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def get_fashion_mnist_dataloaders(batch_size=128):
    """Fashion MNIST dataloader with (32, 32) sized images."""
    # Resize images so they are a power of 2
    all_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    # Get train and test data
    train_data = datasets.FashionMNIST('../fashion_data', train=True, download=True,
                                       transform=all_transforms)
    test_data = datasets.FashionMNIST('../fashion_data', train=False,
                                      transform=all_transforms)
    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def get_lsun_dataloader(path_to_data='../lsun', dataset='bedroom_train',
                        batch_size=64):
    """LSUN dataloader with (128, 128) sized images.

    path_to_data : str
        One of 'bedroom_val' or 'bedroom_train'
    """
    # Compose transforms
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor()
    ])

    # Get dataset
    lsun_dset = datasets.LSUN(db_path=path_to_data, classes=[dataset],
                              transform=transform)

    # Create dataloader
    return DataLoader(lsun_dset, batch_size=batch_size, shuffle=True)



class PotholeDataset(Dataset):
    """Custom dataset for loading pothole images from a directory."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory with all the pothole images.
            transform (callable, optional): Transform to apply to images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = [f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


def get_pothole_dataloader(path_to_data='cropped_400', batch_size=64):
    """Pothole dataloader with (128, 128) sized images."""
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor()
        ,
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1,1] for Tanh activation
    ])

    dataset = PotholeDataset(root_dir=path_to_data, transform=transform)
    print(f"Total images: {len(dataset)}")
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

