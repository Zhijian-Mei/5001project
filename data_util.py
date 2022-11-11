import torchvision
from torch.utils.data import DataLoader


def get_data(image_size,data_path,batch_size):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_size),
        torchvision.transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # dataset =torchvision.datasets.CIFAR10(root='data',transform=transforms,download=True)
    dataset = torchvision.datasets.ImageFolder(data_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader