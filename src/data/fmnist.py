import ssl
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class FMNISTDataset:
    def __init__(self, data_dir='./data', num_classes=10):
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

    def load_data(self):
        # Fix SSL certificate issue on macOS
        ssl._create_default_https_context = ssl._create_unverified_context

        trainset = torchvision.datasets.FashionMNIST(
            root=self.data_dir, train=True, download=True, transform=self.transform_train
        )

        testset = torchvision.datasets.FashionMNIST(
            root=self.data_dir, train=False, download=True, transform=self.transform_test
        )

        return trainset, testset

    def get_dataloader(self, dataset, batch_size=32, shuffle=True):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)