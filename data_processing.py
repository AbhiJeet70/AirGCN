import torch
from torch_geometric.datasets import Airports
from torch_geometric.transforms import NormalizeFeatures

# Function to load Airports data for a given country
def load_airports_data(country):
    dataset = Airports(root='/tmp/Airports', name=country, transform=NormalizeFeatures())
    data = dataset[0]
    return data

# Function to split data into train, validation, and test sets
def split_indices(num_nodes, train_ratio=0.7, val_ratio=0.1):
    indices = np.random.permutation(num_nodes)
    train_end = int(train_ratio * num_nodes)
    val_end = int((train_ratio + val_ratio) * num_nodes)
    train_idx = torch.tensor(indices[:train_end], dtype=torch.long)
    val_idx = torch.tensor(indices[train_end:val_end], dtype=torch.long)
    test_idx = torch.tensor(indices[val_end:], dtype=torch.long)
    return train_idx, val_idx, test_idx
