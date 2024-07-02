import torch
from data_processing import load_airports_data, split_indices
from model import GCNNet
from train import train_model
from utils import set_seed, print_dataset_statistics

# Hyperparameter grid search
hidden_channels_list = [64, 128, 256]
learning_rates = [0.005, 0.001, 0.0005]
weight_decays = [1e-4, 1e-5]

# List of countries to process
countries = ['USA', 'Brazil', 'Europe']

def main():
    set_seed(20)

    # Process each country and print accuracies
    for country in countries:
        print(f'Processing country: {country}')
        data = load_airports_data(country)
        
        # Print dataset statistics
        print_dataset_statistics(data, country)
        
        # Prepare the masks
        train_idx, val_idx, test_idx = split_indices(data.num_nodes)
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.train_mask[train_idx] = True
        data.val_mask[val_idx] = True
        data.test_mask[test_idx] = True
        
        best_acc = 0
        best_params = None
        
        models = [GCNNet(data.num_node_features, hidden_channels, data.y.max().item() + 1) for hidden_channels in hidden_channels_list]
        for model in models:
            for lr in learning_rates:
                for weight_decay in weight_decays:
                    print(f'Training with {model.__class__.__name__}, lr={lr}, weight_decay={weight_decay}')
                    acc = train_model(model, data, lr, weight_decay)
                    if acc > best_acc:
                        best_acc = acc
                        best_params = (model.__class__.__name__, model, lr, weight_decay)
        
        print(f'Best accuracy for {country}: {best_acc:.4f} with params {best_params}')

if __name__ == '__main__':
    main()
