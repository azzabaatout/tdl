"""
Generate Figure 4: Evaluation of global model's difference with n=100, m=20, and c=5.

"""

import torch
import yaml
import copy
import random
import numpy as np
import sys
import os
import json
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from src.data.cifar import CIFARDataset
from src.data.fmnist import FMNISTDataset

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.data import DataPartitioner
from src.data.mnist import MNISTDataset
from src.models.lenet import LeNet
from src.fl.server import FLServer
from src.fl.client import FLClient, MaliciousClient

dataset_hardcoded_lowcaps = "cifar"   # < alternatively change this 'mnist'
dataset_hardcoded_uppercaps = "cifar-10-batches-py" # folder name for the data set. alternatively change this to "MNIST"


class Figure4Experiment:
    """
    Experiment class to generate Figure 4 data.

    Tracks the difference between poisoned and benign global models across rounds.
    """

    def __init__(self, defense_type, dataset_type='mnist', num_rounds=50,
                 num_clients=100, num_malicious=20, classes_per_client=5):
        """
        Initialize experiment.

        Args:
            defense_type: Type of defense ('krum', 'norm_clipping', 'fltrust', 'spp')
            dataset_type: Dataset to use ('mnist')
            num_rounds: Number of FL rounds (default 50 as in Figure 4)
            num_clients: Total number of clients n (default 100)
            num_malicious: Number of malicious clients m (default 20)
            classes_per_client: Non-IID parameter c (default 5)
        """
        self.defense_type = defense_type
        self.dataset_type = dataset_type
        self.num_rounds = num_rounds
        self.num_clients = num_clients
        self.num_malicious = num_malicious
        self.classes_per_client = classes_per_client

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Results storage
        self.differences = []  # Difference per round
        self.round_accuracies = []  # Global model accuracy per round

        # Set random seed for reproducibility
        self.set_random_seeds(42)

    def set_random_seeds(self, seed):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def setup(self):
        """Setup experiment components."""
        print(f"Setting up experiment: {self.defense_type} defense on {self.dataset_type}")

        # Setup dataset
        if self.dataset_type == 'mnist':
            self.dataset = MNISTDataset(data_dir='./data', num_classes=10)
        elif self.dataset_type == 'fmnist':
            self.dataset = FMNISTDataset(data_dir='./data', num_classes=10)
        elif self.dataset_type == 'cifar':
            self.dataset = CIFARDataset(data_dir='./data', num_classes=10)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_type}")

        self.train_dataset, self.test_dataset = self.dataset.load_data()

        # Partition data (non-IID with c classes per client)
        partitioner = DataPartitioner(
            dataset=self.train_dataset,
            num_clients=self.num_clients,
            partition_type='non_iid',
            alpha=self.classes_per_client
        )
        self.client_datasets = partitioner.partition_data()

        # Setup model
        self.model = LeNet(num_classes=10, in_channels=1, input_size=28)
        self.model.to(self.device)

        # Setup server with defense
        defense_params = self._get_defense_params()
        self.server = FLServer(
            model=copy.deepcopy(self.model),
            defense_method=self.defense_type,
            defense_params=defense_params
        )

        # Setup server data for defenses that need it
        if self.defense_type in ['fltrust', 'spp']:
            server_data_size = 100
            server_indices = list(range(min(server_data_size, len(self.test_dataset))))
            from torch.utils.data import Subset
            server_data = Subset(self.test_dataset, server_indices)
            self.server.set_server_data(server_data)

        # Setup clients
        self.clients = []
        malicious_indices = random.sample(range(self.num_clients), self.num_malicious)

        for i in range(self.num_clients):
            is_malicious = i in malicious_indices

            if is_malicious:
                client = MaliciousClient(
                    client_id=i,
                    model=copy.deepcopy(self.model),
                    train_data=self.client_datasets[i],
                    learning_rate=0.01,
                    local_epochs=2,
                    batch_size=32,
                    attack_method='faker',
                    attack_params={
                        'target_defense': self.defense_type,
                        'num_groups': 10,
                        'cooperative': True
                    }
                )
            else:
                client = FLClient(
                    client_id=i,
                    model=copy.deepcopy(self.model),
                    train_data=self.client_datasets[i],
                    learning_rate=0.01,
                    local_epochs=2,
                    batch_size=32
                )

            self.clients.append(client)

        print(f"Setup complete: {self.num_clients} clients ({self.num_malicious} malicious)")

    def _get_defense_params(self):
        """Get defense-specific parameters."""
        params = {}

        if self.defense_type == 'krum':
            params = {'num_malicious': self.num_malicious}
        elif self.defense_type == 'norm_clipping':
            params = {'max_norm': 10.0}
        elif self.defense_type == 'fltrust':
            params = {'clean_data_size': 100}
        elif self.defense_type == 'spp':
            params = {'partial_ratio': 0.5, 'threshold': 0.9}

        return params

    def _flatten_model(self, model):
        """Flatten model parameters into a 1D tensor."""
        params = []
        for param in model.parameters():
            params.append(param.data.view(-1))
        return torch.cat(params)

    def _calculate_model_difference(self, model1, model2):
        """
        Calculate the difference between two models.

        Following the paper: "We randomly select two parameters from both poisoned
        and unpoisoned global models in each round to measure the difference."

        We compute the average absolute difference across randomly selected parameters.
        """
        flat1 = self._flatten_model(model1).cpu()
        flat2 = self._flatten_model(model2).cpu()

        # Randomly select parameters (paper mentions "two parameters" but we use more for stability)
        num_params = len(flat1)
        num_samples = min(100, num_params)  # Sample 100 parameters
        indices = random.sample(range(num_params), num_samples)

        # Calculate absolute difference for selected parameters
        diff = torch.abs(flat1[indices] - flat2[indices])
        return diff.mean().item()

    def _fedavg_aggregate(self, client_updates, weights=None):
        """Simple FedAvg aggregation without defense."""
        if not client_updates:
            return None

        if weights is None:
            weights = [1.0 / len(client_updates)] * len(client_updates)
        else:
            total = sum(weights)
            weights = [w / total for w in weights]

        # Initialize aggregated state dict
        aggregated_state = {}
        first_state = client_updates[0]['model_state']

        for key in first_state:
            aggregated_state[key] = torch.zeros_like(first_state[key], dtype=torch.float32)

        # Weighted average
        for update, weight in zip(client_updates, weights):
            for key in aggregated_state:
                aggregated_state[key] += weight * update['model_state'][key].float()

        return aggregated_state

    def run_round(self, round_num):
        """Run a single FL round and compute model difference."""
        global_state_dict = self.server.broadcast_model()

        client_updates = []
        benign_updates = []

        for client in self.clients:
            client.update_model(global_state_dict)
            client.local_train()

            update = {
                'client_id': client.client_id,
                'model_state': client.get_model_parameters(),
                'data_size': client.get_data_size(),
                'is_malicious': getattr(client, 'is_malicious', False)
            }
            client_updates.append(update)

            # Collect benign updates separately
            if not getattr(client, 'is_malicious', False):
                benign_updates.append(update)

        # agg 1: All clients (poisoned global model)
        self.server.receive_updates(client_updates)
        self.server.aggregate_updates()
        poisoned_global_model = copy.deepcopy(self.server.global_model)

        benign_global_model = copy.deepcopy(self.model)
        benign_global_model.load_state_dict(global_state_dict)

        if benign_updates:
            benign_state = self._fedavg_aggregate(benign_updates)
            if benign_state:
                benign_global_model.load_state_dict(benign_state)

        difference = self._calculate_model_difference(poisoned_global_model, benign_global_model)
        self.differences.append(difference)

        loss, accuracy = self.server.evaluate_model(self.test_dataset)
        self.round_accuracies.append(accuracy)

        return difference, accuracy

    def run(self):
        """Run the complete experiment."""
        self.setup()

        print(f"\nRunning {self.num_rounds} rounds...")
        for round_num in range(1, self.num_rounds + 1):
            diff, acc = self.run_round(round_num)

            if round_num % 10 == 0:
                print(f"Round {round_num}: Difference = {diff:.4f}, Accuracy = {acc:.2f}%")

        return self.differences

    def get_results(self):
        """Get experiment results."""
        return {
            'defense_type': self.defense_type,
            'dataset': self.dataset_type,
            'differences': self.differences,
            'accuracies': self.round_accuracies,
            'config': {
                'num_rounds': self.num_rounds,
                'num_clients': self.num_clients,
                'num_malicious': self.num_malicious,
                'classes_per_client': self.classes_per_client
            }
        }


def run_all_defenses(dataset_type='mnist', num_rounds=50):
    """
    Run experiments for all defenses and collect results.

    Args:
        dataset_type: 'mnist'
        num_rounds: Number of FL rounds

    Returns:
        Dictionary with results for each defense
    """
    # Available defenses in src/defenses/
    defenses = {
        'NC': 'norm_clipping',
        'FT': 'fltrust',
        'KM': 'krum',
        'SPP': 'spp',
    }

    results = {}

    for label, defense_type in defenses.items():
        print(f"\n{'='*60}")
        print(f"Running experiment: {label} ({defense_type})")
        print(f"{'='*60}")

        try:
            experiment = Figure4Experiment(
                defense_type=defense_type,
                dataset_type=dataset_type,
                num_rounds=num_rounds,
                num_clients=100,
                num_malicious=20,
                classes_per_client=5
            )
            experiment.run()
            results[label] = experiment.get_results()

        except Exception as e:
            print(f"Error running {label}: {e}")
            import traceback
            traceback.print_exc()
            results[label] = {'error': str(e), 'differences': [0] * num_rounds}

    return results


def plot_figure4(results, dataset_name='MNIST', save_path=None):
    """
    Generate Figure 4 plot.

    Args:
        results: Dictionary with results for each defense
        dataset_name: Name of dataset for title
        save_path: Path to save the figure (optional)
    """
    plt.figure(figsize=(8, 6))

    # Colors and markers for each defense
    styles = {
        'NC': {'color': 'blue', 'marker': 'o', 'linestyle': '-'},
        'FT': {'color': 'green', 'marker': '^', 'linestyle': '-'},
        'KM': {'color': 'purple', 'marker': 'D', 'linestyle': '-'},
        'SPP': {'color': 'red', 'marker': 's', 'linestyle': '-'},
    }

    for label, result in results.items():
        if 'error' in result:
            continue

        differences = result['differences']
        rounds = list(range(len(differences)))

        style = styles.get(label, {'color': 'gray', 'marker': 'o', 'linestyle': '-'})

        plt.plot(rounds, differences,
                 label=label,
                 color=style['color'],
                 marker=style['marker'],
                 markersize=4,
                 markevery=5,  # Show marker every 5 rounds
                 linestyle=style['linestyle'],
                 linewidth=1.5)

    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Difference', fontsize=12)
    plt.title(f'({dataset_name.lower()[0]}) {dataset_name}.', fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()


def generate_figure4(datasets=None, num_rounds=50, output_dir='figure4_results'):
    """
    Generate complete Figure 4.

    Args:
        datasets: List of datasets to run ('mnist')
        num_rounds: Number of FL rounds
        output_dir: Directory to save results
    """
    if datasets is None:
        datasets = [dataset_hardcoded_lowcaps]  # Start with MNIST, add others as needed

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    all_results = {}

    for dataset in datasets:
        print(f"\n{'#'*70}")
        print(f"# Running experiments for {dataset.upper()}")
        print(f"{'#'*70}")

        results = run_all_defenses(dataset_type=dataset, num_rounds=num_rounds)
        all_results[dataset] = results

        # Save individual results
        result_file = output_path / f'figure4_{dataset}_{timestamp}.json'
        with open(result_file, 'w') as f:
            # Convert to serializable format
            serializable = {}
            for label, res in results.items():
                serializable[label] = {
                    'defense_type': res.get('defense_type', ''),
                    'dataset': res.get('dataset', ''),
                    'differences': res.get('differences', []),
                    'accuracies': res.get('accuracies', []),
                    'config': res.get('config', {}),
                    'error': res.get('error', None)
                }
            json.dump(serializable, f, indent=2)
        print(f"Results saved to: {result_file}")

        # Generate plot for this dataset
        dataset_names = {dataset_hardcoded_lowcaps: dataset_hardcoded_uppercaps}
        plot_path = output_path / f'figure4_{dataset}_{timestamp}.png'
        plot_figure4(results, dataset_name=dataset_names.get(dataset, dataset.upper()),
                    save_path=str(plot_path))

    # Generate combined figure if multiple datasets
    if len(datasets) > 1:
        fig, axes = plt.subplots(1, len(datasets), figsize=(5*len(datasets), 5))

        if len(datasets) == 1:
            axes = [axes]

        styles = {
            'NC': {'color': 'blue', 'marker': 'o'},
            'FT': {'color': 'green', 'marker': '^'},
            'KM': {'color': 'purple', 'marker': 'D'},
            'SPP': {'color': 'red', 'marker': 's'},
        }

        dataset_names = {dataset_hardcoded_lowcaps: dataset_hardcoded_uppercaps}

        for ax, dataset in zip(axes, datasets):
            results = all_results[dataset]

            for label, result in results.items():
                if 'error' in result:
                    continue

                differences = result['differences']
                rounds = list(range(len(differences)))
                style = styles.get(label, {'color': 'gray', 'marker': 'o'})

                ax.plot(rounds, differences, label=label,
                       color=style['color'], marker=style['marker'],
                       markersize=4, markevery=5, linewidth=1.5)

            ax.set_xlabel('Round', fontsize=11)
            ax.set_ylabel('Difference', fontsize=11)
            ax.set_title(f'({chr(97+datasets.index(dataset))}) {dataset_names.get(dataset, dataset.upper())}.',
                        fontsize=11)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        combined_path = output_path / f'figure4_combined_{timestamp}.png'
        plt.savefig(combined_path, dpi=300, bbox_inches='tight')
        print(f"Combined figure saved to: {combined_path}")
        plt.show()

    print(f"\n{'='*70}")
    print("Figure 4 generation complete!")
    print(f"Results saved in: {output_path}")
    print(f"{'='*70}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Figure 4 from the paper")
    parser.add_argument("--datasets", nargs='+', default=[dataset_hardcoded_lowcaps],
                       choices=[dataset_hardcoded_lowcaps],
                       help="Datasets to run experiments on")
    parser.add_argument("--rounds", type=int, default=50,
                       help="Number of FL rounds (default: 50)")
    parser.add_argument("--output", type=str, default='figure4_results',
                       help="Output directory for results")

    args = parser.parse_args()

    generate_figure4(
        datasets=args.datasets,
        num_rounds=args.rounds,
        output_dir=args.output
    )