import torch
import yaml
import copy
import random
import numpy as np
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import FL components
from src.data.cifar import CIFARDataset
from src.data.partition import DataPartitioner
from src.models.lenet import LeNet
from src.models.alexnet_small import AlexNetSmall
from src.models.resnet18 import ResNet18
from src.fl.server import FLServer
from src.fl.client import FLClient, MaliciousClient
from src.attacks.la import LocalAttack, AdaptiveLocalAttack
from src.attacks.mb import ManipulateByzantineAttack, AdaptiveMBAttack
from src.attacks.faker import FakerAttack, AdaptiveFakerAttack
from src.defenses.fltrust import FLTrust, EnhancedFLTrust
from src.defenses.krum import Krum, TrimmedMean, AdaptiveKrum
from src.defenses.norm_clipping import NormClipping, GradientClipping, HybridNormDefense
from src.utils.metrics import FLMetrics
from src.utils.logging import FLLogger


class FLExperiment:
    """Main experiment runner for federated learning with attacks and defenses."""
    
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # init random seeds for reproducibility
        self.set_random_seeds(self.config.get('random_seed', 42))
        
        # init components
        self.dataset = None
        self.model = None
        self.server = None
        self.clients = []
        self.metrics = FLMetrics()
        self.logger = FLLogger(self.config['experiment_name'])
        
        self.setup_experiment()
    
    def load_config(self, config_path):
        """Load experiment configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def set_random_seeds(self, seed):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def setup_experiment(self):
        """Setup all experiment components."""
        self.logger.log_experiment_config(self.config)
        self.logger.log_info(f"Using device: {self.device}")
        
        # setup dataset
        self.setup_dataset()
        
        # setup model
        self.setup_model()
        
        # setup server
        self.setup_server()
        
        # setup clients
        self.setup_clients()
        
        self.logger.log_info("Experiment setup completed")
    
    def setup_dataset(self):
        """Setup CIFAR-10 dataset and partitioning."""
        dataset_config = self.config['dataset']
        
        self.dataset = CIFARDataset(
            data_dir=dataset_config.get('data_dir', './data'),
            num_classes=dataset_config.get('num_classes', 10)
        )
        
        # load data
        self.train_dataset, self.test_dataset = self.dataset.load_data()
        
        # partition data among clients
        partitioner = DataPartitioner(
            dataset=self.train_dataset,
            num_clients=self.config['federated_learning']['num_clients'],
            partition_type=dataset_config.get('partition_type', 'iid'),
            alpha=dataset_config.get('alpha', 0.5)
        )
        
        self.client_datasets = partitioner.partition_data()
        self.logger.log_info(f"Data partitioned among {len(self.client_datasets)} clients")
    
    def setup_model(self):
        """Setup the neural network model."""
        model_config = self.config['model']
        model_type = model_config['type'].lower()
        
        if model_type == 'lenet':
            self.model = LeNet(num_classes=model_config.get('num_classes', 10))
        elif model_type == 'alexnet_small':
            self.model = AlexNetSmall(num_classes=model_config.get('num_classes', 10))
        elif model_type == 'resnet18':
            self.model = ResNet18(num_classes=model_config.get('num_classes', 10))
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model.to(self.device)
        self.logger.log_info(f"Model initialized: {model_type}")
    
    def setup_server(self):
        """Setup federated learning server."""
        server_config = self.config.get('server', {})
        defense_config = self.config.get('defense', {})
        
        self.server = FLServer(
            model=copy.deepcopy(self.model),
            defense_method=defense_config.get('type', 'none'),
            defense_params=defense_config.get('params', {})
        )
        
        # set server data for defenses that need it (FLTrust or any other)
        if defense_config.get('type') == 'fltrust':
            server_data_size = server_config.get('data_size', 100)
            # use a small portion of test data as server data
            server_indices = list(range(min(server_data_size, len(self.test_dataset))))
            from torch.utils.data import Subset
            server_data = Subset(self.test_dataset, server_indices)
            self.server.set_server_data(server_data)
        
        self.logger.log_info(f"Server initialized with defense: {defense_config.get('type', 'none')}")
    
    def setup_clients(self):
        """Setup federated learning clients."""
        fl_config = self.config['federated_learning']
        attack_config = self.config.get('attack', {})
        
        num_clients = fl_config['num_clients']
        num_malicious = attack_config.get('num_malicious_clients', 0)
        
        # find malicious clients
        if num_malicious > 0:
            malicious_indices = random.sample(range(num_clients), num_malicious)
        else:
            malicious_indices = []
        
        # instantiate clients
        for i in range(num_clients):
            is_malicious = i in malicious_indices
            
            if is_malicious:
                client = MaliciousClient(
                    client_id=i,
                    model=copy.deepcopy(self.model),
                    train_data=self.client_datasets[i],
                    learning_rate=fl_config.get('learning_rate', 0.01),
                    local_epochs=fl_config.get('local_epochs', 1),
                    batch_size=fl_config.get('batch_size', 32),
                    attack_method=attack_config.get('type', 'none'),
                    attack_params=attack_config.get('params', {})
                )
            else:
                client = FLClient(
                    client_id=i,
                    model=copy.deepcopy(self.model),
                    train_data=self.client_datasets[i],
                    learning_rate=fl_config.get('learning_rate', 0.01),
                    local_epochs=fl_config.get('local_epochs', 1),
                    batch_size=fl_config.get('batch_size', 32)
                )
            
            self.clients.append(client)
        
        self.logger.log_info(f"Created {num_clients} clients ({num_malicious} malicious)")
    
    def run_experiment(self):
        """Run the complete federated learning experiment."""
        fl_config = self.config['federated_learning']
        num_rounds = fl_config['num_rounds']
        client_participation_rate = fl_config.get('client_participation_rate', 1.0)
        
        self.logger.log_info(f"Starting experiment with {num_rounds} rounds")
        
        for round_num in range(1, num_rounds + 1):
            self.run_federated_round(round_num, client_participation_rate)
        
        self.finalize_experiment()
    
    def run_federated_round(self, round_num, participation_rate=1.0):
        """Run a single round of federated learning."""
        # select participating clients
        num_participants = max(1, int(len(self.clients) * participation_rate))
        participating_clients = random.sample(self.clients, num_participants)
        participating_ids = [client.client_id for client in participating_clients]
        
        self.logger.log_round_start(round_num, participating_ids)
        
        # broadcast global model
        global_state_dict = self.server.broadcast_model()
        
        # client local training
        client_updates = []
        for client in participating_clients:
            # update client model with global parameters
            client.update_model(global_state_dict)
            
            # local training
            local_loss = client.local_train()
            
            # eval client model
            local_test_loss, local_test_accuracy = client.local_test()
            
            # log client metrics
            self.metrics.log_client_metrics(
                client.client_id, round_num, local_test_accuracy, local_test_loss,
                client.get_data_size(), getattr(client, 'is_malicious', False)
            )
            
            self.logger.log_client_training(
                client.client_id, round_num, local_test_accuracy, local_loss,
                client.local_epochs, getattr(client, 'is_malicious', False)
            )
            
            # prepare update for server
            client_update = {
                'client_id': client.client_id,
                'model_state': client.get_model_parameters(),
                'data_size': client.get_data_size(),
                'is_malicious': getattr(client, 'is_malicious', False)
            }
            client_updates.append(client_update)
        
        # log attack information
        malicious_clients = [update for update in client_updates if update['is_malicious']]
        if malicious_clients:
            attacker_ids = [update['client_id'] for update in malicious_clients]
            attack_type = self.config.get('attack', {}).get('type', 'unknown')
            
            # calc similarity metrics for attacks
            if len(malicious_clients) > 0 and len(client_updates) > len(malicious_clients):
                # compare  malicious vs benign models
                benign_updates = [update for update in client_updates if not update['is_malicious']]
                if benign_updates:
                    malicious_model = copy.deepcopy(self.server.global_model)
                    malicious_model.load_state_dict(malicious_clients[0]['model_state'])
                    
                    benign_model = copy.deepcopy(self.server.global_model)
                    benign_model.load_state_dict(benign_updates[0]['model_state'])
                    
                    similarity_metrics = self.metrics.calculate_similarity_metrics(
                        malicious_model, benign_model
                    )
                else:
                    similarity_metrics = {}
            else:
                similarity_metrics = {}
            
            self.metrics.log_attack_metrics(
                round_num, attack_type, True, similarity_metrics, attacker_ids
            )
            
            self.logger.log_attack(
                round_num, attack_type, attacker_ids,
                self.config.get('attack', {}).get('params', {}),
                attack_success=True, similarity_metrics=similarity_metrics
            )
        
        self.server.receive_updates(client_updates)
        self.server.aggregate_updates()
        
        defense_type = self.config.get('defense', {}).get('type', 'none')
        if defense_type != 'none':
            # count rejected/filtered models (simplified)
            rejected_count = len(client_updates) - len([u for u in client_updates if not u['is_malicious']])
            detected_malicious = [u['client_id'] for u in client_updates if u['is_malicious']]
            
            self.metrics.log_defense_metrics(
                round_num, defense_type, rejected_count, detected_malicious
            )
            
            self.logger.log_defense(
                round_num, defense_type,
                self.config.get('defense', {}).get('params', {}),
                rejected_clients=[], detected_malicious=detected_malicious
            )
        
        # eval global model
        global_loss, global_accuracy = self.server.evaluate_model(self.test_dataset)
        
        # log round metrics
        self.metrics.log_round_metrics(
            round_num, global_accuracy, global_loss, global_accuracy, global_loss
        )
        
        self.logger.log_round_end(
            round_num, global_accuracy, global_loss, global_accuracy, global_loss
        )
    
    def finalize_experiment(self):
        """Finalize experiment and save results."""
        self.logger.log_info("Experiment completed")
        
        # save all logs and metrics
        summary = self.logger.save_all_logs()
        
        # gen and save plots
        log_dir = self.logger.get_log_directory()
        self.metrics.plot_training_curves(save_path=log_dir / "training_curves.png")
        self.metrics.plot_defense_analysis(save_path=log_dir / "defense_analysis.png")
        
        # save metrics
        metrics_data = self.metrics.export_to_dict()
        import json
        with open(log_dir / "metrics.json", 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)
        
        # print summary
        convergence_metrics = self.metrics.get_convergence_metrics()
        summary_stats = self.metrics.get_summary_stats()
        
        print("\n" + "="*50)
        print("EXPERIMENT SUMMARY")
        print("="*50)
        print(f"Experiment: {self.config['experiment_name']}")
        print(f"Total Rounds: {summary_stats['total_rounds']}")
        print(f"Final Accuracy: {convergence_metrics.get('final_accuracy', 'N/A'):.4f}")
        print(f"Attack Success Rate: {summary_stats['attack_success_rate']:.4f}")
        print(f"Total Rejected Models: {summary_stats['total_rejected_models']}")
        print(f"Results saved to: {log_dir}")
        print("="*50)
        
        return summary


def run_experiment_from_config(config_path):
    """Convenience function to run experiment from config file."""
    experiment = FLExperiment(config_path)
    return experiment.run_experiment()


def run_multiple_experiments(config_dir, pattern="*.yaml"):
    """Run multiple experiments from a directory of config files."""
    config_path = Path(config_dir)
    config_files = list(config_path.glob(pattern))
    
    results = {}
    
    for config_file in config_files:
        print(f"\nRunning experiment: {config_file.name}")
        try:
            experiment = FLExperiment(config_file)
            summary = experiment.run_experiment()
            results[config_file.name] = summary
            print(f"Completed experiment: {config_file.name}")
        except Exception as e:
            print(f"Failed experiment {config_file.name}: {e}")
            results[config_file.name] = {"error": str(e)}
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Federated Learning Experiments")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to experiment configuration file")
    parser.add_argument("--multiple", action="store_true",
                       help="Run multiple experiments from config directory")
    
    args = parser.parse_args()
    
    if args.multiple:
        results = run_multiple_experiments(args.config)
        print(f"\nCompleted {len(results)} experiments")
    else:
        run_experiment_from_config(args.config)