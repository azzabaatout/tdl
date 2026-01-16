import torch
import torch.nn as nn
import copy
from torch.utils.data import DataLoader
from .aggregator import Aggregator, DefensiveAggregator


class FLServer:
    def __init__(self, model, defense_method='none', defense_params=None):
        self.global_model = model
        self.defense_method = defense_method
        self.defense_params = defense_params or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if defense_method == 'none':
            self.aggregator = Aggregator(aggregation_method='fedavg')
        else:
            self.aggregator = DefensiveAggregator(defense_method=defense_method,
                                                  **defense_params)

        self.round_num = 0
        self.client_updates = []
        self.server_data = None

    def set_server_data(self, server_data):
        """Set server data for defenses like FLTrust."""
        self.server_data = server_data

    def broadcast_model(self):
        """Broadcast global model to clients."""
        return copy.deepcopy(self.global_model.state_dict())

    def receive_updates(self, client_updates):
        """Receive updates from clients."""
        self.client_updates = client_updates

    def aggregate_updates(self):

        if not self.client_updates:
            return {'defense_stats': None}

        # extra models and weights from client updates
        client_models = []
        client_weights = []
        client_ids = []

        for update in self.client_updates:
            client_model = copy.deepcopy(self.global_model)
            client_model.load_state_dict(update['model_state'])
            client_models.append(client_model)
            client_weights.append(update.get('data_size', 1))
            client_ids.append(update['client_id'])

        # norm weights by total data size
        total_data = sum(client_weights)
        client_weights = [w / total_data for w in client_weights]

        # prepare server model for defenses that need it
        server_model = None
        if self.defense_method in ['fltrust', 'spp'] and self.server_data:
            server_model = self._train_server_model()

        # agg using the specified method
        defense_stats = None
        if hasattr(self.aggregator, 'defense_method'):
            result = self.aggregator.aggregate(
                client_models,
                client_weights,
                server_model,
                client_ids,
                current_round=self.round_num,
                global_model=self.global_model
            )
            if isinstance(result, tuple):
                global_state_dict, defense_stats = result
            else:
                global_state_dict = result
                # check if aggregator has defense statistics
                if hasattr(self.aggregator, 'last_defense_stats'):
                    defense_stats = {
                        'defense_type': self.aggregator.defense_method,
                        'defense_params': self.aggregator.defense_params,
                        'rejected_clients': self.aggregator.last_defense_stats.get('rejected_clients', []),
                        'detected_malicious': self.aggregator.last_defense_stats.get('detected_malicious', []),
                        'effectiveness_metrics': None
                    }
        else:
            global_state_dict = self.aggregator.aggregate(client_models, client_weights)

        # update global model
        self.global_model.load_state_dict(global_state_dict)
        self.round_num += 1

        self.client_updates = []

        return {'defense_stats': defense_stats}

    def _train_server_model(self):
        """Train a clean model on server data (for FLTrust)."""
        if not self.server_data:
            return None

        server_model = copy.deepcopy(self.global_model)
        server_model.to(self.device)
        server_model.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(server_model.parameters(), lr=0.01)

        server_loader = DataLoader(self.server_data, batch_size=32, shuffle=True)

        # train for a few epochs
        for epoch in range(5):
            for data, target in server_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = server_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        return server_model

    def evaluate_model(self, test_data):
        """Evaluate global model on test data."""
        self.global_model.to(self.device)
        self.global_model.eval()

        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
        criterion = nn.CrossEntropyLoss()

        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)

                test_loss += criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        avg_loss = test_loss / len(test_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def get_round_number(self):
        """Get current round number."""
        return self.round_num