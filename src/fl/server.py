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
        self.server_data = None  # for defenses that require server data
        
    def set_server_data(self, server_data):
        """set server data for defenses like FLTrust"""
        self.server_data = server_data
        
    def broadcast_model(self):
        """broadcasst global model to clients."""
        return copy.deepcopy(self.global_model.state_dict())
    
    def receive_updates(self, client_updates):
        """receive updates from clients"""
        self.client_updates = client_updates
    
    def aggregate_updates(self):
        """agg client updates to form new global model"""
        if not self.client_updates:
            return
        
        # we extract models and weights from client updates
        client_models = []
        client_weights = []
        
        for update in self.client_updates:
            client_model = copy.deepcopy(self.global_model)
            client_model.load_state_dict(update['model_state'])
            client_models.append(client_model)
            client_weights.append(update.get('data_size', 1))
        
        # norm weights by total data size
        total_data = sum(client_weights)
        client_weights = [w / total_data for w in client_weights]
        
        # prepare server model for defenses that need it
        server_model = None
        if self.defense_method == 'fltrust' and self.server_data:
            server_model = self._train_server_model()
        
        # we agg using the specified method
        if hasattr(self.aggregator, 'defense_method'):
            global_state_dict = self.aggregator.aggregate(
                client_models, client_weights, server_model
            )
        else:
            global_state_dict = self.aggregator.aggregate(client_models, client_weights)
        
        # we update global model
        self.global_model.load_state_dict(global_state_dict)
        self.round_num += 1
        
        # we clear updates for next round
        self.client_updates = []
    
    def _train_server_model(self):
        """train a clean model on server data (for FLTrust)"""
        if not self.server_data:
            return None
        
        server_model = copy.deepcopy(self.global_model)
        server_model.to(self.device)
        server_model.train()
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(server_model.parameters(), lr=0.01)
        
        server_loader = DataLoader(self.server_data, batch_size=32, shuffle=True)
        
        # train only  for a few epochs
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
        """eval global model on test data"""
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
        # current round number
        return self.round_num