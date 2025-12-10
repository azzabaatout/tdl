import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy


class FLClient:
    def __init__(self, client_id, model, train_data, test_data=None, 
                 learning_rate=0.01, local_epochs=1, batch_size=32):
        self.client_id = client_id
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        if test_data:
            self.test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        
    def update_model(self, global_state_dict):
        """update local model with global parameters"""
        self.model.load_state_dict(global_state_dict)
    
    def local_train(self):
        """perform local training for specified epochs"""
        self.model.to(self.device)
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(self.local_epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def local_test(self):
        """eval local model on test data"""
        if not self.test_data:
            return 0.0, 0.0
        
        self.model.to(self.device)
        self.model.eval()
        
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                test_loss += self.criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_loss = test_loss / len(self.test_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def get_model_parameters(self):
        """get current model parameters"""
        return copy.deepcopy(self.model.state_dict())
    
    def get_data_size(self):
        """get size of training data"""
        return len(self.train_data)


class MaliciousClient(FLClient):
    def __init__(self, client_id, model, train_data, test_data=None,
                 learning_rate=0.01, local_epochs=1, batch_size=32,
                 attack_method='none', attack_params=None):
        super().__init__(client_id, model, train_data, test_data,
                        learning_rate, local_epochs, batch_size)
        self.attack_method = attack_method
        self.attack_params = attack_params or {}
        self.is_malicious = True
    
    def local_train(self):
        """Train locally and then apply attack"""
        # perform normal training
        train_loss = super().local_train()
        
        # apply attack to the model
        if self.attack_method != 'none':
            self.apply_attack()
        
        return train_loss
    
    def apply_attack(self):
        """Apply the specified attack to the local model"""
        if self.attack_method == 'la':
            self._local_attack()
        elif self.attack_method == 'mb':
            self._model_replacement_attack()
        elif self.attack_method == 'faker':
            self._faker_attack()
        else:
            pass  # No attack
    
    def _local_attack(self):
        """Local Attack
        change param directions"""
        scaling_factor = self.attack_params.get('scaling_factor', 10.0)
        
        for param in self.model.parameters():
            # gen random signs
            random_signs = torch.randint(0, 2, param.shape, device=param.device) * 2 - 1
            # scale and flip signs
            param.data = param.data * random_signs * scaling_factor
    
    def _model_replacement_attack(self):
        """Model Replacement Attack
        Replace with malicious model"""
        scaling_factor = self.attack_params.get('scaling_factor', -10.0)
        
        for param in self.model.parameters():
            param.data = param.data * scaling_factor
    
    def _faker_attack(self):
        """ Faker attack
        Sophisticated similarity-based attack."""
        # implemented a simple version
        noise_scale = self.attack_params.get('noise_scale', 0.1)
        
        for param in self.model.parameters():
            noise = torch.randn_like(param) * noise_scale
            param.data = param.data + noise