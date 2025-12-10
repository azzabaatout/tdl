import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class FLMetrics:
    """Comprehensive metrics tracking for federated learning experiments."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.round_metrics = []
        self.client_metrics = {}
        self.attack_metrics = {}
        self.defense_metrics = {}
    
    def log_round_metrics(self, round_num, global_accuracy, global_loss, 
                         test_accuracy=None, test_loss=None):
        # log metrics
        round_data = {
            'round': round_num,
            'global_accuracy': global_accuracy,
            'global_loss': global_loss,
            'test_accuracy': test_accuracy,
            'test_loss': test_loss
        }
        self.round_metrics.append(round_data)
    
    def log_client_metrics(self, client_id, round_num, local_accuracy, local_loss,
                          data_size=None, is_malicious=False):
        # log metrics for individual clients
        if client_id not in self.client_metrics:
            self.client_metrics[client_id] = []
        
        client_data = {
            'round': round_num,
            'local_accuracy': local_accuracy,
            'local_loss': local_loss,
            'data_size': data_size,
            'is_malicious': is_malicious
        }
        self.client_metrics[client_id].append(client_data)
    
    def log_attack_metrics(self, round_num, attack_type, attack_success,
                          similarity_metrics=None, target_clients=None):
        # log attack related metrics
        if round_num not in self.attack_metrics:
            self.attack_metrics[round_num] = []
        
        attack_data = {
            'attack_type': attack_type,
            'attack_success': attack_success,
            'similarity_metrics': similarity_metrics or {},
            'target_clients': target_clients or []
        }
        self.attack_metrics[round_num].append(attack_data)
    
    def log_defense_metrics(self, round_num, defense_type, rejected_count,
                           detected_malicious=None, defense_effectiveness=None):
        if round_num not in self.defense_metrics:
            self.defense_metrics[round_num] = []
        
        defense_data = {
            'defense_type': defense_type,
            'rejected_count': rejected_count,
            'detected_malicious': detected_malicious or [],
            'defense_effectiveness': defense_effectiveness
        }
        self.defense_metrics[round_num].append(defense_data)
    
    def calculate_similarity_metrics(self, model1, model2):
        """Calculate comprehensive similarity metrics between models."""
        flat1 = self._flatten_model(model1)
        flat2 = self._flatten_model(model2)
        
        # l2 norm ratio
        l2_norm1 = torch.norm(flat1, p=2)
        l2_norm2 = torch.norm(flat2, p=2)
        l2_ratio = (l2_norm1 / l2_norm2).item() if l2_norm2 != 0 else float('inf')
        
        # eucl distance
        euclidean_dist = torch.norm(flat1 - flat2, p=2).item()
        
        #  cos sim
        dot_product = torch.dot(flat1, flat2)
        cosine_sim = (dot_product / (l2_norm1 * l2_norm2)).item() if l2_norm1 != 0 and l2_norm2 != 0 else 0
        
        # extra metrics
        # paramwise correlation
        correlation = torch.corrcoef(torch.stack([flat1, flat2]))[0, 1].item()
        if torch.isnan(torch.tensor(correlation)):
            correlation = 0.0
        
        return {
            'l2_ratio': l2_ratio,
            'euclidean_distance': euclidean_dist,
            'cosine_similarity': cosine_sim,
            'correlation': correlation,
            'l2_norm1': l2_norm1.item(),
            'l2_norm2': l2_norm2.item()
        }
    
    def calculate_defense_effectiveness(self, true_malicious, detected_malicious):
        #Calculation of defense effectiveness metrics
        if not true_malicious and not detected_malicious:
            return {
                'precision': 1.0,
                'recall': 1.0,
                'f1_score': 1.0,
                'accuracy': 1.0
            }
        
        # Convert to sets for easier calculation
        true_set = set(true_malicious)
        detected_set = set(detected_malicious)
        
        # True positives: correctly identified malicious
        tp = len(true_set.intersection(detected_set))
        
        # False positives: benign identified as malicious
        fp = len(detected_set - true_set)
        
        # False negatives: malicious not detected
        fn = len(true_set - detected_set)
        
        # True negatives: benign correctly identified as benign (need total count)
        # This calculation assumes we know the total number of clients
        
        # calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn
        }
    
    def get_convergence_metrics(self):
        if len(self.round_metrics) < 2:
            return {}
        
        accuracies = [r['global_accuracy'] for r in self.round_metrics if r['global_accuracy'] is not None]
        losses = [r['global_loss'] for r in self.round_metrics if r['global_loss'] is not None]
        
        if not accuracies or not losses:
            return {}
        
        # final acc and loss
        final_accuracy = accuracies[-1]
        final_loss = losses[-1]
        
        # Convergence speed (rounds to reach 80% of final accuracy)
        target_accuracy = 0.8 * final_accuracy
        convergence_round = None
        for i, acc in enumerate(accuracies):
            if acc >= target_accuracy:
                convergence_round = i + 1
                break
        
        # stability - variance in last 20% of rounds
        stability_window = max(1, len(accuracies) // 5)
        recent_accuracies = accuracies[-stability_window:]
        stability = np.var(recent_accuracies) if len(recent_accuracies) > 1 else 0.0
        
        return {
            'final_accuracy': final_accuracy,
            'final_loss': final_loss,
            'convergence_round': convergence_round,
            'stability_variance': stability,
            'total_rounds': len(accuracies)
        }
    
    def get_summary_stats(self):
        summary = {
            'convergence': self.get_convergence_metrics(),
            'total_rounds': len(self.round_metrics),
            'total_clients': len(self.client_metrics),
            'attack_rounds': len(self.attack_metrics),
            'defense_rounds': len(self.defense_metrics)
        }
        
        # Attack effectiveness
        if self.attack_metrics:
            total_attacks = sum(len(attacks) for attacks in self.attack_metrics.values())
            successful_attacks = sum(
                sum(1 for attack in attacks if attack['attack_success'])
                for attacks in self.attack_metrics.values()
            )
            summary['attack_success_rate'] = successful_attacks / total_attacks if total_attacks > 0 else 0.0
        else:
            summary['attack_success_rate'] = 0.0
        
        # Defense effectiveness
        if self.defense_metrics:
            total_rejected = sum(
                sum(defense['rejected_count'] for defense in defenses)
                for defenses in self.defense_metrics.values()
            )
            summary['total_rejected_models'] = total_rejected
        else:
            summary['total_rejected_models'] = 0
        
        return summary
    
    def plot_training_curves(self, save_path=None):
        if not self.round_metrics:
            return
        
        rounds = [r['round'] for r in self.round_metrics]
        accuracies = [r['global_accuracy'] for r in self.round_metrics if r['global_accuracy'] is not None]
        losses = [r['global_loss'] for r in self.round_metrics if r['global_loss'] is not None]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        if accuracies:
            ax1.plot(rounds[:len(accuracies)], accuracies, 'b-', linewidth=2)
            ax1.set_xlabel('Round')
            ax1.set_ylabel('Global Accuracy')
            ax1.set_title('Training Accuracy')
            ax1.grid(True, alpha=0.3)
        
        # Loss plot
        if losses:
            ax2.plot(rounds[:len(losses)], losses, 'r-', linewidth=2)
            ax2.set_xlabel('Round')
            ax2.set_ylabel('Global Loss')
            ax2.set_title('Training Loss')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def plot_defense_analysis(self, save_path=None):
        if not self.defense_metrics:
            return
        
        defense_types = []
        rejection_counts = []
        
        for round_defenses in self.defense_metrics.values():
            for defense in round_defenses:
                defense_types.append(defense['defense_type'])
                rejection_counts.append(defense['rejected_count'])
        
        if not defense_types:
            return
        
        # create summary plot
        unique_defenses = list(set(defense_types))
        avg_rejections = []
        
        for defense_type in unique_defenses:
            rejections = [count for dt, count in zip(defense_types, rejection_counts) if dt == defense_type]
            avg_rejections.append(np.mean(rejections))
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(unique_defenses, avg_rejections)
        plt.xlabel('Defense Type')
        plt.ylabel('Average Rejected Models per Round')
        plt.title('Defense Effectiveness: Average Rejections')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, avg_rejections):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def export_to_dict(self):
        return {
            'round_metrics': self.round_metrics,
            'client_metrics': self.client_metrics,
            'attack_metrics': self.attack_metrics,
            'defense_metrics': self.defense_metrics,
            'summary': self.get_summary_stats()
        }
    
    def _flatten_model(self, model):
        #flatten model parameters into a 1D tensor
        params = []
        for param in model.parameters():
            params.append(param.data.view(-1))
        return torch.cat(params)