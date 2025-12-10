import logging
import os
import json
import datetime
from pathlib import Path


class FLLogger:
    """Comprehensive logging system for federated learning experiments."""
    
    def __init__(self, experiment_name, log_dir="logs", log_level=logging.INFO):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # experiment-specific directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.log_dir / f"{experiment_name}_{timestamp}"
        self.experiment_dir.mkdir(exist_ok=True)
        
        # setup loggers
        self.setup_loggers(log_level)
        
        # init log data structures
        self.experiment_config = {}
        self.round_logs = []
        self.client_logs = {}
        self.attack_logs = {}
        self.defense_logs = {}
        
    def setup_loggers(self, log_level):
        """Setup different loggers for different components."""
        # Main experiment logger
        self.main_logger = logging.getLogger(f"FL_Main_{self.experiment_name}")
        self.main_logger.setLevel(log_level)
        
        # create file handler
        main_handler = logging.FileHandler(self.experiment_dir / "experiment.log")
        main_handler.setLevel(log_level)
        
        # create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        main_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # add handlers to logger
        self.main_logger.addHandler(main_handler)
        self.main_logger.addHandler(console_handler)
        
        # attack logger
        self.attack_logger = logging.getLogger(f"FL_Attack_{self.experiment_name}")
        self.attack_logger.setLevel(log_level)
        attack_handler = logging.FileHandler(self.experiment_dir / "attacks.log")
        attack_handler.setFormatter(formatter)
        self.attack_logger.addHandler(attack_handler)
        
        # defense logger
        self.defense_logger = logging.getLogger(f"FL_Defense_{self.experiment_name}")
        self.defense_logger.setLevel(log_level)
        defense_handler = logging.FileHandler(self.experiment_dir / "defenses.log")
        defense_handler.setFormatter(formatter)
        self.defense_logger.addHandler(defense_handler)
        
        # client logger
        self.client_logger = logging.getLogger(f"FL_Client_{self.experiment_name}")
        self.client_logger.setLevel(log_level)
        client_handler = logging.FileHandler(self.experiment_dir / "clients.log")
        client_handler.setFormatter(formatter)
        self.client_logger.addHandler(client_handler)
    
    def log_experiment_config(self, config):
        """Log experiment configuration."""
        self.experiment_config = config
        self.main_logger.info(f"Experiment Configuration: {json.dumps(config, indent=2)}")
        
        with open(self.experiment_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2, default=str)
    
    def log_round_start(self, round_num, participating_clients=None):
        self.main_logger.info(f"=== Round {round_num} Started ===")
        if participating_clients:
            self.main_logger.info(f"Participating clients: {participating_clients}")
        
        round_data = {
            'round': round_num,
            'timestamp': datetime.datetime.now().isoformat(),
            'participating_clients': participating_clients or [],
            'events': []
        }
        self.round_logs.append(round_data)
    
    def log_round_end(self, round_num, global_accuracy, global_loss, 
                     test_accuracy=None, test_loss=None):
        self.main_logger.info(f"=== Round {round_num} Completed ===")
        self.main_logger.info(f"Global Accuracy: {global_accuracy:.4f}, Global Loss: {global_loss:.4f}")
        if test_accuracy is not None:
            self.main_logger.info(f"Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")
        
        # update round log
        if self.round_logs and self.round_logs[-1]['round'] == round_num:
            self.round_logs[-1].update({
                'global_accuracy': global_accuracy,
                'global_loss': global_loss,
                'test_accuracy': test_accuracy,
                'test_loss': test_loss,
                'end_timestamp': datetime.datetime.now().isoformat()
            })
    
    def log_client_training(self, client_id, round_num, local_accuracy, local_loss, 
                           epochs_completed, is_malicious=False):
        # Log training infos
        message = f"Client {client_id} - Round {round_num}: Acc={local_accuracy:.4f}, Loss={local_loss:.4f}, Epochs={epochs_completed}"
        if is_malicious:
            message += " [MALICIOUS]"
        
        self.client_logger.info(message)
        
        # Store in structured format
        if client_id not in self.client_logs:
            self.client_logs[client_id] = []
        
        client_data = {
            'round': round_num,
            'local_accuracy': local_accuracy,
            'local_loss': local_loss,
            'epochs_completed': epochs_completed,
            'is_malicious': is_malicious,
            'timestamp': datetime.datetime.now().isoformat()
        }
        self.client_logs[client_id].append(client_data)
    
    def log_attack(self, round_num, attack_type, attacker_ids, attack_params, 
                   attack_success=None, similarity_metrics=None):
        # log attack infos
        attack_msg = f"Round {round_num}: {attack_type} attack by clients {attacker_ids}"
        if attack_success is not None:
            attack_msg += f" - Success: {attack_success}"
        
        self.attack_logger.info(attack_msg)
        self.attack_logger.info(f"Attack parameters: {attack_params}")
        
        if similarity_metrics:
            self.attack_logger.info(f"Similarity metrics: {similarity_metrics}")
        
        # store structured data
        if round_num not in self.attack_logs:
            self.attack_logs[round_num] = []
        
        attack_data = {
            'attack_type': attack_type,
            'attacker_ids': attacker_ids,
            'attack_params': attack_params,
            'attack_success': attack_success,
            'similarity_metrics': similarity_metrics,
            'timestamp': datetime.datetime.now().isoformat()
        }
        self.attack_logs[round_num].append(attack_data)
    
    def log_defense(self, round_num, defense_type, defense_params, 
                   rejected_clients=None, detected_malicious=None, effectiveness_metrics=None):
        # log defense infos
        defense_msg = f"Round {round_num}: {defense_type} defense applied"
        if rejected_clients:
            defense_msg += f" - Rejected clients: {rejected_clients}"
        if detected_malicious:
            defense_msg += f" - Detected malicious: {detected_malicious}"
        
        self.defense_logger.info(defense_msg)
        self.defense_logger.info(f"Defense parameters: {defense_params}")
        
        if effectiveness_metrics:
            self.defense_logger.info(f"Effectiveness metrics: {effectiveness_metrics}")
        
        # Store structured data
        if round_num not in self.defense_logs:
            self.defense_logs[round_num] = []
        
        defense_data = {
            'defense_type': defense_type,
            'defense_params': defense_params,
            'rejected_clients': rejected_clients or [],
            'detected_malicious': detected_malicious or [],
            'effectiveness_metrics': effectiveness_metrics,
            'timestamp': datetime.datetime.now().isoformat()
        }
        self.defense_logs[round_num].append(defense_data)
    
    def log_model_metrics(self, round_num, model_similarities, model_norms):
        # log model relevant metrics
        self.main_logger.info(f"Round {round_num} - Model Similarities: {model_similarities}")
        self.main_logger.info(f"Round {round_num} - Model Norms: {model_norms}")
        
        # add to round log
        if self.round_logs and self.round_logs[-1]['round'] == round_num:
            self.round_logs[-1]['events'].append({
                'type': 'model_metrics',
                'model_similarities': model_similarities,
                'model_norms': model_norms,
                'timestamp': datetime.datetime.now().isoformat()
            })
    
    def log_error(self, error_msg, exception=None):
        self.main_logger.error(error_msg)
        if exception:
            self.main_logger.exception(exception)
    
    def log_warning(self, warning_msg):
        self.main_logger.warning(warning_msg)
    
    def log_info(self, info_msg):
        self.main_logger.info(info_msg)
    
    def save_summary(self):
        summary = {
            'experiment_name': self.experiment_name,
            'experiment_config': self.experiment_config,
            'experiment_dir': str(self.experiment_dir),
            'total_rounds': len(self.round_logs),
            'total_clients': len(self.client_logs),
            'total_attack_rounds': len(self.attack_logs),
            'total_defense_rounds': len(self.defense_logs),
            'final_timestamp': datetime.datetime.now().isoformat()
        }
        
        # we calculate final metrics
        if self.round_logs:
            final_round = self.round_logs[-1]
            summary['final_accuracy'] = final_round.get('global_accuracy')
            summary['final_loss'] = final_round.get('global_loss')
            summary['final_test_accuracy'] = final_round.get('test_accuracy')
            summary['final_test_loss'] = final_round.get('test_loss')
        
        with open(self.experiment_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return summary
    
    def save_all_logs(self):
        # save all logs to json files
        with open(self.experiment_dir / "round_logs.json", 'w') as f:
            json.dump(self.round_logs, f, indent=2, default=str)
        
        with open(self.experiment_dir / "client_logs.json", 'w') as f:
            json.dump(self.client_logs, f, indent=2, default=str)
        
        with open(self.experiment_dir / "attack_logs.json", 'w') as f:
            json.dump(self.attack_logs, f, indent=2, default=str)
        
        with open(self.experiment_dir / "defense_logs.json", 'w') as f:
            json.dump(self.defense_logs, f, indent=2, default=str)
        
        summary = self.save_summary()
        
        self.main_logger.info(f"All logs saved to {self.experiment_dir}")
        return summary
    
    def get_log_directory(self):
        return self.experiment_dir