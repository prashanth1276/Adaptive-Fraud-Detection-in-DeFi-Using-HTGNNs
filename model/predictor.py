import torch
import torch.nn.functional as F
from model.htg_nn import HTGNN
from model.adaptive_engine import adapt_model_to_new_fraud

class FraudPredictor:
    def __init__(self, model_path, input_dim, edge_types):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Initialize HTGNN with dimensions consistent with your graph data
        self.model = HTGNN(
            input_dim=input_dim,
            hidden_dim=128,
            output_dim=1,
            time_dim=32,
            edge_types=edge_types
        ).to(self.device)
        
        # 2. Load the trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state"])
            self.threshold = checkpoint.get("threshold", 0.5)
        else:
            self.model.load_state_dict(checkpoint)
            self.threshold = 0.5

        self.model.eval()

    def analyze_transaction(self, subgraph, target_address, node_id_map, current_time=None):
        if subgraph is None:
            return None, "Error: Subgraph context could not be built."

        if current_time is not None:
            subgraph.graph_data['current_time'] = current_time
        elif 'current_time' not in subgraph.graph_data:
            subgraph.graph_data['current_time'] = subgraph.graph_data.get('max_timestamp', 0)

        subgraph = subgraph.to(self.device)
        node_idx = node_id_map.get(target_address)
        if node_idx is None:
            return None, f"Target {target_address} not found."

        with torch.no_grad():
            logits = self.model(subgraph)
            prob = torch.sigmoid(logits.view(-1)[node_idx]).item()
            prediction = int(prob > self.threshold)

        return {
            "target_address": target_address,
            "fraud_probability": round(prob, 4),
            "is_fraud": bool(prediction),
            "risk_level": "High" if prob > 0.8 else "Medium" if prob > 0.5 else "Low",
            "context_size": subgraph.num_nodes()
        }, None

    def trigger_adaptive_learning(self, subgraph, target_address, node_id_map, user_label, current_time=None):
        """
        Step 12 & 13: Self-Evolving Loop.
        When a human confirms a new fraud pattern, update the model weights.
        """
        node_idx = node_id_map.get(target_address)
        if node_idx is None:
            return "Failed: Node index missing."

        # Redirect to our specialized adaptive_engine
        # This ensures we use FocalLoss and the specific 15-epoch loop we perfected
        status = adapt_model_to_new_fraud(
            self.model, 
            subgraph.to(self.device), 
            node_idx, 
            label=user_label,
            current_time=current_time
        )
        
        # Reload updated weights to ensure predictor stays current
        # In a real system, you'd save to 'best_model.pth' inside adaptive_engine
        return status