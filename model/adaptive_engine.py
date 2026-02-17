import torch
import torch.optim as optim
import os

# Import FocalLoss from your train module to handle class imbalance
try:
    from train.train import FocalLoss
except ImportError:
    # Fallback if structure varies slightly in different environments
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from train.train import FocalLoss

def adapt_model_to_new_fraud(model, graph, confirmed_node_id, label=1, current_time=None):
    device = next(model.parameters()).device
    graph = graph.to(device)
    model.train()

    if current_time is not None:
        graph.graph_data['current_time'] = current_time
    elif 'current_time' not in graph.graph_data:
        graph.graph_data['current_time'] = graph.graph_data.get('max_timestamp', 0)
    
    # 1. Setup: Very low LR to prevent 'Catastrophic Forgetting' of old patterns
    optimizer = optim.Adam(model.parameters(), lr=5e-6) 
    criterion = FocalLoss(alpha=0.85, gamma=2.0)
    
    # 2. Update Knowledge Graph State
    graph.nodes['node'].data['label'] = graph.nodes['node'].data['label'].clone()
    graph.nodes['node'].data['label'][confirmed_node_id] = torch.tensor(label, device=device).long()
    
    # 3. Micro-Training Loop (15 Epochs)
    # This specifically optimizes the weights for the target node and its neighborhood
    for epoch in range(15):
        logits = model(graph)
        
        # Select only the target node's prediction
        pred = logits.view(-1)[confirmed_node_id].unsqueeze(0)
        target = torch.tensor([float(label)], device=device)
        
        loss = criterion(pred, target)
        
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to maintain model stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
    # 4. Persistence: Save as 'adapted_model.pth' as per system requirements
    torch.save(model.state_dict(), 'adapted_model.pth')
    
    model.eval() # Return to inference mode
    return "✅ Model Weights Adapted to New Pattern!"