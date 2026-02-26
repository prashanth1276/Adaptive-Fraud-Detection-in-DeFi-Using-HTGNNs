"""
Adaptive online learning engine for the Heterogeneous Temporal Graph Neural Network.

This module implements the inference-time fine-tuning mechanism that allows the
HTGNN to incorporate analyst-confirmed fraud labels without retraining from scratch.
It performs a micro-training loop of 15 epochs over a single confirmed node using
FocalLoss with a conservative learning rate (5e-6) to minimize catastrophic
forgetting of previously learned fraud patterns. After adaptation, the updated model
weights are persisted to disk. This module is invoked exclusively at inference time
via the Streamlit dashboard (app.py) in response to analyst feedback.
"""

import torch
import torch.optim as optim

from train.train import FocalLoss


def adapt_model_to_new_fraud(
    model, graph, confirmed_node_id, label=1, current_time=None
):
    """Performs targeted online adaptation of HTGNN weights on a single confirmed label.

    Executes a 15-epoch micro-training loop restricted to the prediction for the
    specified node, using a very low learning rate to preserve global model
    calibration while encoding the analyst-confirmed label into the weight space.
    After adaptation, the model is returned to evaluation mode and the updated
    checkpoint is written to disk.

    The adaptation strategy is equivalent to a single-sample continual learning
    step under the Elastic Weight Consolidation (EWC) spirit: the extremely low
    learning rate (5e-6) acts as an implicit regularizer, preventing catastrophic
    forgetting of the pre-trained fraud distribution.

    Args:
        model (torch.nn.Module): The HTGNN instance to adapt. Must already be
            loaded with trained weights on the target device.
        graph (dgl.DGLHeteroGraph): The k-hop subgraph centered on the target
            node, containing pre-scaled node feature tensors and edge timestamps.
        confirmed_node_id (int): Local node index within `graph` corresponding
            to the analyst-confirmed address.
        label (int, optional): Ground-truth label to enforce. 1 indicates
            confirmed fraud; 0 indicates confirmed benign. Defaults to 1.
        current_time (float | None, optional): Unix timestamp representing the
            temporal query boundary for edge masking during adaptation. If None,
            defaults to the graph's stored max_timestamp.

    Returns:
        str: Status message indicating successful weight adaptation.
    """
    device = next(model.parameters()).device
    graph = graph.to(device)
    model.train()

    # Set the temporal query boundary for causal edge masking during adaptation
    if current_time is not None:
        graph.graph_data["current_time"] = current_time
    elif "current_time" not in graph.graph_data:
        graph.graph_data["current_time"] = graph.graph_data.get("max_timestamp", 0)

    # Very low learning rate (5e-6) prevents large gradient updates that would
    # overwrite previously learned fraud pattern representations (catastrophic forgetting)
    optimizer = optim.Adam(model.parameters(), lr=5e-6)
    criterion = FocalLoss(alpha=0.85, gamma=2.0)

    # Update the graph's label tensor in-place to reflect the analyst's ground truth
    graph.nodes["node"].data["label"] = graph.nodes["node"].data["label"].clone()
    graph.nodes["node"].data["label"][confirmed_node_id] = torch.tensor(
        label, device=device
    ).long()

    # Micro-training loop: 15 epochs targeting only the confirmed node's prediction
    for epoch in range(15):
        logits = model(graph)

        # Isolate the logit for the single target node; unsqueeze for loss compatibility
        pred = logits.view(-1)[confirmed_node_id].unsqueeze(0)
        target = torch.tensor([float(label)], device=device)

        loss = criterion(pred, target)

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping at L2 norm = 1.0 prevents destabilizing weight updates
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    # Persist adapted weights for subsequent inference sessions
    torch.save(model.state_dict(), "adapted_model.pth")

    # Restore evaluation mode; disables dropout and batch norm training behavior
    model.eval()
    return "Model Weights Adapted to New Pattern!"
