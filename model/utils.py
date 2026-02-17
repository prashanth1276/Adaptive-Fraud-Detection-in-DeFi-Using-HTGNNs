import dgl
import torch
import numpy as np

def process_transaction_context(neo4j_data, node_id_map):
    """
    Refined Step 5: Bridges the raw Neo4j contextual subgraph to 
    the HTGNN-ready tensor format for the demo.
    """
    if not neo4j_data:
        return None, "No transaction data available"
    
    # In a demo, we ensure that the feature vectors are consistent
    # Ethereum data often has huge values (Wei), so we log-normalize them
    def normalize_features(feat_list):
        # Convert to numpy for easier manipulation
        feats = np.array(feat_list, dtype=np.float32)
        # Log scaling to handle extreme Ethereum values (Value, Gas)
        # We use log1p(x) which is log(1+x) to avoid log(0)
        return torch.from_numpy(np.log1p(feats))

    try:
        # If the input is already a DGL graph from database.py:
        if isinstance(neo4j_data, dgl.DGLGraph):
            g = neo4j_data
            # Apply normalization to the features
            g.nodes['node'].data['feat'] = normalize_features(g.nodes['node'].data['feat'])
            return g, None
            
        return None, "Unexpected data format: Expected DGL Graph Context"

    except Exception as e:
        return None, f"Preprocessing Error: {str(e)}"

def format_prediction_report(results):
    """
    Formats the HTGNN output into a human-readable dashboard format
    for your project presentation.
    """
    prob = results['fraud_probability']
    
    # ANSI color codes for terminal-based demo visibility
    COLOR_RED = "\033[91m"
    COLOR_GREEN = "\033[92m"
    COLOR_YELLOW = "\033[93m"
    RESET = "\033[0m"

    status = f"{COLOR_RED}⚠️ FRAUD DETECTED{RESET}" if results['is_fraud'] else f"{COLOR_GREEN}✅ LEGITIMATE{RESET}"
    
    report = f"""
    {'='*40}
    DEFI FRAUD ANALYSIS REPORT
    {'='*40}
    Target Address: {results['target_address']}
    Status:         {status}
    Risk Score:     {prob * 100:.2f}%
    Confidence:     {results['risk_level']}
    Context Nodes:  {results['context_size']} (Multi-hop Subgraph)
    {'='*40}
    """
    return report