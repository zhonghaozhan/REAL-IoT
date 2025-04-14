"""
Mini-batch training code for DGL 1.1.3

This file contains the correct imports and sample code for mini-batch training 
with DGL 1.1.3 to avoid memory issues.
"""

import torch as th
import dgl
from torch.utils.data import DataLoader
import dgl.function as fn
import torch.nn.functional as F

# For edge mini-batch training in DGL 1.1.3
# Instead of EdgeDataLoader, use a combination of a custom dataset and DataLoader
class SimpleEdgeDataset:
    def __init__(self, g, eids):
        self.g = g
        self.eids = eids

    def __len__(self):
        return len(self.eids)

    def __getitem__(self, idx):
        return self.g, self.eids[idx]

def create_edge_dataloader(graph, batch_size, shuffle=True):
    """
    Create a mini-batch dataloader for edge prediction tasks.
    
    Args:
        graph: DGL graph
        batch_size: Number of edges per batch
        shuffle: Whether to shuffle the edges
        
    Returns:
        DataLoader for mini-batch training
    """
    # Create a dataset with all edge IDs
    edge_dataset = SimpleEdgeDataset(graph, th.arange(graph.num_edges()))
    
    # Create a DataLoader
    def collate_fn(batch_data):
        graphs, eids = zip(*batch_data)
        batched_eids = th.cat([eids.view(-1) for eids in eids])
        return graphs[0], batched_eids
    
    return DataLoader(
        edge_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )

def train_model_with_mini_batches(model, graph, optimizer, criterion, edge_mask, num_epochs, batch_size):
    """
    Train a GNN model using mini-batches to reduce memory usage.
    
    Args:
        model: GNN model
        graph: DGL graph
        optimizer: PyTorch optimizer
        criterion: Loss function
        edge_mask: Boolean mask for training edges
        num_epochs: Number of training epochs
        batch_size: Batch size
    """
    # Create dataloader
    dataloader = create_edge_dataloader(graph, batch_size)
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for g, batch_eids in dataloader:
            # Filter for training edges
            if edge_mask is not None:
                mask = edge_mask[batch_eids]
                if not mask.any():
                    continue
                batch_eids = batch_eids[mask]
                
            # Forward pass and compute loss
            pred = model(g, g.ndata['h'], g.edata['h'])
            batch_pred = pred[batch_eids]
            batch_labels = g.edata['label'][batch_eids]
            
            loss = criterion(batch_pred, batch_labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = th.max(batch_pred, 1)
            correct += (predicted == batch_labels).sum().item()
            total += batch_labels.size(0)
            total_loss += loss.item() * batch_labels.size(0)
            
        # Print statistics
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss/total:.4f}, Accuracy: {100*correct/total:.2f}%')
    
    return model

# Example usage in notebook:
"""
# Define batch size
batch_size = 1024  # Adjust based on your memory constraints

# Import mini-batch helpers
from dgl_batch_helpers.edge_batch import train_model_with_mini_batches

# Train model with mini-batches
train_model_with_mini_batches(
    model=model,
    graph=G,
    optimizer=opt,
    criterion=criterion,
    edge_mask=G.edata['train_mask'],
    num_epochs=1000,
    batch_size=batch_size
)
""" 