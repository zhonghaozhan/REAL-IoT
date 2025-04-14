"""
This file contains sample code to add to your notebook for mini-batch training.
Copy these cells to your notebook.
"""

# Cell 1: Define batch size and import helpers
"""
# Define batch size
batch_size = 1024  # Adjust based on your memory constraints

# Add helper module path to import path
import sys
sys.path.append('/media/ssd/test/GNN')

# Import mini-batch helpers
from dgl_batch_helpers.edge_batch import create_edge_dataloader, train_model_with_mini_batches
"""

# Cell 2: Modify training loop to use mini-batches
"""
# Create dataloader
dataloader = create_edge_dataloader(G, batch_size=batch_size, shuffle=True)

# Initialize model, optimizer, and criterion
# (use your existing model initialization code)
model = Model(G.ndata['h'].shape[2], 128, G.ndata['h'].shape[2], F.relu, 0.2).cuda()
opt = th.optim.Adam(model.parameters())

# Train model with mini-batches
num_epochs = 1000  # Adjust as needed
for epoch in range(num_epochs):
    total_loss = 0
    correct = 0
    total = 0
    
    for g, batch_eids in dataloader:
        # Get the batch labels
        batch_labels = g.edata['label'][batch_eids]
        
        # Forward pass
        pred = model(g, g.ndata['h'], g.edata['h'])
        batch_pred = pred[batch_eids]
        
        # Calculate loss
        loss = criterion(batch_pred, batch_labels)
        
        # Backward and optimize
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        # Calculate accuracy
        total_loss += loss.item() * batch_labels.size(0)
        total += batch_labels.size(0)
    
    # Print statistics
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {total_loss/total:.4f}')
        
        # Validate on training set
        with th.no_grad():
            train_pred = model(g, g.ndata['h'], g.edata['h'])
            train_acc = compute_accuracy(train_pred[g.edata['train_mask']], 
                                         g.edata['label'][g.edata['train_mask']])
            print(f'Training accuracy: {train_acc:.4f}')
"""

# Cell 3: Mini-batch prediction for test data
"""
# Test prediction with mini-batches to avoid OOM
test_dataloader = create_edge_dataloader(G_test, batch_size=batch_size, shuffle=False)
test_preds = []

with th.no_grad():
    for g, batch_eids in test_dataloader:
        # Forward pass
        node_features_test = g.ndata['feature']
        edge_features_test = g.edata['h']
        batch_pred = model(g, node_features_test, edge_features_test)
        
        # Store predictions
        test_preds.append(batch_pred[batch_eids])
    
    # Combine all predictions
    test_pred = th.cat(test_preds, dim=0)
    
    # Convert to class indices
    test_pred = test_pred.argmax(1)
    test_pred = th.Tensor.cpu(test_pred).detach().numpy()
    
    # Convert to original labels
    test_pred = le.inverse_transform(test_pred)
""" 