
import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    """NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss"""
    
    def __init__(self, temperature=0.07, batch_size=128):
        super().__init__()
        self.temperature = temperature
        self.batch_size = batch_size
    
    def forward(self, z_i, z_j):
        # Normalize projections
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        batch_size = z_i.shape[0]
        
        # Concatenate both views
        z = torch.cat([z_i, z_j], dim=0)
        
        # Compute similarity matrix
        similarity_matrix = torch.mm(z, z.t()) / self.temperature
        
        # Create mask for positive pairs
        mask = torch.eye(batch_size, dtype=torch.bool, device=z.device)
        mask = torch.cat([
            torch.cat([torch.zeros_like(mask), mask], dim=1),
            torch.cat([mask, torch.zeros_like(mask)], dim=1)
        ], dim=0)
        
        # Remove self-similarity
        self_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        
        # Compute logits
        pos = similarity_matrix[mask].view(2 * batch_size, 1)
        neg = similarity_matrix[~self_mask].view(2 * batch_size, -1)
        
        logits = torch.cat([pos, neg], dim=1)
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss
