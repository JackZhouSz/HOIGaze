import torch
import torch.nn.functional as F
from torch import layer_norm, nn
import math


class temporal_self_attention(nn.Module):

    def __init__(self, latent_dim, num_head, dropout):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim, bias=False)
        self.key = nn.Linear(latent_dim, latent_dim, bias=False)
        self.value = nn.Linear(latent_dim, latent_dim, bias=False)
        self.dropout = nn.Dropout(dropout)        
        
    def forward(self, x):
        """
        x: B, T, D
        """
        B, T, D = x.shape
        H = self.num_head
        # B, T, 1, D
        query = self.query(self.norm(x)).unsqueeze(2)
        # B, 1, T, D
        key = self.key(self.norm(x)).unsqueeze(1)
        query = query.view(B, T, H, -1)
        key = key.view(B, T, H, -1)
        # B, T, T, H
        attention = torch.einsum('bnhd,bmhd->bnmh', query, key) / math.sqrt(D // H)
        weight = self.dropout(F.softmax(attention, dim=2))
        value = self.value(self.norm(x)).view(B, T, H, -1)
        y = torch.einsum('bnmh,bmhd->bnhd', weight, value).reshape(B, T, D)
        y = x + y
        return y

        
class spatial_self_attention(nn.Module):

    def __init__(self, latent_dim, num_head, dropout):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim, bias=False)
        self.key = nn.Linear(latent_dim, latent_dim, bias=False)
        self.value = nn.Linear(latent_dim, latent_dim, bias=False)
        self.dropout = nn.Dropout(dropout)        
        
    def forward(self, x):
        """
        x: B, S, D
        """
        B, S, D = x.shape
        H = self.num_head
        # B, S, 1, D
        query = self.query(self.norm(x)).unsqueeze(2)
        # B, 1, S, D
        key = self.key(self.norm(x)).unsqueeze(1)
        query = query.view(B, S, H, -1)
        key = key.view(B, S, H, -1)
        # B, S, S, H
        attention = torch.einsum('bnhd,bmhd->bnmh', query, key) / math.sqrt(D // H)
        weight = self.dropout(F.softmax(attention, dim=2))
        value = self.value(self.norm(x)).view(B, S, H, -1)
        y = torch.einsum('bnmh,bmhd->bnhd', weight, value).reshape(B, S, D)
        y = x + y
        return y

        
class temporal_cross_attention(nn.Module):

    def __init__(self, latent_dim, mod_dim, num_head, dropout):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.mod_norm = nn.LayerNorm(mod_dim)
        self.query = nn.Linear(latent_dim, latent_dim, bias=False)
        self.key = nn.Linear(mod_dim, latent_dim, bias=False)
        self.value = nn.Linear(mod_dim, latent_dim, bias=False)
        self.dropout = nn.Dropout(dropout)        
    
    def forward(self, x, xf):
        """
        x: B, T, D
        xf: B, N, L
        """
        B, T, D = x.shape
        N = xf.shape[1]
        H = self.num_head
        # B, T, 1, D
        query = self.query(self.norm(x)).unsqueeze(2)
        # B, 1, N, D
        key = self.key(self.mod_norm(xf)).unsqueeze(1)
        query = query.view(B, T, H, -1)
        key = key.view(B, N, H, -1)
        # B, T, N, H
        attention = torch.einsum('bnhd,bmhd->bnmh', query, key) / math.sqrt(D // H)
        weight = self.dropout(F.softmax(attention, dim=2))
        value = self.value(self.mod_norm(xf)).view(B, N, H, -1)
        y = torch.einsum('bnmh,bmhd->bnhd', weight, value).reshape(B, T, D)
        y = x + y
        return y

        
class spatial_cross_attention(nn.Module):

    def __init__(self, latent_dim, mod_dim, num_head, dropout):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.mod_norm = nn.LayerNorm(mod_dim)
        self.query = nn.Linear(latent_dim, latent_dim, bias=False)
        self.key = nn.Linear(mod_dim, latent_dim, bias=False)
        self.value = nn.Linear(mod_dim, latent_dim, bias=False)
        self.dropout = nn.Dropout(dropout)        
    
    def forward(self, x, xf):
        """
        x: B, S, D
        xf: B, N, L
        """
        B, S, D = x.shape
        N = xf.shape[1]
        H = self.num_head
        # B, S, 1, D
        query = self.query(self.norm(x)).unsqueeze(2)
        # B, 1, N, D
        key = self.key(self.mod_norm(xf)).unsqueeze(1)
        query = query.view(B, S, H, -1)
        key = key.view(B, N, H, -1)
        # B, S, N, H
        attention = torch.einsum('bnhd,bmhd->bnmh', query, key) / math.sqrt(D // H)
        weight = self.dropout(F.softmax(attention, dim=2))
        value = self.value(self.mod_norm(xf)).view(B, N, H, -1)
        y = torch.einsum('bnmh,bmhd->bnhd', weight, value).reshape(B, S, D)
        y = x + y
        return y