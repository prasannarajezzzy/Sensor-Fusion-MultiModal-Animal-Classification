import torch
import torch.nn as nn
from torchvision import models

class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, num_classes, embed_dim, num_heads, num_layers):
        super(VisionTransformer, self).__init__()
        
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2  # Assuming input images have 3 channels
        
        self.patch_embedding = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.positional_embedding = nn.Parameter(torch.rand(1, num_patches + 1, embed_dim))
        
        self.transformer = models.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers
        )
        
        self.fc = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embedding(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.positional_embedding.repeat(B, 1, 1), x], dim=1)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc(x)
        return x
