import torch.nn as nn

class MLP(nn.Module):
    """Evaluation MLP architecture for predicting time to BCR.
    
    Attributes:
        network: Deep learning architecture for predicting time to BCR
        
    Args:
        embedding_dim (int): Input embedding dimension
        layer_sizes (list[int]): List of layer sizes of MLP architecture
        dropout (float): Dropout probability
    """
    def __init__(self, embedding_dim, layer_sizes, dropout):
        super(MLP, self).__init__()
        original_embedding_dim = embedding_dim
        layers = []
        for layer_size in layer_sizes:
            layers.append(nn.Linear(embedding_dim, layer_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            embedding_dim = layer_size
        layers.append(nn.Linear(embedding_dim, 1)) 
        self.network = nn.Sequential(*layers)

        self.fusion_MLP = nn.Sequential(
            nn.Linear(original_embedding_dim, original_embedding_dim * 4),
            nn.Dropout(dropout),
            nn.Linear(original_embedding_dim * 4, original_embedding_dim * 2),              
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(original_embedding_dim * 2, original_embedding_dim)
        )
        
    def forward(self, embedding):
        new_embedding = self.fusion_MLP(embedding)
        return self.network(new_embedding)