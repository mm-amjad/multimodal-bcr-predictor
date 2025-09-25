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
        layers = []
        for layer_size in layer_sizes:
            layers.append(nn.Linear(embedding_dim, layer_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            embedding_dim = layer_size
        layers.append(nn.Linear(embedding_dim, 1)) 
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)