import torch.nn as nn

def initialize_weights(layer):
    """
    Initialize the weights of the layer.
    This will initialize the weights using a normal distribution 
    and the biases to zero.
    """
    if isinstance(layer, nn.Linear):
        nn.init.normal_(layer.weight, mean=0.0, std=0.01)
        nn.init.zeros_(layer.bias)
    elif isinstance(layer, nn.Conv2d):
        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    elif isinstance(layer, nn.LSTM):
        for name, param in layer.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
            elif 'bias' in name:
                nn.init.zeros_(param)
