import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Primeira camada + ativação
        output = self.layer1(x)
        output = self.relu(output)

        # Segunda camada + ativação
        output = self.layer2(output)
        output = self.relu(output)

        # Terceira camada
        output = self.layer3(output)
        
        return output