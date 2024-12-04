import numpy as np
import pandas as pd
import torch as T


# preprocess the data and make test train samples

# make a covariance matrix to see if you can make this a little more efficient

# make the model 64 x 64 x 64 should be easy enough

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            # FILL THIS WITH NUMBER OF FEATURES IN DATASET
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

# 