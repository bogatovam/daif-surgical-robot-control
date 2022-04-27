import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from base import BaseModel


class FcModel(BaseModel):

    def __init__(self, n_inputs, n_outputs, n_hidden=64, lr=1e-3, softmax=False, device='cpu'):
        super(FcModel, self).__init__()

        self.n_inputs = n_inputs  # Number of inputs
        self.n_hidden = n_hidden  # Number of hidden units
        self.n_outputs = n_outputs  # Number of outputs
        self.softmax = softmax  # If true apply a softmax function to the output

        self.fc1 = nn.Linear(self.n_inputs, self.n_hidden)  # Hidden layer
        self.fc2 = nn.Linear(self.n_hidden, self.n_outputs)  # Output layer

        self.optimizer = optim.Adam(self.parameters(), lr)  # Adam optimizer

        self.device = device
        self.to(self.device)

    def forward(self, x):
        # Define the forward pass:
        h_relu = F.relu(self.fc1(x))
        y = self.fc2(h_relu)

        if self.softmax:  # If true apply a softmax function to the output
            y = F.softmax(self.fc2(h_relu), dim=-1).clamp(min=1e-9, max=1 - 1e-9)

        return y
