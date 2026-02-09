import torch
import torch.nn as nn
import torch.nn.functional as F


class EuchreQNetwork(nn.Module):
    """
    Deep Q-Network for euchre card play.

    Input:  60-dim state vector  (from state_encoder.encode_state)
    Output: 24 Q-values, one per card in the 24-card euchre deck.

    The caller applies the legal mask AFTER the forward pass, so this network
    does not need to know anything about legality.  It just learns values.

    Architecture kept deliberately small.  Euchre has a tiny state/action space;
    two hidden layers of 128 is more than enough.  Widen only if you see
    underfitting after thousands of training episodes.
    """

    def __init__(self, state_dim=60, action_dim=24, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        """
        state : tensor  (batch_size, 60)
        return: tensor  (batch_size, 24)   raw Q-values
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
