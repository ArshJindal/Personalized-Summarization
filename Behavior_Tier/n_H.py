import torch
import torch.nn as nn
import torch.nn.functional as F

class BehaviorTier(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(BehaviorTier, self).__init__()
        # Initialize weight matrices and bias vectors for each transformation
        self.W_nh = nn.Parameter(torch.randn(hidden_dim, input_dim))
        self.b_nh = nn.Parameter(torch.zeros(hidden_dim))
        
        self.W_h = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.b_h = nn.Parameter(torch.zeros(hidden_dim))
        
        self.W_nr = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.b_nr = nn.Parameter(torch.zeros(hidden_dim))
        
        self.W_nt = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.b_nt = nn.Parameter(torch.zeros(hidden_dim))
        
        self.W_b = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.b_b = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x, pos_enc):
        """
        Forward pass through the Behavior Tier model.
        
        Parameters:
        x (Tensor): The input features of shape (batch_size, sequence_length, input_dim).
        pos_enc (Tensor): The one-hot positional encoding.
        
        Returns:
        Tensor: The output tensor representing the updated behavior state.
        """
        # Initial hidden state
        h_prev = torch.zeros_like(x[:, 0, :])

        for i in range(x.size(1)):  # Iterate over the sequence length
            nh = torch.tanh(F.linear(x[:, i, :], self.W_nh, self.b_nh))
            h = torch.tanh(F.linear(h_prev, self.W_h, self.b_h))
            h_curr = h + nh  # Update current hidden state

            nr = torch.tanh(F.linear(h_curr, self.W_nr, self.b_nr))
            nt = torch.tanh(F.linear(nr, self.W_nt, self.b_nt))
            h_next = h_curr + nt  # Update next hidden state

            # Apply one-hot positional encoding modification if needed
            if pos_enc is not None:
                h_next += torch.tanh(F.linear(h_next + pos_enc[:, i, :], self.W_b, self.b_b))

            # Update previous hidden state for the next step
            h_prev = h_next

        return h_next

# Example usage
batch_size = 1
seq_length = 5
input_dim = 10
hidden_dim = 10

# Example input data
x = torch.randn(batch_size, seq_length, input_dim)
pos_enc = torch.randn(batch_size, seq_length, hidden_dim)  # One-hot positional encoding

model = BehaviorTier(input_dim, hidden_dim)
output = model(x, pos_enc)
print("Output shape:", output.shape)
