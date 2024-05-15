import torch
import torch.nn as nn
import torch.nn.functional as F

class RunTier(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(RunTier, self).__init__()
        # Initialize parameters for transformations within the Run Tier
        self.W_b = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.b_b = nn.Parameter(torch.zeros(hidden_dim))
        
        self.W_hb = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.b_hb = nn.Parameter(torch.zeros(hidden_dim))
        
        self.W_r = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.b_r = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x, pos_enc):
        """
        Forward pass through the Run Tier model.
        
        Parameters:
        x (Tensor): The input features from the Behavior Tier of shape (batch_size, sequence_length, hidden_dim).
        pos_enc (Tensor): The one-hot positional encoding.
        
        Returns:
        Tensor: The output tensor representing the updated run state.
        """
        # Initial hidden state
        h_prev = torch.zeros_like(x[:, 0, :])

        for i in range(x.size(1)):  # Iterate over the sequence length
            bi = torch.tanh(F.linear(x[:, i, :], self.W_b, self.b_b))
            hi = torch.tanh(F.linear(h_prev, self.W_hb, self.b_hb)) + bi
            ri = torch.tanh(F.linear(hi, self.W_r, self.b_r))

            # Apply one-hot positional encoding modification if needed
            if pos_enc is not None:
                hi += torch.tanh(F.linear(ri + pos_enc[:, i, :], self.W_b, self.b_b))

            # Update previous hidden state for the next step
            h_prev = hi

        return hi

# Example usage
batch_size = 1
seq_length = 5
hidden_dim = 10

# Example input data (output from Behavior Tier)
x = torch.randn(batch_size, seq_length, hidden_dim)
pos_enc = torch.randn(batch_size, seq_length, hidden_dim)  # One-hot positional encoding

model = RunTier(hidden_dim, hidden_dim)
output = model(x, pos_enc)
print("Output shape:", output.shape)
