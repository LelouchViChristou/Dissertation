import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super(L2Norm, self).__init__()
        self.dim = dim

    def forward(self, x):
        return nn.functional.normalize(x, p=2, dim=self.dim)
    
# Define model
class SimpleAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
         
        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        # 784 ==> 9
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(1536, 1024),
            torch.nn.GELU(),
            nn.Dropout(0.2),
            torch.nn.Linear(1024, 512),
            torch.nn.GELU(),
            nn.Dropout(0.2),
            torch.nn.Linear(512, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, 128),
            torch.nn.GELU(),
            L2Norm(dim=1)  
        )
         
        # Building an linear decoder with Linear
        # layer followed by Relu activation function
        # The Sigmoid activation function
        # outputs the value between 0 and 1
        # 9 ==> 784
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(128, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, 512),
            torch.nn.GELU(),
            nn.Dropout(0.2),
            torch.nn.Linear(512, 1024),
            torch.nn.GELU(),
            nn.Dropout(0.2),
            torch.nn.Linear(1024, 1536),
            torch.nn.GELU()
        )
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded



import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Autoencoder(nn.Module):
    def __init__(self):
        super(CNN_Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size=3, stride=1, padding=1),  # Output: (16, 384)
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),    # Output: (16, 192)
            nn.BatchNorm1d(16),
            nn.Dropout(0.2),
            
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),  # Output: (32, 192)
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),      # Output: (32, 96)
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),  # Output: (64, 96)
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),       # Output: (64, 48)
            
            nn.Flatten(),  # Flatten the tensor before the fully connected layer
            nn.Linear(64 * 48, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),  # Activation
            L2Norm(dim=1)  
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, 64 * 48),  # Fully-connected layer to expand dimensions
            nn.GELU(),  # Activation
            nn.Unflatten(1, (64, 48)),  # Unflatten to prepare for transposed convolutions
            
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=1, padding=1),  # Output: (32, 48)
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # Output: (32, 96)
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),

            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=1, padding=1),  # Output: (16, 96)
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # Output: (16, 192)
            nn.BatchNorm1d(16),
            nn.Dropout(0.2),

            nn.ConvTranspose1d(16, 4, kernel_size=3, stride=1, padding=1),  # Output: (4, 192)
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='nearest')  # Final Output: (4, 384)
        )
        
    def forward(self, x):
        x = x.view(-1, 4, 384)  # Reshape the input tensor
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(x.size(0), -1)  # Flatten before output
        return x

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.seq_len, self.n_features = 4, 384
        self.embedding_dim, self.hidden_dim = 128, 256  # Hidden dim is twice the embedding dim
        self.rnn1 = nn.LSTM(
            input_size=self.n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.embedding_dim,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        x, (_, _) = self.rnn1(x)
        x, (hidden_n, _) = self.rnn2(x)
        batch_size = x.shape[0]  # Dynamically get the batch size
                # Normalizing the output
        hidden_n = hidden_n.squeeze(0)
        normalized_hidden_n = hidden_n / hidden_n.norm(dim=1, keepdim=True)
        return normalized_hidden_n.reshape((batch_size, self.embedding_dim))

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.seq_len, self.input_dim = 4, 128
        self.hidden_dim, self.n_features = 256, 384  # Hidden dim is twice the input dim
        self.rnn1 = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.input_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.output_layer = nn.Linear(self.hidden_dim, self.n_features)

    def forward(self, x):
        batch_size = x.shape[0]
        # Initialize a tensor of zeros with the desired output shape: [batch_size, seq_len, input_dim]
        decoder_input = torch.zeros(batch_size, self.seq_len, self.input_dim, device=x.device)
        # Copy the embedding into the first timestep of each sequence
        decoder_input[:, 0, :] = x
        # Pass this tensor through the LSTM layers
        x, (hidden_n, cell_n) = self.rnn1(decoder_input)
        x, (hidden_n, cell_n) = self.rnn2(x)
        # Reshape and apply the output layer
        x = x.reshape((batch_size, self.seq_len, self.hidden_dim))
        return self.output_layer(x)

class RecurrentAutoencoder(nn.Module):
    def __init__(self, device='cuda'):
        super(RecurrentAutoencoder, self).__init__()
        self.encoder = Encoder().to(device)
        self.decoder = Decoder().to(device)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        # Creating a layer of TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        # Stacking 'num_layers' encoder layers
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # Define the transformation before sum pooling
        self.pre_pooling_transform = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )

        # Define the transformation after sum pooling
        self.post_pooling_transform = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 128),
            nn.GELU()
        )

    def forward(self, src):
        """
        src: tensor of shape (batch_size, seq_length, embed_dim)
        """
        output = self.transformer_encoder(src)
        output = self.pre_pooling_transform(output)
        output = output.sum(dim=1)  # sum pooling
        output = self.post_pooling_transform(output)
        output = output / output.norm(dim=1, keepdim=True)
        return output
    
class LSTMDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, seq_length, num_layers=1):
        super(LSTMDecoder, self).__init__()
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(input_size=input_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=num_layers,
                            batch_first=True)

        # Output layer
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Expand the context vector to the sequence length
        x = x.unsqueeze(1).repeat(1, self.seq_length, 1)
        
        # LSTM output
        lstm_out, _ = self.lstm(x)
        
        # Pass through the linear layer and reshape
        output = self.linear(lstm_out)
        return output

class TransformerAutoencoder(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward, num_layers, seq_length, dropout=0.1):
        super(TransformerAutoencoder, self).__init__()
        self.encoder = TransformerEncoder(embed_dim, num_heads, dim_feedforward, num_layers, dropout)
        self.decoder = LSTMDecoder(128, 256, embed_dim, seq_length)  # Adjust hidden_dim as needed

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
