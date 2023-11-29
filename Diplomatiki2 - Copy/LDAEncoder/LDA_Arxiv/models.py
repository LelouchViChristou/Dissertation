import torch.nn as nn

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
    



class LSTMEncoder(nn.Module):
    def __init__(self, embedding_dim, sequence_length, hidden_dim, output_dim, num_layers, dropout):
        super(LSTMEncoder, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True  # Enable bidirectional LSTM
        )
        
        # Define a fully connected layer that maps from the bidirectional LSTM output to the output_dim
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # Times 2 because of bidirectionality
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Concatenate the final forward and backward hidden state
        h_n = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)

        # Dropout for regularization
        h_n = self.dropout(h_n)

        # Fully connected layer
        out = self.fc(h_n)
        
        # Apply GELU activation
        out = F.gelu(out)
        
        # Normalize the output to unit length
        norm = out.norm(p=2, dim=1, keepdim=True)
        normalized_output = out.div(norm)
        return normalized_output


class ConvolutionalEncoder(nn.Module):
    def __init__(self, sequence_length, num_channels, embedding_dim, output_dim=128):
        super(ConvolutionalEncoder, self).__init__()

        self.layers = nn.ModuleList()

        # Convolutional layers
        current_channels = embedding_dim
        for out_channels in num_channels:
            self.layers.append(nn.Conv1d(current_channels, out_channels, kernel_size=3, padding=1))
            self.layers.append(nn.GELU())
            self.layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            self.layers.append(nn.BatchNorm1d(out_channels))
            self.layers.append(nn.Dropout(0.4))
            current_channels = out_channels

        # Size of the output from convolutional layers
        conv_output_size = current_channels * (sequence_length // (2 ** len(num_channels)))

        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 2 * output_dim)
        self.gelu_fc1 = nn.GELU()
        self.fc2 = nn.Linear(2 * output_dim, output_dim)
        self.gelu_fc2 = nn.GELU()

    def forward(self, x):
        # Pass through convolutional layers
        for layer in self.layers:
            x = layer(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)
        x = self.gelu_fc1(x)
        x = self.fc2(x)
        x = self.gelu_fc2(x)

        # Normalize the output
        x = F.normalize(x, p=2, dim=1)

        return x


import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleLinearModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate):
        super(SimpleLinearModel, self).__init__()
        self.layers = nn.ModuleList()
        
        # Add the first layer
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.layers.append(nn.GELU())
        self.layers.append(nn.Dropout(dropout_rate))
        
        # Add any additional layers
        for i in range(1, len(hidden_dims)):
            self.layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            self.layers.append(nn.GELU())
            self.layers.append(nn.Dropout(dropout_rate))
        
        # Add the output layer
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.layers.append(nn.GELU())
    def forward(self, x):
        # Flatten the input
        x = x.view(x.size(0), -1)  # This changes the shape from (256, 4, 384) to (256, 4*384)
        
        for layer in self.layers:
            x = layer(x)
                # Normalize the output
        norm = x.norm(p=2, dim=1, keepdim=True)
        normalized_out = x.div(norm.clamp(min=1e-8))

        return normalized_out




