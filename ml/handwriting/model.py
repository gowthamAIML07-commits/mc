"""CRNN / CNN-BiLSTM Deep Neural Architecture for Handwritten Medicine Recognition."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CRNNHandwritingClassifier(nn.Module):
    """Convolutional Recurrent Neural Network (CRNN) for handwritten medicine recognition."""

    def __init__(self, num_classes: int = 78, in_channels: int = 3, rnn_hidden: int = 128):
        super().__init__()
        self.num_classes = num_classes
        self.rnn_hidden = rnn_hidden

        # CNN Feature Extractor
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # (32, H/2, W/2)

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # (64, H/4, W/4)

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)), # (128, H/8, W/4)

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 24)) # (256, 1, 24)
        )

        # Sequence Recurrent Encoder
        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=rnn_hidden,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.2
        )

        # Classification & Projection Heads
        self.fc = nn.Sequential(
            nn.Linear(rnn_hidden * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, H, W)
        conv = self.features(x) # (B, 256, 1, 24)
        conv = conv.squeeze(2).permute(0, 2, 1) # (B, 24, 256)
        
        rnn_out, _ = self.rnn(conv) # (B, 24, 256)
        
        # Global temporal pooling over sequence length
        pooled = torch.mean(rnn_out, dim=1) # (B, 256)
        logits = self.fc(pooled) # (B, num_classes)
        return logits
