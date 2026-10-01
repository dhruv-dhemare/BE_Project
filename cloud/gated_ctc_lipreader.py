import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class GatedCTCLipReader(nn.Module):
    def __init__(self, vocab_size=28, landmark_dropout=0.2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
        )
        self.landmark_mlp = nn.Sequential(
            nn.Linear(80, 128), nn.ReLU(), nn.Dropout(landmark_dropout), nn.Linear(128, 64), nn.ReLU()
        )
        self.visual_proj = nn.Linear(128, 128)
        self.landmark_proj = nn.Linear(64, 128)
        self.gate = nn.Sequential(nn.Linear(256, 128), nn.Sigmoid())
        self.lstm = nn.LSTM(128, 128, num_layers=2, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(256, vocab_size)

    def forward(self, video, landmarks, lengths):
        batch, time, channels, height, width = video.shape
        visual = self.cnn(video.reshape(batch * time, channels, height, width))
        visual = visual.reshape(batch, time, 128)
        geometry = self.landmark_mlp(landmarks.reshape(batch * time, -1)).reshape(batch, time, 64)
        visual = self.visual_proj(visual)
        geometry = self.landmark_proj(geometry)
        gate = self.gate(torch.cat([visual, geometry], dim=-1))
        fused = gate * visual + (1 - gate) * geometry
        packed = pack_padded_sequence(fused, lengths.cpu(), batch_first=True, enforce_sorted=False)
        encoded, _ = self.lstm(packed)
        encoded, _ = pad_packed_sequence(encoded, batch_first=True)
        return self.classifier(encoded)
