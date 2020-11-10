import torch
from torch import nn
from torch.nn import functional as F
from machine_learning.data_loading import get_data_loaders_from_dataset_path
from settings import Settings
from torch.nn.utils import weight_norm


class ResidualBlock(nn.Module):
    def __init__(self, sequence_length, kernel_size=3, dropout_rate=0.2, dilation=3, input_channels=1,
                 output_channels=1, stride=1):
        super(ResidualBlock, self).__init__()
        self.output_length = sequence_length
        self.p = ((self.output_length - 1) * stride + 1 + dilation * (kernel_size - 1) - sequence_length) // 2

        self.d_conv_1 = weight_norm(
            nn.Conv1d(in_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size,
                      dilation=dilation, padding=self.p))

        self.relu_1 = nn.ReLU()
        self.dropout_1 = nn.Dropout(dropout_rate)

        self.d_conv_2 = weight_norm(
            nn.Conv1d(in_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size,
                      dilation=dilation, padding=self.p))

        self.relu_2 = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout_rate)

        self.net = nn.Sequential(self.d_conv_1, self.relu_1, self.dropout_1,
                                 self.d_conv_2, self.relu_2, self.dropout_2)

        self.relu = nn.ReLU()
        self.init_weights()

    def forward(self, input_sig):
        output = self.net(input_sig)
        return self.relu(output + input_sig)

    def init_weights(self):
        self.d_conv_1.weight.data.normal_(0, 0.01)
        self.d_conv_2.weight.data.normal_(0, 0.01)


class DCNNClassifier(nn.Module):
    def __init__(self, sequence_length, kernel_size=3, dropout_rate=0.2, number_of_outputs=6, hidden_layers_number=3,
                 input_channels=1, output_channels=1, stride=1):
        super(DCNNClassifier, self).__init__()

        dilations = [2 ** x for x in range(0, hidden_layers_number)]
        layers = []
        for hl_id, dilation in enumerate(dilations):
            layers += [
                ResidualBlock(sequence_length, kernel_size=kernel_size, dropout_rate=dropout_rate, dilation=dilation)]
        self.network = nn.Sequential(*layers)
        self.flatten_layer1 = nn.Flatten()
        self.lin_layer1 = nn.Linear(sequence_length, sequence_length//6)
        self.relu_linear = nn.ReLU()
        self.lin_layer2 = nn.Linear(sequence_length // 6, number_of_outputs)
        self.flatten_layer = nn.Flatten()

    def forward(self, input_signal):
        output = self.network(input_signal)
        output = self.flatten_layer1(output)
        output = self.lin_layer1(output)
        output = self.relu_linear(output)
        output = self.lin_layer2(output)
        return output
