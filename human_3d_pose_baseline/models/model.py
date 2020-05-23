# references:
# https://github.com/weigq/3d_pose_baseline_pytorch/blob/master/src/model.py
# https://github.com/una-dinosauria/3d-pose-baseline/blob/master/src/linear_model.py

import torch.nn as nn


def init_weights(module):
    """Initialize weights of the linear model.

    Our initialization scheme is different from the official implementation in TensorFlow.
    Official one inits bias of linear layer with kaiming normal but we init with 0.
    Also we init weights of batchnorm layer with 1 and bias with 0.
    We have not investigated if this affects the accuracy.

    Args:
        module (torch.nn.Module): torch.nn.Module composing the linear model.
    """
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight.data, mode="fan_in", nonlinearity="relu")
        module.bias.data.zero_()
    if isinstance(module, nn.BatchNorm1d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()


class Linear(nn.Module):
    def __init__(self, linear_size, p_dropout):
        """

        Args:
            linear_size (int): Number of nodes in the linear layers.
            p_dropout (float): Dropout probability.
        """
        super(Linear, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(linear_size, linear_size)
        self.bn1 = nn.BatchNorm1d(linear_size)

        self.w2 = nn.Linear(linear_size, linear_size)
        self.bn2 = nn.BatchNorm1d(linear_size)

    def forward(self, x):
        """Forward operations of the linear block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            y (torch.Tensor): Output tensor.
        """
        h = self.w1(x)
        h = self.bn1(h)
        h = self.relu(h)
        h = self.dropout(h)

        h = self.w2(h)
        h = self.bn2(h)
        h = self.relu(h)
        h = self.dropout(h)

        y = x + h
        return y


class LinearModel(nn.Module):
    def __init__(self, linear_size=1024, num_stages=2, p_dropout=0.5, predict_14=False):
        """

        Args:
            linear_size (int, optional): Number of nodes in the linear layers. Defaults to 1024.
            num_stages (int, optional): Number to repeat the linear block. Defaults to 2.
            p_dropout (float, optional): Dropout probability. Defaults to 0.5.
            predict_14 (bool, optional): Whether to predict 14 3d-joints. Defaults to False.
        """
        super(LinearModel, self).__init__()

        input_size = 16 * 2  # Input 2d-joints.
        output_size = 14 * 3 if predict_14 else 16 * 3  # Output 3d-joints.

        self.w1 = nn.Linear(input_size, linear_size)
        self.bn1 = nn.BatchNorm1d(linear_size)

        self.linear_stages = [Linear(linear_size, p_dropout) for _ in range(num_stages)]
        self.linear_stages = nn.ModuleList(self.linear_stages)

        self.w2 = nn.Linear(linear_size, output_size)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        # initialize model weights
        self.apply(init_weights)

    def forward(self, x):
        """Forward operations of the linear block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            y (torch.Tensor): Output tensor.
        """
        y = self.w1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.dropout(y)

        # linear blocks
        for linear in self.linear_stages:
            y = linear(y)

        y = self.w2(y)

        return y
