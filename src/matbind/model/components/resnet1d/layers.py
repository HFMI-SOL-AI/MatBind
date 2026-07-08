import math

import torch.nn as nn
import torch.nn.functional as F


class SamePaddingBase(nn.Module):
    """Base class to implement 'SAME' padding functionality."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def compute_same_padding(input_width, kernel_size, stride, dilation=1):
        effective_kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
        output_width = math.ceil(input_width / stride)
        required_padding_total = max(
            0, (output_width - 1) * stride + effective_kernel_size - input_width
        )
        padding_left = required_padding_total // 2
        padding_right = required_padding_total - padding_left
        return padding_left, padding_right


class Conv1dSamePadding(SamePaddingBase):
    """Class that mimics Tensorflow's 'SAME' padding for Convolution1D."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.stride = stride
        self.kernel_size = kernel_size
        self.dilation = dilation

    def forward(self, x):
        padding_left, padding_right = self.compute_same_padding(
            x.size(-1), self.kernel_size, self.stride, self.dilation
        )
        x = nn.functional.pad(x, (padding_left, padding_right))
        return self.conv(x)


class AvgPool1dSamePadding(SamePaddingBase):
    """Class that mimics Tensorflow's 'SAME' padding for AvgPool1d."""

    def __init__(
        self,
        kernel_size,
        stride=None,
        ceil_mode=False,
        count_include_pad=True,
    ):
        super().__init__()

        self.avgpool = nn.AvgPool1d(
            kernel_size,
            stride=stride,
            padding=0,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
        )
        self.stride = stride or kernel_size
        self.kernel_size = kernel_size

    def forward(self, x):
        padding_left, padding_right = self.compute_same_padding(
            x.size(-1), self.kernel_size, self.stride
        )
        x = nn.functional.pad(x, (padding_left, padding_right))
        return self.avgpool(x)


class MaxPool1DWithSamePadding(SamePaddingBase):
    """Class that mimics Tensorflow's 'SAME' padding for MaxPool1D."""

    def __init__(
        self,
        kernel_size,
        stride=None,
        dilation=1,
    ):
        super().__init__()

        self.pool = nn.MaxPool1d(
            kernel_size,
            stride,
            padding=0,
            dilation=dilation,
        )
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.dilation = dilation

    def forward(self, x):
        pad_left, pad_right = self.compute_same_padding(
            len(x[0]), self.kernel_size, self.stride, self.dilation
        )
        x = F.pad(x, (pad_left, pad_right))
        return self.pool(x)
