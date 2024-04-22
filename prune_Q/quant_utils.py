import torch
import torch.nn.functional as F



# Quantization functions
def integer_linear(input, weight):
    assert input.dtype == torch.int32
    assert weight.dtype == torch.int32

    if 'cpu' in input.device.type:
        output = F.linear(input, weight)
    else:
        output = F.linear(input.float(), weight.float())
        output = output.to(torch.int32)
    return output
def integer_conv2d(input, weight, stride, padding, dilation, groups):
    assert input.dtype == torch.int32
    assert weight.dtype == torch.int32

    if 'cpu' in input.device.type:
        output = F.conv2d(input, weight, None, stride, padding, dilation, groups)
    else:
        output = F.conv2d(input.float(), weight.float(), None, stride, padding, dilation, groups)
        output = output.to(torch.int32)
    return output


def linear_quantize(input: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor,
                    N_bits: int, signed: bool = True) -> torch.Tensor:
    """
    linear uniform quantization for real tensor
    Args:
        input: torch.tensor
        scale: scale factor
        zero_point: zero point
        N_bits: bitwidth
        signed: flag to indicate signed ot unsigned quantization

    Returns:
        quantized_tensor: quantized tensor whose values are integers
    """
    # Determine the range of the quantized data type
    if signed:
        mini, maxi = -2**(N_bits - 1), 2**(N_bits - 1) - 1
    else:
        mini, maxi = 0, 2**N_bits - 1

    # Quantize the input tensor
    scale[scale==0.0] = 1.0
    input_scaled = input / scale
    input_scaled_rounded = torch.round(input_scaled + zero_point)
    quantized_tensor = torch.clip(input_scaled_rounded, min=mini, max=maxi)

    return quantized_tensor


def linear_dequantize(input: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
    """
    linear uniform de-quantization for quantized tensor
    Args:
        input: input quantized tensor
        scale: scale factor
        zero_point: zero point

    Returns:
        reconstructed_tensor: de-quantized tensor whose values are real
    """
    quantized_tensor = scale*(input + zero_point)
    return quantized_tensor



def get_scale(input, N_bits=8):
    """
    extract optimal scale based on statistics of the input tensor.
    Args:
        input: input real tensor
        N_bits: bitwidth
    Returns:
        scale optimal scale
    """
    assert N_bits in [2, 4, 8]
    z_typical = {'2bit': [0.311, 0.678], '4bit': [0.077, 1.013], '8bit': [0.032, 1.085]}
    z = z_typical[f'{N_bits}bit']
    c1, c2 = 1 / z[0], z[1] / z[0]
    var = torch.mean(input**2)
    m = torch.mean(input.abs())
    alpha = c1*(var.sqrt()) - c2*m
    q_scale = alpha/(2**(N_bits-1))
    return q_scale


def reset_scale_and_zero_point(input: torch.tensor, N_bits: int = 4, method: str = "sym"):
    with torch.no_grad():
        # Reshape input to merge non-channel dimensions
        reshaped_input = input.view(input.size(0), -1)

        if method == 'sym':
            max_vals = torch.max(reshaped_input.abs(), dim=1).values
            scale = max_vals / (2 ** (N_bits - 1))
            zero_point = torch.zeros_like(scale)
        elif method == 'asym':
            min_vals = torch.min(reshaped_input, dim=1).values
            max_vals = torch.max(reshaped_input, dim=1).values
            scale = (max_vals - min_vals) / (2 ** N_bits - 1)
            zero_point = torch.round(min_vals / scale)
        else:
            raise ValueError("Unknown quantization method.")

        # Squeeze to remove extra dimensions if any
        scale = scale.squeeze()
        zero_point = zero_point.squeeze()

    return scale, zero_point



class _quantize_func_STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, zero_point, N_bits, signed=True):
        """
        Args:
            ctx: a context object that can be used to stash information for backward computation
            input: torch.tensor
            scale: scale factor
            zero_point: zero point
            N_bits: bitwidth
            signed: flag to indicate signed ot unsigned quantization
        Returns:
            quantized_tensor: quantized tensor whose values are integers
        """
        ctx.scale = scale
        quantized_tensor = linear_quantize(input, scale, zero_point, N_bits, signed)
        return quantized_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output/ctx.scale
        return grad_input, None, None, None, None

linear_quantize_STE = _quantize_func_STE.apply


def quantized_linear_function(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                              input_scale: torch.float, weight_scale: torch.float):
    """
    integer only fully connected layer.
    Note that you are only allowed to use <integer_linear> function!
    Args:
        input: quantized input
        weight: quantized weight
        bias: quantized bias
        input_scale: input scaling factor
        weight_scale: weight scaling factor

    Returns:
        output: output feature
    """
    bias = 0 if bias is None else bias
    output = input_scale*weight_scale*(integer_linear(input, weight) + bias)
    return output


def quantized_conv2d_function(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                              input_scale: torch.float, weight_scale: torch.float, stride,
                              padding, dilation, groups):
    """
    integer only fully connected layer
    Note that you are only allowed to use <integer_conv2d> function!
    Args:
        groups: number of groups
        stride: stride
        dilation: dilation
        padding: padding
        input: quantized input
        weight: quantized weight
        bias: quantized bias
        input_scale: input scaling factor
        weight_scale: weight scaling factor

    Returns:
        output: output feature
    """
    bias = 0 if bias is None else bias
    output = input_scale*weight_scale.view(1, -1, 1, 1)*(integer_conv2d(input, weight, stride, padding, dilation, groups) + bias)
    return output




