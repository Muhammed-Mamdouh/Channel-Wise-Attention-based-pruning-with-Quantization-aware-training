import torch
import torch.nn as nn
import torch.nn.functional as F
from quant_utils import linear_quantize_STE, quantized_linear_function, linear_dequantize, reset_scale_and_zero_point, quantized_conv2d_function
from typing import Dict


class Quantized_Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(Quantized_Linear, self).__init__(in_features, out_features, bias=bias)

        self.method = 'normal'  # normal, sym, asym, SAWB,
        self.act_N_bits = None
        self.weight_N_bits = None
        self.input_scale = nn.Parameter(torch.tensor(0.).to(self.weight.device), requires_grad=False)
        self.weight_scale = nn.Parameter(torch.zeros(out_features).to(self.weight.device), requires_grad=False)
        self.decay = .9964
        self.weight_scale_shape = [self.weight.size(0)] + [1] * (self.weight.ndim - 1)

    def forward(self, input):
        if self.method == 'normal':
            # default floating point mode.
            return F.linear(input, self.weight, self.bias)
        else:
            # update scale and zero
            self.__reset_scale_and_zero__(input)
            zero_point_w = torch.zeros_like(self.weight_scale)
            zero_point_I = torch.zeros_like(self.input_scale)
            # compute quantized
            quantized_weight = linear_quantize_STE(self.weight, self.weight_scale.view(self.weight_scale_shape), zero_point_w.view(self.weight_scale_shape), self.weight_N_bits,True)
            quantized_input = linear_quantize_STE(input, self.input_scale, zero_point_I, self.act_N_bits, False)
            if self.bias is None:
                quantized_bias = None
            else:
                quantized_bias = linear_quantize_STE(self.bias, self.weight_scale * self.input_scale, zero_point_w, 32).to(torch.int32)
            # output = quantized_linear_function(quantized_input.to(torch.int32), quantized_weight.to(torch.int32),
            #                                    quantized_bias, self.input_scale, self.weight_scale)


            input_reconstructed = linear_dequantize(quantized_input, self.input_scale, zero_point_I)
            weight_reconstructed = linear_dequantize(quantized_weight, self.weight_scale.view(self.weight_scale_shape), zero_point_w.view(self.weight_scale_shape))
            simulated_output = F.linear(input_reconstructed, weight_reconstructed, self.bias)
            # return output + simulated_output - simulated_output.detach()
            return simulated_output

    def __reset_scale_and_zero__(self, input):
        """
        update scale factor and zero point
            Args:
                input: input feature
            Returns:
        """
        if self.training:
            input_scale_update, _ = reset_scale_unsigned(input, self.act_N_bits)
            input_scale_update = input_scale_update.to(self.weight.device)
            self.input_scale.data -= (1 - self.decay) * (self.input_scale - input_scale_update)
        weight_scale, _ = reset_scale_and_zero_point(self.weight, self.weight_N_bits, self.method)
        self.weight_scale.data = weight_scale.to(self.weight.device)


class Quantized_Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, multiplier=0.4):
        super(Quantized_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                               dilation=dilation, groups=groups, bias=bias)
        self.method = 'normal'  # normal, sym, asym, SAWB,
        self.act_N_bits = None
        self.weight_N_bits = None

        self.input_scale = nn.Parameter(torch.tensor(0.).to(self.weight.device), requires_grad=False)
        self.weight_scale = nn.Parameter(torch.zeros(out_channels).to(self.weight.device), requires_grad=False)
        self.decay = .9964

        self._attention = nn.Parameter((torch.rand(out_channels))/2+multiplier)
        self._attention.requires_grad = True
        self.attention = None
        self.attention = self._attention/torch.max(self._attention)
        self.attended_weight = self.weight * self.attention.view(-1, 1, 1, 1)
        self.weight_scale_shape = [self.attended_weight.size(0)] + [1] * (self.attended_weight.ndim - 1)

    def forward(self, input):
        attention_clone = self._attention.clone().detach()
        quantile_value = torch.torch.max(self._attention)
        self._attention.data = self._attention / quantile_value
        self.attention = self._attention
        self.attended_weight =  self.weight * self.attention.view(-1, 1, 1, 1)

        if self.method == 'normal':
            # default floating point mode.
            return F.conv2d(input, self.attended_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            # update scale and zero
            self.__reset_scale_and_zero__(input)
            zero_point_w = torch.zeros_like(self.weight_scale)
            zero_point_I = torch.zeros_like(self.input_scale)
            # compute quantized
            quantized_weight = linear_quantize_STE(self.attended_weight, self.weight_scale.view(self.weight_scale_shape), zero_point_w.view(self.weight_scale_shape), self.weight_N_bits,True)
            quantized_input = linear_quantize_STE(input, self.input_scale, zero_point_I, self.act_N_bits, False)
            if self.bias is None:
                quantized_bias = None
            else:
                quantized_bias = linear_quantize_STE(self.bias, self.weight_scale * self.input_scale, zero_point_w, 32).to(torch.int32)

            # output = quantized_conv2d_function(quantized_input.to(torch.int32), quantized_weight.to(torch.int32),
            #                                    quantized_bias, self.input_scale, self.weight_scale, self.stride,
            #                                    self.padding, self.dilation, self.groups)

            input_reconstructed = linear_dequantize(quantized_input, self.input_scale, zero_point_I)

            weight_reconstructed = linear_dequantize(quantized_weight, self.weight_scale.view(self.weight_scale_shape), zero_point_w.view(self.weight_scale_shape))

            simulated_output = F.conv2d(input_reconstructed, weight_reconstructed, self.bias, self.stride, self.padding,
                                        self.dilation, self.groups)
            # return output + simulated_output - simulated_output.detach()
            return simulated_output 



    def __reset_scale_and_zero__(self, input):
        """
        update scale factor and zero point
            Args:
                input: input feature
            Returns:
        """
        if self.training:
            input_scale_update, _ = reset_scale_unsigned(input, self.act_N_bits)
            input_scale_update = input_scale_update.to(self.weight.device)
            self.input_scale.data -= (1 - self.decay) * (self.input_scale - input_scale_update)
        weight_scale, _ = reset_scale_and_zero_point(self.attended_weight, self.weight_N_bits, self.method)
        self.weight_scale.data = weight_scale.to(self.weight.device)

# Model to Quant + attention aware
def reset_scale_unsigned(input: torch.tensor, N_bits: int = 4):
    with torch.no_grad():
        zero_point = torch.tensor(0.)
        step_size = torch.max(torch.abs(input)) / ((2**(N_bits))-1)
    return step_size, zero_point

def input_activation_hook(model, data):
    # add hook to record the min max value of the activation
    input_activation = {}
    output_activation = {}

    def add_range_recoder_hook(model):
        import functools
        def _record_range(self, x, y, module_name):
            x = x[0]
            input_activation[module_name] = x.detach()
            output_activation[module_name] = y.detach()

        all_hooks = []
        for name, m in model.named_modules():
            if isinstance(m, (Quantized_Conv2d, Quantized_Linear)):
                all_hooks.append(m.register_forward_hook(
                    functools.partial(_record_range, module_name=name)))
        return all_hooks

    hooks = add_range_recoder_hook(model)
    model(data)

    # remove hooks
    for h in hooks:
        h.remove()
    return input_activation, output_activation


def model_to_quant(model, calibration_loader, act_N_bits=8, weight_N_bits=8, method='sym',
                   device=torch.device("cuda"), bitwidth_dict: Dict = None):
    quantized_model = model
    input_activation, output_activation = input_activation_hook(quantized_model,
                                                                next(iter(calibration_loader))[0].to(device))
    for name, m in quantized_model.named_modules():
        if isinstance(m, Quantized_Conv2d) or isinstance(m, Quantized_Linear):
            if name != 'conv1':
                if bitwidth_dict is None:
                    m.weight_N_bits = weight_N_bits
                else:
                    m.weight_N_bits = bitwidth_dict[name]
                m.act_N_bits = act_N_bits
                m.method = method
                m.input_scale.data, _= reset_scale_unsigned(input_activation[name], m.act_N_bits)

            else:
                m._attention.data = m._attention*0+1.0
                m._attention.requires_grad = False
    return quantized_model
