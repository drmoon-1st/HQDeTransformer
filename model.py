import math
from typing import Callable, Optional, Union

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, LayerNorm, Linear, Module, TransformerEncoder
from torch.nn.modules.transformer import _get_activation_fn


def u_su4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires):
    qml.U3(*weights_0, wires=wires[0])
    qml.U3(*weights_1, wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(weights_2, wires=wires[0])
    qml.RZ(weights_3, wires=wires[1]])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RY(weights_4, wires=wires[0])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.U3(*weights_5, wires=wires[0])
    qml.U3(*weights_6, wires=wires[1])


def build_weight_shapes(n_layers: int, n_wires: int):
    return {
        "weights_0": 3,
        "weights_1": 3,
        "weights_2": 1,
        "weights_3": 1,
        "weights_4": 1,
        "weights_5": 3,
        "weights_6": 3,
        "weights_7": 3,
        "weights_8": 3,
        "weights_9": 1,
        "weights_10": 1,
        "weights_11": 1,
        "weights_12": 3,
        "weights_13": 3,
        "stornlg": qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_wires),
    }


def apply_conv_block(weights_a, weights_b, wires_list):
    for wires in wires_list:
        u_su4(*weights_a, wires=wires)
    for wires in wires_list:
        u_su4(*weights_b, wires=wires)


def build_quantum_layer(
    n_wires: int,
    n_layers: int,
    embedding: str,
    device_name: str,
    interface: Optional[str] = None,
):
    weight_shapes = build_weight_shapes(n_layers=n_layers, n_wires=n_wires)
    dev = qml.device(device_name, wires=n_wires)

    ring_wires = [[i, (i + 1) % n_wires] for i in range(n_wires)]

    @qml.qnode(dev, interface=interface)
    def qnode(
        inputs,
        weights_0,
        weights_1,
        weights_2,
        weights_3,
        weights_4,
        weights_5,
        weights_6,
        weights_7,
        weights_8,
        weights_9,
        weights_10,
        weights_11,
        weights_12,
        weights_13,
        stornlg,
    ):
        if embedding == "angle":
            qml.AngleEmbedding(inputs, wires=range(n_wires))
        elif embedding == "amplitude":
            qml.AmplitudeEmbedding(inputs, wires=range(n_wires), normalize=True)
        else:
            raise ValueError(f"Unsupported embedding type: {embedding}")

        weights_first = (
            weights_0,
            weights_1,
            weights_2,
            weights_3,
            weights_4,
            weights_5,
            weights_6,
        )
        weights_second = (
            weights_7,
            weights_8,
            weights_9,
            weights_10,
            weights_11,
            weights_12,
            weights_13,
        )

        apply_conv_block(weights_first, weights_second, ring_wires)
        qml.StronglyEntanglingLayers(weights=stornlg, wires=range(n_wires))

        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_wires)]

    qlayer = qml.qnn.TorchLayer(qnode, weight_shapes).cpu()

    for p in qlayer.parameters():
        p.requires_grad = False

    return qlayer


class MLPBlock(nn.Module):
    def __init__(self, d_input, d_hidden, d_output):
        super().__init__()
        self.qlayer = build_quantum_layer(
            n_wires=4,
            n_layers=15,
            embedding="amplitude",
            device_name="default.qubit",
        )
        self.linear1 = nn.Linear(d_input, d_hidden)
        self.norm = nn.LayerNorm(d_hidden)
        self.linear2 = nn.Linear(4, d_output)

    def forward(self, x):
        device = x.device
        x = self.linear1(x)
        x = self.norm(x)
        x = F.relu(x)
        with torch.no_grad():
            x = self.qlayer(x.to("cpu")).to(device)
        x = self.linear2(x)
        return x


class MLPAttention(nn.Module):
    def __init__(self, d_model, nhead, hidden_dim=None, batch_first=True):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.batch_first = batch_first

        hidden_dim = hidden_dim or d_model

        self.mlp_q = MLPBlock(d_model, hidden_dim, d_model)
        self.mlp_k = MLPBlock(d_model, hidden_dim, d_model)
        self.mlp_v = MLPBlock(d_model, hidden_dim, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

        self.in_proj_weight = None
        self.in_proj_bias = None
        self._qkv_same_embed_dim = True
        self.num_heads = nhead

    def forward(self, x, mask=None, key_padding_mask=None):
        B, T, D = x.size()

        Q = self.mlp_q(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        K = self.mlp_k(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        V = self.mlp_v(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask[:, None, None, :],
                float("-inf"),
            )

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)

        return self.out_proj(attn_output)


class TransformerEncoderLayer(Module):
    __constants__ = ["norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.linear1 = Linear(d_model, 4, bias=bias, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(4, d_model, bias=bias, **factory_kwargs)

        self.qlayerB = build_quantum_layer(
            n_wires=4,
            n_layers=15,
            embedding="angle",
            device_name="default.qubit",
        )

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.self_attn = MLPAttention(d_model, nhead, batch_first=batch_first)

        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0

        self.activation = activation

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, "activation"):
            self.activation = F.relu

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype,
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        is_fastpath_enabled = torch.backends.mha.get_fastpath_enabled()

        why_not_sparsity_fast_path = ""
        if not is_fastpath_enabled:
            why_not_sparsity_fast_path = "torch.backends.mha.get_fastpath_enabled() was not True"
        elif not src.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        elif self.training:
            why_not_sparsity_fast_path = "training is enabled"
        elif not self.self_attn.batch_first:
            why_not_sparsity_fast_path = "self_attn.batch_first was not True"
        elif self.self_attn.in_proj_bias is None:
            why_not_sparsity_fast_path = "self_attn was passed bias=False"
        elif not self.self_attn._qkv_same_embed_dim:
            why_not_sparsity_fast_path = "self_attn._qkv_same_embed_dim was not True"
        elif not self.activation_relu_or_gelu:
            why_not_sparsity_fast_path = "activation_relu_or_gelu was not True"
        elif not (self.norm1.eps == self.norm2.eps):
            why_not_sparsity_fast_path = "norm1.eps is not equal to norm2.eps"
        elif src.is_nested and (src_key_padding_mask is not None or src_mask is not None):
            why_not_sparsity_fast_path = "neither src_key_padding_mask nor src_mask are not supported with NestedTensor input"
        elif self.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = "num_head is odd"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"
        elif any(
            len(getattr(m, "_forward_hooks", {})) + len(getattr(m, "_forward_pre_hooks", {}))
            for m in self.modules()
        ):
            why_not_sparsity_fast_path = "forward pre-/hooks are attached to the module"

        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                self.self_attn.in_proj_weight,
                self.self_attn.in_proj_bias,
                self.self_attn.out_proj.weight,
                self.self_attn.out_proj.bias,
                self.norm1.weight,
                self.norm1.bias,
                self.norm2.weight,
                self.norm2.bias,
                self.linear1.weight,
                self.linear1.bias,
                self.linear2.weight,
                self.linear2.bias,
            )

            _supported_device_type = [
                "cpu",
                "cuda",
                torch.utils.backend_registration._privateuse1_backend_name,
            ]

            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif not all((x.device.type in _supported_device_type) for x in tensor_args if x is not None):
                why_not_sparsity_fast_path = f"some Tensor argument's device is neither one of {_supported_device_type}"
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args if x is not None):
                why_not_sparsity_fast_path = "grad is enabled and at least one of query or the input/output projection weights or biases requires_grad"

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal))
            x = self.norm2(x + self._ff_block(x))

        return x

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        x = self.self_attn(x, mask=attn_mask, key_padding_mask=key_padding_mask)
        return self.dropout1(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        device = x.device
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        with torch.no_grad():
            x = self.qlayerB(x.to("cpu")).to(device)
        x = self.linear2(x)
        return self.dropout2(x)


class Autoencoder(nn.Module):
    def __init__(self, input_size=16, hidden_dim=8, noise_level=0.01):
        super().__init__()
        self.noise_level = noise_level

        self.qlayerB = build_quantum_layer(
            n_wires=3,
            n_layers=1,
            embedding="amplitude",
            device_name="default.mixed",
            interface="torch",
        )
        self.qlayerC = build_quantum_layer(
            n_wires=3,
            n_layers=1,
            embedding="amplitude",
            device_name="default.mixed",
            interface="torch",
        )

        self.fc1 = nn.Linear(input_size, 8, bias=False)
        self.fc3 = nn.Linear(3, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, 8)
        self.fc4 = nn.Linear(3, input_size)

    def encoder(self, x):
        x = self.fc1(x)
        x = self.qlayerB(x)
        x = self.fc3(x)
        x = F.relu(x)
        return x

    def mask(self, x):
        return x + self.noise_level * torch.randn_like(x)

    def decoder(self, x):
        x = self.fc2(x)
        x = self.qlayerC(x)
        x = self.fc4(x)
        return x

    def forward(self, x):
        out = self.mask(x)
        encode = self.encoder(out)
        decode = self.decoder(encode)
        return encode, decode


class PositionalEncoding(nn.Module):
    def __init__(self, feature_len, feature_size, dropout=0.0):
        super().__init__()

        pe = torch.zeros(feature_len, feature_size)
        position = torch.arange(0, feature_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, feature_size, 2).float() * (-math.log(10000.0) / feature_size)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe


class Net(nn.Module):
    def __init__(
        self,
        feature_size=16,
        hidden_dim=32,
        feature_num=1,
        num_layers=1,
        nhead=1,
        dropout=0.0,
        noise_level=0.01,
    ):
        super().__init__()
        self.auto_hidden = int(feature_size / 2)
        input_size = self.auto_hidden

        if feature_num == 1:
            self.pos = PositionalEncoding(feature_len=feature_num, feature_size=input_size)
            encoder_layers = TransformerEncoderLayer(
                d_model=input_size,
                nhead=nhead,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True,
            )
        else:
            self.pos = PositionalEncoding(feature_len=input_size, feature_size=feature_num)
            encoder_layers = TransformerEncoderLayer(
                d_model=feature_num,
                nhead=nhead,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True,
            )

        self.cell = TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.linear = nn.Linear(feature_num * self.auto_hidden, 1)
        self.autoencoder = Autoencoder(
            input_size=feature_size,
            hidden_dim=self.auto_hidden,
            noise_level=noise_level,
        )

    def forward(self, x):
        batch_size, feature_num, feature_size = x.shape
        out, decode = self.autoencoder(x)

        if feature_num > 1:
            out = out.reshape(batch_size, -1, feature_num)

        out = self.pos(out)
        out = self.cell(out)
        out = out.reshape(batch_size, -1)
        out = self.linear(out)

        return out, decode
