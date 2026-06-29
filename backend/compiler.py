"""
Graph compiler: converts Oneiros graph JSON into a PyTorch nn.Module.
No exec() or eval() is used – the model is built entirely from Python objects.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Union


# ── Activation helpers ────────────────────────────────────────────────────────

ACTIVATION_FNS: dict[str, Any] = {
    "relu": F.relu,
    "gelu": F.gelu,
    "silu": F.silu,
    "elu": F.elu,
    "leaky_relu": F.leaky_relu,
    "mish": F.mish,
    "softplus": F.softplus,
    "hardswish": F.hardswish,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "softmax": lambda x: F.softmax(x, dim=-1),
    "none": None,
}


# ── Topological sort ──────────────────────────────────────────────────────────

def topological_sort(nodes: list[dict], edges: list[dict]) -> list[dict] | None:
    node_map = {n["id"]: n for n in nodes}
    in_degree: dict[str, int] = {n["id"]: 0 for n in nodes}
    adj: dict[str, list[str]] = {n["id"]: [] for n in nodes}

    for e in edges:
        adj[e["source"]].append(e["target"])
        in_degree[e["target"]] += 1

    queue = deque(nid for nid, deg in in_degree.items() if deg == 0)
    sorted_nodes: list[dict] = []

    while queue:
        nid = queue.popleft()
        sorted_nodes.append(node_map[nid])
        for neighbor in adj[nid]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return sorted_nodes if len(sorted_nodes) == len(nodes) else None


# ── Shape types ───────────────────────────────────────────────────────────────

@dataclass
class FlatShape:
    kind: Literal["flat"] = "flat"
    features: int = 0


@dataclass
class SpatialShape:
    kind: Literal["spatial"] = "spatial"
    channels: int = 1
    height: int = 1
    width: int = 1


NodeShape = Union[FlatShape, SpatialShape]


def flat_features(shape: NodeShape) -> int:
    if shape.kind == "flat":
        return shape.features
    return shape.channels * shape.height * shape.width


def conv_dim(dim: int, kernel: int, stride: int, padding: int) -> int:
    return max(1, math.floor((dim + 2 * padding - kernel) / stride + 1))


# ── Shape inference ───────────────────────────────────────────────────────────

def infer_shapes(sorted_nodes: list[dict], edges: list[dict]) -> dict[str, NodeShape]:
    parents: dict[str, list[str]] = {n["id"]: [] for n in sorted_nodes}
    for e in edges:
        parents[e["target"]].append(e["source"])

    shapes: dict[str, NodeShape] = {}

    for node in sorted_nodes:
        ntype = node.get("type", "")
        data = node.get("data", {})
        nid = node["id"]
        node_parents = parents[nid]
        first_parent_shape = shapes.get(node_parents[0]) if node_parents else None

        if ntype == "inputNode":
            shapes[nid] = SpatialShape(
                channels=int(data.get("channels", 1)),
                height=int(data.get("height", 28)),
                width=int(data.get("width", 28)),
            )

        elif ntype == "conv2dNode":
            in_c = first_parent_shape.channels if isinstance(first_parent_shape, SpatialShape) else 1
            in_h = first_parent_shape.height if isinstance(first_parent_shape, SpatialShape) else 1
            in_w = first_parent_shape.width if isinstance(first_parent_shape, SpatialShape) else 1
            out_c = int(data.get("outChannels", 32))
            k, s, p = int(data.get("kernelSize", 3)), int(data.get("stride", 1)), int(data.get("padding", 0))
            shapes[nid] = SpatialShape(channels=out_c, height=conv_dim(in_h, k, s, p), width=conv_dim(in_w, k, s, p))

        elif ntype == "conv1dNode":
            out_c = int(data.get("outChannels", 64))
            shapes[nid] = FlatShape(features=out_c)

        elif ntype in ("maxPool2dNode", "avgPool2dNode"):
            in_c = first_parent_shape.channels if isinstance(first_parent_shape, SpatialShape) else 1
            in_h = first_parent_shape.height if isinstance(first_parent_shape, SpatialShape) else 1
            in_w = first_parent_shape.width if isinstance(first_parent_shape, SpatialShape) else 1
            k = int(data.get("kernelSize", 2))
            s = int(data.get("stride", k))
            p = int(data.get("padding", 0))
            shapes[nid] = SpatialShape(channels=in_c, height=conv_dim(in_h, k, s, p), width=conv_dim(in_w, k, s, p))

        elif ntype == "adaptiveAvgPool2dNode":
            in_c = first_parent_shape.channels if isinstance(first_parent_shape, SpatialShape) else 1
            out_sz = int(data.get("outputSize", 1))
            shapes[nid] = FlatShape(features=in_c) if out_sz == 1 else SpatialShape(channels=in_c, height=out_sz, width=out_sz)

        elif ntype == "flattenNode":
            features = flat_features(first_parent_shape) if first_parent_shape else 0
            shapes[nid] = FlatShape(features=features)

        elif ntype in ("dropoutNode", "batchNormNode", "activationNode"):
            if first_parent_shape:
                shapes[nid] = first_parent_shape

        elif ntype in ("rnnNode", "gruNode", "lstmNode"):
            hidden = int(data.get("hiddenSize", 128))
            bidir = bool(data.get("bidirectional", False))
            shapes[nid] = FlatShape(features=hidden * (2 if bidir else 1))

        elif ntype == "transformerEncoderNode":
            shapes[nid] = FlatShape(features=int(data.get("dModel", 256)))

        elif ntype == "denseNode":
            units = int(data.get("units", 128))
            in_features = sum(flat_features(shapes[pid]) for pid in node_parents if pid in shapes)
            shapes[nid] = FlatShape(features=units)
            shapes[f"{nid}__in"] = FlatShape(features=in_features)

        elif ntype == "outputNode":
            classes = int(data.get("classes", 10))
            in_features = sum(flat_features(shapes[pid]) for pid in node_parents if pid in shapes)
            shapes[nid] = FlatShape(features=classes)
            shapes[f"{nid}__in"] = FlatShape(features=in_features)

    return shapes


# ── Operation descriptors ─────────────────────────────────────────────────────

@dataclass
class Op:
    kind: str
    node_id: str
    layer_attr: str | None = None
    activation: str | None = None
    input_node_ids: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)  # node-type-specific extra state


# ── Dynamic graph model ───────────────────────────────────────────────────────

class GraphModel(nn.Module):
    def __init__(self, ops: list[Op]):
        super().__init__()
        self._ops = ops

    def _activate(self, tensor: torch.Tensor, activation: str | None) -> torch.Tensor:
        if not activation:
            return tensor
        fn: Callable | None = ACTIVATION_FNS.get(activation)
        return fn(tensor) if fn is not None else tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var: dict[str, torch.Tensor] = {}

        for op in self._ops:
            if op.kind == "input":
                var[op.node_id] = x

            elif op.kind == "conv2d":
                inp = self._single(var, op.input_node_ids, x)
                layer = getattr(self, op.layer_attr)  # type: ignore[arg-type]
                var[op.node_id] = self._activate(layer(inp), op.activation)

            elif op.kind == "conv1d":
                inp = self._single(var, op.input_node_ids, x)
                layer = getattr(self, op.layer_attr)  # type: ignore[arg-type]
                result = self._activate(layer(inp.unsqueeze(-1)).squeeze(-1), op.activation)
                var[op.node_id] = result

            elif op.kind in ("pool", "avgpool", "adapool"):
                inp = self._single(var, op.input_node_ids, x)
                layer = getattr(self, op.layer_attr)  # type: ignore[arg-type]
                out = layer(inp)
                if op.kind == "adapool" and op.extra.get("flatten"):
                    out = out.view(out.size(0), -1)
                var[op.node_id] = out

            elif op.kind == "flatten":
                inp = self._single(var, op.input_node_ids, x)
                var[op.node_id] = inp.view(inp.size(0), -1)

            elif op.kind in ("dropout", "batchnorm"):
                inp = self._single(var, op.input_node_ids, x)
                layer = getattr(self, op.layer_attr)  # type: ignore[arg-type]
                var[op.node_id] = layer(inp)

            elif op.kind == "activation":
                inp = self._single(var, op.input_node_ids, x)
                var[op.node_id] = self._activate(inp, op.activation)

            elif op.kind in ("rnn", "gru"):
                inp = self._single(var, op.input_node_ids, x)
                rnn_layer = getattr(self, op.layer_attr)  # type: ignore[arg-type]
                _, hn = rnn_layer(inp.unsqueeze(1))
                bidir = op.extra.get("bidirectional", False)
                var[op.node_id] = hn.transpose(0, 1).reshape(inp.size(0), -1) if bidir else hn[-1]

            elif op.kind == "lstm":
                inp = self._single(var, op.input_node_ids, x)
                lstm_layer = getattr(self, op.layer_attr)  # type: ignore[arg-type]
                _, (hn, _) = lstm_layer(inp.unsqueeze(1))
                bidir = op.extra.get("bidirectional", False)
                var[op.node_id] = hn.transpose(0, 1).reshape(inp.size(0), -1) if bidir else hn[-1]

            elif op.kind == "transformer_encoder":
                inp = self._single(var, op.input_node_ids, x)
                proj = getattr(self, op.layer_attr)  # type: ignore[arg-type]
                enc = getattr(self, op.extra["encoder_attr"])
                var[op.node_id] = enc(proj(inp).unsqueeze(1)).squeeze(1)

            elif op.kind == "linear":
                inp = self._gather(var, op.input_node_ids, x)
                layer = getattr(self, op.layer_attr)  # type: ignore[arg-type]
                var[op.node_id] = self._activate(layer(inp), op.activation)

            elif op.kind == "output":
                inp = self._gather(var, op.input_node_ids, x)
                layer = getattr(self, op.layer_attr)  # type: ignore[arg-type]
                var[op.node_id] = layer(inp)

        return var[self._ops[-1].node_id]

    @staticmethod
    def _single(var: dict[str, torch.Tensor], node_ids: list[str], fallback: torch.Tensor) -> torch.Tensor:
        if not node_ids:
            return fallback
        return var.get(node_ids[0], fallback)

    @staticmethod
    def _gather(var: dict[str, torch.Tensor], node_ids: list[str], fallback: torch.Tensor) -> torch.Tensor:
        if not node_ids:
            return fallback.view(fallback.size(0), -1)
        tensors = [var.get(nid, fallback).view(var.get(nid, fallback).size(0), -1) for nid in node_ids]
        return tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=-1)


# ── Public entry point ────────────────────────────────────────────────────────

def build_model(graph: dict) -> tuple[GraphModel, list[str]]:
    nodes: list[dict] = graph.get("nodes", [])
    edges: list[dict] = graph.get("edges", [])

    if not nodes:
        return GraphModel([]), ["Graph is empty"]

    sorted_nodes = topological_sort(nodes, edges)
    if sorted_nodes is None:
        return GraphModel([]), ["Cycle detected in graph"]

    shapes = infer_shapes(sorted_nodes, edges)

    parents: dict[str, list[str]] = {n["id"]: [] for n in sorted_nodes}
    for e in edges:
        parents[e["target"]].append(e["source"])

    model = GraphModel.__new__(GraphModel)
    nn.Module.__init__(model)
    ops: list[Op] = []

    for node in sorted_nodes:
        ntype = node.get("type", "")
        nid = node["id"]
        safe_id = nid.replace("-", "_")
        data = node.get("data", {})
        node_parents = parents[nid]
        first_parent_shape = shapes.get(node_parents[0]) if node_parents else None

        if ntype == "inputNode":
            ops.append(Op(kind="input", node_id=nid))

        elif ntype == "conv2dNode":
            in_c = first_parent_shape.channels if isinstance(first_parent_shape, SpatialShape) else 1
            out_c = int(data.get("outChannels", 32))
            k, s, p = int(data.get("kernelSize", 3)), int(data.get("stride", 1)), int(data.get("padding", 0))
            dil = int(data.get("dilation", 1))
            grp = int(data.get("groups", 1))
            bias = bool(data.get("bias", True))
            pmode = str(data.get("paddingMode", "zeros"))
            attr = f"conv_{safe_id}"
            setattr(model, attr, nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p,
                                           dilation=dil, groups=grp, bias=bias, padding_mode=pmode))
            ops.append(Op(kind="conv2d", node_id=nid, layer_attr=attr, activation=str(data.get("activation", "relu")), input_node_ids=node_parents))

        elif ntype == "conv1dNode":
            in_f = flat_features(first_parent_shape) if first_parent_shape else 1
            out_c = int(data.get("outChannels", 64))
            k, s, p = int(data.get("kernelSize", 3)), int(data.get("stride", 1)), int(data.get("padding", 1))
            dil = int(data.get("dilation", 1))
            grp = int(data.get("groups", 1))
            bias = bool(data.get("bias", True))
            attr = f"conv1d_{safe_id}"
            setattr(model, attr, nn.Conv1d(in_f, out_c, kernel_size=k, stride=s, padding=p,
                                           dilation=dil, groups=grp, bias=bias))
            ops.append(Op(kind="conv1d", node_id=nid, layer_attr=attr, activation=str(data.get("activation", "relu")), input_node_ids=node_parents))

        elif ntype == "maxPool2dNode":
            k = int(data.get("kernelSize", 2))
            s = int(data.get("stride", k))
            pad = int(data.get("padding", 0))
            dil = int(data.get("dilation", 1))
            ceil_mode = bool(data.get("ceilMode", False))
            attr = f"pool_{safe_id}"
            setattr(model, attr, nn.MaxPool2d(kernel_size=k, stride=s, padding=pad, dilation=dil, ceil_mode=ceil_mode))
            ops.append(Op(kind="pool", node_id=nid, layer_attr=attr, input_node_ids=node_parents))

        elif ntype == "avgPool2dNode":
            k = int(data.get("kernelSize", 2))
            s = int(data.get("stride", k))
            pad = int(data.get("padding", 0))
            ceil_mode = bool(data.get("ceilMode", False))
            count_pad = bool(data.get("countIncludePad", True))
            attr = f"avgpool_{safe_id}"
            setattr(model, attr, nn.AvgPool2d(kernel_size=k, stride=s, padding=pad,
                                              ceil_mode=ceil_mode, count_include_pad=count_pad))
            ops.append(Op(kind="avgpool", node_id=nid, layer_attr=attr, input_node_ids=node_parents))

        elif ntype == "adaptiveAvgPool2dNode":
            out_sz = int(data.get("outputSize", 1))
            attr = f"adapool_{safe_id}"
            setattr(model, attr, nn.AdaptiveAvgPool2d(out_sz))
            ops.append(Op(kind="adapool", node_id=nid, layer_attr=attr, input_node_ids=node_parents, extra={"flatten": out_sz == 1}))

        elif ntype == "flattenNode":
            ops.append(Op(kind="flatten", node_id=nid, input_node_ids=node_parents))

        elif ntype == "dropoutNode":
            attr = f"drop_{safe_id}"
            setattr(model, attr, nn.Dropout(p=float(data.get("p", 0.5))))
            ops.append(Op(kind="dropout", node_id=nid, layer_attr=attr, input_node_ids=node_parents))

        elif ntype == "batchNormNode":
            eps = float(data.get("eps", 1e-5))
            mom = float(data.get("momentum", 0.1))
            affine = bool(data.get("affine", True))
            track = bool(data.get("trackRunningStats", True))
            attr = f"bn_{safe_id}"
            if isinstance(first_parent_shape, SpatialShape):
                setattr(model, attr, nn.BatchNorm2d(first_parent_shape.channels, eps=eps, momentum=mom,
                                                    affine=affine, track_running_stats=track))
            else:
                nf = flat_features(first_parent_shape) if first_parent_shape else 1
                setattr(model, attr, nn.BatchNorm1d(nf, eps=eps, momentum=mom,
                                                    affine=affine, track_running_stats=track))
            ops.append(Op(kind="batchnorm", node_id=nid, layer_attr=attr, input_node_ids=node_parents))

        elif ntype == "activationNode":
            ops.append(Op(kind="activation", node_id=nid, activation=str(data.get("fn", "relu")), input_node_ids=node_parents))

        elif ntype == "rnnNode":
            in_f = flat_features(first_parent_shape) if first_parent_shape else 1
            hidden = int(data.get("hiddenSize", 128))
            layers = int(data.get("numLayers", 1))
            nonlin = str(data.get("nonlinearity", "tanh"))
            drop = float(data.get("dropout", 0))
            bidir = bool(data.get("bidirectional", False))
            attr = f"rnn_{safe_id}"
            setattr(model, attr, nn.RNN(in_f, hidden, num_layers=layers, nonlinearity=nonlin, dropout=drop, bidirectional=bidir, batch_first=True))
            ops.append(Op(kind="rnn", node_id=nid, layer_attr=attr, input_node_ids=node_parents, extra={"bidirectional": bidir}))

        elif ntype == "gruNode":
            in_f = flat_features(first_parent_shape) if first_parent_shape else 1
            hidden = int(data.get("hiddenSize", 128))
            layers = int(data.get("numLayers", 1))
            drop = float(data.get("dropout", 0))
            bidir = bool(data.get("bidirectional", False))
            attr = f"gru_{safe_id}"
            setattr(model, attr, nn.GRU(in_f, hidden, num_layers=layers, dropout=drop, bidirectional=bidir, batch_first=True))
            ops.append(Op(kind="gru", node_id=nid, layer_attr=attr, input_node_ids=node_parents, extra={"bidirectional": bidir}))

        elif ntype == "lstmNode":
            in_f = flat_features(first_parent_shape) if first_parent_shape else 1
            hidden = int(data.get("hiddenSize", 128))
            layers = int(data.get("numLayers", 1))
            drop = float(data.get("dropout", 0))
            bidir = bool(data.get("bidirectional", False))
            proj = int(data.get("projSize", 0))
            attr = f"lstm_{safe_id}"
            setattr(model, attr, nn.LSTM(in_f, hidden, num_layers=layers, dropout=drop,
                                         bidirectional=bidir, batch_first=True, proj_size=proj))
            ops.append(Op(kind="lstm", node_id=nid, layer_attr=attr, input_node_ids=node_parents, extra={"bidirectional": bidir}))

        elif ntype == "transformerEncoderNode":
            in_f = flat_features(first_parent_shape) if first_parent_shape else 1
            d_model = int(data.get("dModel", 256))
            nhead = int(data.get("nhead", 8))
            ff_dim = int(data.get("dimFeedforward", 512))
            num_layers = int(data.get("numLayers", 2))
            drop = float(data.get("dropout", 0.1))
            proj_attr = f"te_proj_{safe_id}"
            enc_attr = f"te_{safe_id}"
            setattr(model, proj_attr, nn.Linear(in_f, d_model))
            te_act = str(data.get("activation", "relu"))
            norm_first = bool(data.get("normFirst", False))
            enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=ff_dim,
                                                   dropout=drop, activation=te_act, norm_first=norm_first,
                                                   batch_first=True)
            setattr(model, enc_attr, nn.TransformerEncoder(enc_layer, num_layers=num_layers))
            ops.append(Op(kind="transformer_encoder", node_id=nid, layer_attr=proj_attr, input_node_ids=node_parents, extra={"encoder_attr": enc_attr}))

        elif ntype == "denseNode":
            in_shape = shapes.get(f"{nid}__in")
            out_shape = shapes.get(nid)
            if in_shape is None or out_shape is None:
                continue
            attr = f"layer_{safe_id}"
            setattr(model, attr, nn.Linear(flat_features(in_shape), flat_features(out_shape)))
            ops.append(Op(kind="linear", node_id=nid, layer_attr=attr, activation=str(data.get("activation", "relu")), input_node_ids=node_parents))

        elif ntype == "outputNode":
            in_shape = shapes.get(f"{nid}__in")
            out_shape = shapes.get(nid)
            if in_shape is None or out_shape is None:
                continue
            attr = f"layer_{safe_id}"
            setattr(model, attr, nn.Linear(flat_features(in_shape), flat_features(out_shape)))
            ops.append(Op(kind="output", node_id=nid, layer_attr=attr, input_node_ids=node_parents))

    model._ops = ops
    return model, []
