"""Local implementation of the aggregation equations in Pei et al., CIKM 2024.

The caller supplies independent drug/protein representations and the frozen
affinity head that consumes them. This module implements both representation
and label aggregation. It does not substitute for the paper's encoder weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn


def _matrix(value: Tensor, name: str) -> None:
    if value.ndim != 2 or not value.shape[0] or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be a nonempty finite matrix")
    if not value.is_floating_point():
        raise ValueError(f"{name} must have floating-point dtype")


def _unique_rows(values: Tensor, keys: Sequence[str]) -> Tensor:
    if len(values) != len(keys):
        raise ValueError("Entity keys and embedding rows differ")
    positions: dict[str, int] = {}
    inverse = []
    for position, key in enumerate(keys):
        if key not in positions:
            positions[key] = position
        inverse.append(positions[key])
    if not torch.allclose(values, values[inverse], atol=1e-6, rtol=1e-5):
        raise ValueError("One entity has inconsistent independent representations")
    return values[list(positions.values())]


@dataclass(frozen=True)
class Neighbors:
    distances: Tensor
    values: Tensor
    indices: Tensor

    def first(self, k: int) -> "Neighbors":
        if k < 1:
            raise ValueError("k must be positive")
        k = min(k, self.distances.shape[1])
        return Neighbors(self.distances[:, :k], self.values[:, :k], self.indices[:, :k])


@dataclass(frozen=True)
class Query:
    drug: Tensor
    protein: Tensor
    labels: Neighbors
    drug_neighbors: Neighbors
    protein_neighbors: Neighbors


def nearest(query: Tensor, keys: Tensor, values: Tensor, k: int,
            chunk_size: int = 128) -> Neighbors:
    """Exact squared-L2 search with deterministic ordering of tied distances."""
    _matrix(query, "query")
    _matrix(keys, "keys")
    if k < 1 or chunk_size < 1 or len(keys) != len(values):
        raise ValueError("Invalid k, chunk size, or datastore dimensions")
    if query.shape[1] != keys.shape[1] or query.device != keys.device:
        raise ValueError("Query and datastore dimensions/devices differ")
    if query.dtype != keys.dtype or keys.device != values.device:
        raise ValueError("Datastore dtype/device mismatch")
    if not torch.isfinite(values).all():
        raise ValueError("Datastore values contain nonfinite entries")
    indices, distances = [], []
    with torch.no_grad():
        for start in range(0, len(query), chunk_size):
            # Direct Euclidean accumulation avoids cancellation near exact matches.
            dist = torch.cdist(query[start:start + chunk_size], keys, p=2,
                               compute_mode="donot_use_mm_for_euclid_dist").square()
            idx = torch.argsort(dist, dim=1, stable=True)[:, :min(k, len(keys))].contiguous()
            indices.append(idx)
            distances.append(dist.gather(1, idx))
        index = torch.cat(indices)
        return Neighbors(torch.cat(distances), values[index].detach(), index)


class TrainingStore:
    def __init__(self, drug: Tensor, protein: Tensor, labels: Tensor,
                 drug_keys: Sequence[str], protein_keys: Sequence[str],
                 row_ids: Sequence[int], splits: Sequence[str]):
        _matrix(drug, "drug")
        _matrix(protein, "protein")
        if not (len(drug) == len(protein) == len(labels) == len(row_ids) == len(splits)):
            raise ValueError("Datastore row counts differ")
        if labels.ndim != 1 or not torch.isfinite(labels).all():
            raise ValueError("Labels must be a finite vector")
        if set(splits) != {"train"}:
            raise ValueError("Only training rows may enter a retrieval datastore")
        if len(set(row_ids)) != len(row_ids):
            raise ValueError("Datastore contains duplicate row IDs")
        if drug.device != protein.device or drug.device != labels.device:
            raise ValueError("Datastore tensors must be on the same device")
        self.drug = _unique_rows(drug.detach(), drug_keys)
        self.protein = _unique_rows(protein.detach(), protein_keys)
        self.pairs = torch.cat((drug, protein), dim=1).detach()
        self.labels = labels.detach()
        self.row_ids = tuple(row_ids)
        drug_lookup = {key: i for i, key in enumerate(dict.fromkeys(drug_keys))}
        protein_lookup = {key: i for i, key in enumerate(dict.fromkeys(protein_keys))}
        self.drug_index = torch.tensor([drug_lookup[x] for x in drug_keys], device=drug.device)
        self.protein_index = torch.tensor([protein_lookup[x] for x in protein_keys], device=protein.device)

    def retrieve(self, drug: Tensor, protein: Tensor, pair_k: int,
                 drug_k: int, protein_k: int) -> Query:
        if len(drug) != len(protein):
            raise ValueError("Query drug/protein counts differ")
        if min(pair_k, drug_k, protein_k) < 1:
            raise ValueError("Neighbor counts must be positive")
        # The squared distance of concatenated vectors is the sum of the two
        # entity distances. Reusing these distances avoids repeating work for
        # every measured pair, while keeping exact search over training pairs.
        pair_indices, pair_distances = [], []
        with torch.no_grad():
            for start in range(0, len(drug), 128):
                dd = torch.cdist(drug[start:start + 128], self.drug,
                                 compute_mode="donot_use_mm_for_euclid_dist").square()
                pd = torch.cdist(protein[start:start + 128], self.protein,
                                 compute_mode="donot_use_mm_for_euclid_dist").square()
                distances = dd[:, self.drug_index] + pd[:, self.protein_index]
                order = torch.argsort(distances, dim=1, stable=True)[:, :min(pair_k, len(self.labels))].contiguous()
                pair_indices.append(order)
                pair_distances.append(distances.gather(1, order))
            indices = torch.cat(pair_indices)
            labels = Neighbors(torch.cat(pair_distances), self.labels[indices], indices)
        return Query(
            drug.detach(), protein.detach(),
            labels,
            nearest(drug, self.drug, self.drug, drug_k),
            nearest(protein, self.protein, self.protein, protein_k),
        )


@dataclass(frozen=True)
class Mix:
    k: int
    temperature: float
    base_weight: float

    def __post_init__(self):
        if self.k < 1 or not 0 < self.temperature < float("inf"):
            raise ValueError("k and temperature must be positive and finite")
        if not 0 <= self.base_weight <= 1:
            raise ValueError("Base weight must lie in [0, 1]")


@dataclass(frozen=True)
class FixedParameters:
    label: Mix
    drug: Mix
    protein: Mix


def _average(neighbors: Neighbors, mix: Mix) -> Tensor:
    selected = neighbors.first(mix.k)
    weights = torch.softmax(-selected.distances / mix.temperature, dim=1)
    if selected.values.ndim == 3:
        weights = weights.unsqueeze(-1)
    return (weights * selected.values).sum(dim=1)


def _head_prediction(head: nn.Module, drug: Tensor, protein: Tensor) -> Tensor:
    prediction = head(drug, protein)
    if prediction.shape not in ((len(drug),), (len(drug), 1)):
        raise ValueError("Affinity head must return one prediction per pair")
    return prediction.reshape(-1)


def fixed_prediction(query: Query, head: nn.Module, parameters: FixedParameters) -> Tensor:
    """Equations 2 and the unified framework in section 3.1.3."""
    if any(module.training for module in head.modules()):
        raise ValueError("Affinity head must be in evaluation mode")
    drug = (parameters.drug.base_weight * query.drug
            + (1 - parameters.drug.base_weight) * _average(query.drug_neighbors, parameters.drug))
    protein = (parameters.protein.base_weight * query.protein
               + (1 - parameters.protein.base_weight) * _average(query.protein_neighbors, parameters.protein))
    prediction = _head_prediction(head, drug, protein)
    return (parameters.label.base_weight * prediction
            + (1 - parameters.label.base_weight) * _average(query.labels, parameters.label))


class AdaptiveAggregation(nn.Module):
    """Three distance-conditioned FFNs; the affinity head remains frozen."""
    def __init__(self, head: nn.Module, pair_k: int, drug_k: int,
                 protein_k: int, hidden_dim: int = 32):
        super().__init__()
        if min(pair_k, drug_k, protein_k, hidden_dim) < 1:
            raise ValueError("Network dimensions must be positive")
        self.head = head
        self.head.requires_grad_(False)
        self.head.eval()
        self.pair_k, self.drug_k, self.protein_k = pair_k, drug_k, protein_k

        def network(k: int) -> nn.Sequential:
            return nn.Sequential(nn.Linear(k, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, k + 1))

        self.label_network = network(pair_k)
        self.drug_network = network(drug_k)
        self.protein_network = network(protein_k)
        self.register_buffer("distance_scale", torch.ones(3))

    def train(self, mode: bool = True):
        super().train(mode)
        self.head.eval()
        return self

    @staticmethod
    def weights(network: nn.Module, neighbors: Neighbors, k: int, scale: Tensor) -> Tensor:
        if neighbors.distances.shape[1] < k:
            raise ValueError("Initialize adaptive network with the effective neighbor count")
        return torch.softmax(network(-neighbors.distances[:, :k] / scale), dim=1)

    def forward(self, query: Query) -> Tensor:
        wd = self.weights(self.drug_network, query.drug_neighbors, self.drug_k, self.distance_scale[1])
        wp = self.weights(self.protein_network, query.protein_neighbors, self.protein_k, self.distance_scale[2])
        drug = (wd[:, :1] * query.drug
                + (wd[:, 1:].unsqueeze(-1) * query.drug_neighbors.values[:, :self.drug_k]).sum(1))
        protein = (wp[:, :1] * query.protein
                   + (wp[:, 1:].unsqueeze(-1) * query.protein_neighbors.values[:, :self.protein_k]).sum(1))
        prediction = _head_prediction(self.head, drug, protein)
        wl = self.weights(self.label_network, query.labels, self.pair_k, self.distance_scale[0])
        return wl[:, 0] * prediction + (wl[:, 1:] * query.labels.values[:, :self.pair_k]).sum(1)
