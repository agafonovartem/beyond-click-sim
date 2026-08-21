"""PyTorch/PyG LightGCN policy, following the reference implementation.

`LightGCNPolicy` in `policies.py` is a pure-NumPy LightGCN that applies one
full-batch Adam update per epoch. That is not the protocol the paper specifies:
He et al. (SIGIR 2020, Section 3.3) use Adam "in a mini-batch manner", and
Section 4.1.2 gives batch size 1024-2048 with ~1000 epochs — roughly 300k
parameter updates on MovieLens-1M where the NumPy class performs a few hundred.

This module implements the paper's protocol on top of PyG's `LightGCN` module,
matching the author-endorsed PyTorch reference (gusye1234/LightGCN-PyTorch,
cited as footnote 7 of the paper) wherever it and the paper text differ:

- embeddings initialised N(0, 0.1^2), not Xavier (the reference argues normal
  init is better for a model with no nonlinearity; the NumPy class agrees);
- BPR loss averaged over the batch, with L2 on the layer-0 ("ego") embeddings
  of the batch triples only, scaled by 1/batch_size;
- one (user, positive, negative) triple per training interaction per epoch,
  with users drawn *uniformly at random* rather than by iterating interactions;
- negatives drawn uniformly with rejection against the user's positives.

The NumPy class is deliberately left in place as a reference implementation and
a fallback. The two are drop-in interchangeable: same constructor parameters
(plus `device`), same `Policy` contract, same manifest surface. To switch back,
point `POLICY_ROSTER` in `policy_tuning.py` at `LightGCNPolicy` and rename the
matching `SEARCH_SPACES` key.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Self

import torch
import numpy as np
import pandas as pd
from loguru import logger
from torch_geometric.nn.models import LightGCN

from beyond_click_sim.tasks.policies import Policy


class LightGCNTorchPolicy(Policy):
    """Recommend top-k items via LightGCN (He et al. 2020), PyTorch/PyG backend.

    Propagation, symmetric D^(-1/2) A D^(-1/2) normalisation (no self-loops) and
    the uniform 1/(L+1) layer combination are provided by
    `torch_geometric.nn.models.LightGCN`; this class owns the graph
    construction, the BPR triple sampling, the training loop and the `Policy`
    contract.

    Only items seen in training have embeddings and can be recommended.
    Cold-start users (absent from training) receive no recommendations. With
    `positive_feedback_column` set, the bipartite graph contains only
    positive-feedback rows, while seen-item exclusion at recommend time still
    uses the full (unfiltered) training frame.

    Note on `iterations`: here it counts *epochs of mini-batch updates*
    (ceil(n_pairs / batch_size) optimiser steps each), whereas the NumPy
    `LightGCNPolicy` counts single full-batch updates. The parameter name is
    shared to keep the two interchangeable, but the units are not comparable —
    `iterations=1000` here is the paper's setting, not 5x the NumPy default.
    """

    def __init__(
        self,
        k: int,
        n_factors: int = 64,
        n_layers: int = 3,
        learning_rate: float = 0.001,
        regularization: float = 1e-4,
        iterations: int = 1000,
        batch_size: int = 2048,
        seed: int = 0,
        positive_feedback_column: str | None = None,
        name: str | None = None,
        device: str = "auto",
    ) -> None:
        super().__init__(
            k=k,
            seed=seed,
            positive_feedback_column=positive_feedback_column,
            name=name,
        )
        if n_factors < 1:
            raise ValueError("n_factors must be positive.")
        if n_layers < 1:
            raise ValueError("n_layers must be positive.")
        if iterations < 1:
            raise ValueError("iterations must be positive.")
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        self.n_factors = n_factors
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.iterations = iterations
        self.batch_size = batch_size
        self.device = device
        # Recorded in manifests: which backend produced the fitted policy, and
        # which device it actually resolved to.
        self.implementation = "torch_pyg"
        self.resolved_device: str | None = None

    @staticmethod
    def _resolve_device(requested: str) -> str:
        if requested != "auto":
            return requested
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _sample_triples(
        self,
        rng: np.random.Generator,
        *,
        n_users: int,
        n_items: int,
        pos_items_sorted: np.ndarray,
        pos_offsets: np.ndarray,
        pos_counts: np.ndarray,
        pair_keys: np.ndarray,
        n_pairs: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw `n_pairs` (user, positive, negative) triples for one epoch.

        Mirrors the reference `UniformSample_original_python`: users are drawn
        uniformly at random over training users (so activity does *not* weight
        the sample), the positive is uniform over that user's interactions, and
        the negative is uniform over the catalogue with rejection against the
        user's positive set.
        """

        users = rng.integers(0, n_users, size=n_pairs)
        slots = (rng.random(n_pairs) * pos_counts[users]).astype(np.int64)
        positives = pos_items_sorted[pos_offsets[users] + slots]

        negatives = rng.integers(0, n_items, size=n_pairs)
        # Rejection-resample negatives that are actually positives for the user.
        for _ in range(32):
            keys = users.astype(np.int64) * n_items + negatives
            hits = np.searchsorted(pair_keys, keys)
            hits = np.clip(hits, 0, len(pair_keys) - 1)
            collides = pair_keys[hits] == keys
            if not collides.any():
                break
            negatives[collides] = rng.integers(0, n_items, size=int(collides.sum()))
        return users, positives, negatives

    def fit(
        self,
        train_interactions: pd.DataFrame,
        *,
        items: pd.DataFrame,
        user_column: str = "user_id",
        item_column: str = "item_id",
    ) -> Self:

        device_name = self._resolve_device(self.device)
        self.resolved_device = device_name
        device = torch.device(device_name)

        frame = self._fit_frame(train_interactions, user_column=user_column)
        # Deduplicate: LightGCN treats interactions as binary edges.
        unique = frame[[user_column, item_column]].drop_duplicates()
        users_seen = unique[user_column].unique()
        items_seen = unique[item_column].unique()
        n_users = len(users_seen)
        n_items = len(items_seen)

        user_idx = {u: i for i, u in enumerate(users_seen)}
        item_idx = {it: i for i, it in enumerate(items_seen)}
        u_ids = unique[user_column].map(user_idx).to_numpy()
        i_ids = unique[item_column].map(item_idx).to_numpy()
        n_pairs = len(u_ids)

        # Bipartite graph in a single node space: items are offset by n_users.
        # Both directions must be present — LGConv does not symmetrise.
        src = np.concatenate([u_ids, i_ids + n_users])
        dst = np.concatenate([i_ids + n_users, u_ids])
        edge_index = torch.as_tensor(
            np.stack([src, dst]), dtype=torch.long, device=device
        )
        # Pre-normalise D^(-1/2) A D^(-1/2) once. LGConv would otherwise call
        # gcn_norm over every edge on every layer of every mini-batch, which
        # dominates runtime at ~300 batches/epoch. The reference implementation
        # likewise builds the normalised graph a single time.
        degree = np.bincount(src, minlength=n_users + n_items).astype(np.float32)
        inv_sqrt = np.where(degree > 0, degree**-0.5, 0.0)
        weights = torch.as_tensor(
            inv_sqrt[src] * inv_sqrt[dst], dtype=torch.float32, device=device
        )
        # Two propagation layouts, both pre-normalised so LGConv never recomputes
        # gcn_norm (it would otherwise re-normalise every edge on every layer of
        # every mini-batch, which dominates runtime at ~300 batches/epoch):
        #
        #   CPU/CUDA: a sparse CSR adjacency, dispatched through spmm.
        #   MPS:      dense edge_index + edge_weight, dispatched through
        #             gather/scatter. PyG's spmm calls `aten::_to_sparse_csr`,
        #             which the SparseMPS backend does not implement, so the
        #             sparse path is unavailable there entirely.
        #
        # The graph is symmetric, so the adj_t convention LGConv expects holds.
        if device.type == "mps":
            propagation_index: Any = edge_index
            propagation_weight: Any = weights
        else:
            propagation_index = (
                torch.sparse_coo_tensor(
                    edge_index, weights, (n_users + n_items, n_users + n_items)
                )
                .coalesce()
                .to_sparse_csr()
            )
            propagation_weight = None

        # Per-user positive lists in CSR layout, for O(1) positive sampling.
        order = np.argsort(u_ids, kind="stable")
        pos_items_sorted = i_ids[order]
        pos_counts = np.bincount(u_ids, minlength=n_users)
        pos_offsets = np.concatenate([[0], np.cumsum(pos_counts)])[:-1]
        # Sorted (user, item) keys for negative rejection.
        pair_keys = np.sort(u_ids.astype(np.int64) * n_items + i_ids)

        torch.manual_seed(self.seed)
        model = LightGCN(
            num_nodes=n_users + n_items,
            embedding_dim=self.n_factors,
            num_layers=self.n_layers,
            normalize=False,  # adjacency is pre-normalised above
        ).to(device)
        # PyG resets to xavier_uniform_; the reference implementation uses
        # N(0, 0.1^2) and argues it suits a model with no nonlinearity.
        torch.nn.init.normal_(model.embedding.weight, std=0.1)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        rng = np.random.default_rng(self.seed)

        model.train()
        for epoch in range(self.iterations):
            users, positives, negatives = self._sample_triples(
                rng,
                n_users=n_users,
                n_items=n_items,
                pos_items_sorted=pos_items_sorted,
                pos_offsets=pos_offsets,
                pos_counts=pos_counts,
                pair_keys=pair_keys,
                n_pairs=n_pairs,
            )
            perm = rng.permutation(n_pairs)
            u_t = torch.as_tensor(users[perm], dtype=torch.long, device=device)
            p_t = torch.as_tensor(
                positives[perm] + n_users, dtype=torch.long, device=device
            )
            n_t = torch.as_tensor(
                negatives[perm] + n_users, dtype=torch.long, device=device
            )

            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n_pairs, self.batch_size):
                end = min(start + self.batch_size, n_pairs)
                ub, pb, nb = u_t[start:end], p_t[start:end], n_t[start:end]
                edge_label_index = torch.stack(
                    [torch.cat([ub, ub]), torch.cat([pb, nb])]
                )
                ranks = model(
                    propagation_index, edge_label_index, propagation_weight
                )
                pos_rank, neg_rank = ranks.chunk(2)
                # node_id must be the (non-unique) batch triples: passing None
                # would regularise the entire embedding table once per batch.
                # PyG applies lambda_reg * ||E||^2 / B while the reference
                # applies decay * 0.5 * ||E||^2 / B, hence the factor of two.
                loss = model.recommendation_loss(
                    pos_rank,
                    neg_rank,
                    node_id=torch.cat([ub, pb, nb]),
                    lambda_reg=self.regularization / 2.0,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach())
                n_batches += 1

            if (epoch + 1) % 50 == 0 or epoch + 1 == self.iterations:
                logger.info(
                    f"{self.name} epoch {epoch + 1}/{self.iterations} "
                    f"loss={epoch_loss / max(n_batches, 1):.4f} "
                    f"device={device_name}"
                )

        model.eval()
        with torch.no_grad():
            embedding = (
                model.get_embedding(propagation_index, propagation_weight)
                .cpu()
                .numpy()
            )

        self._model_ = model
        self._edge_index_ = edge_index
        self._propagation_index_ = propagation_index
        self._propagation_weight_ = propagation_weight
        self._embedding_ = embedding
        self._n_users_ = n_users
        self._n_items_ = n_items
        self._user_idx_ = user_idx
        self._item_idx_ = item_idx
        self._items_in_train_: list[Any] = items_seen.tolist()
        return self

    def recommend(
        self,
        users: pd.DataFrame,
        *,
        train_interactions: pd.DataFrame,
        items: pd.DataFrame,
        user_column: str = "user_id",
        item_column: str = "item_id",
    ) -> pd.DataFrame:
        embedding_users = self._embedding_[: self._n_users_]
        embedding_items = self._embedding_[self._n_users_ :]

        # Seen-exclusion uses the full (unfiltered) training frame so that
        # consumed-but-disliked items are never recommended, even when a
        # positive-feedback filter kept them out of the fitted graph.
        seen_by_user = self._seen_items_by_user(
            train_interactions,
            user_column=user_column,
            item_column=item_column,
        )
        output_columns = [user_column, item_column, "policy", "rank"]
        rows: list[dict[str, Any]] = []

        for user_id in users[user_column]:
            user_row = self._user_idx_.get(user_id)
            if user_row is None:
                continue

            seen = seen_by_user.get(user_id, set())
            seen_indices = {
                self._item_idx_[it] for it in seen if it in self._item_idx_
            }

            scores = embedding_items @ embedding_users[user_row]
            for idx in seen_indices:
                scores[idx] = -np.inf

            n_candidates = self._n_items_ - len(seen_indices)
            if n_candidates == 0:
                continue
            n_take = min(self.k, n_candidates)
            top_indices = np.argpartition(scores, -n_take)[-n_take:]
            top_indices = top_indices[np.argsort(-scores[top_indices])]
            top_indices = [i for i in top_indices if not np.isinf(scores[i])]

            selected = [self._items_in_train_[i] for i in top_indices]
            rows.extend(
                self._build_recommendation_rows(
                    user_id,
                    selected,
                    user_column=user_column,
                    item_column=item_column,
                )
            )

        return pd.DataFrame(rows, columns=output_columns)

    def rank_candidates(
        self,
        user_id: Any,
        candidate_items: Sequence[Any],
        *,
        train_interactions: pd.DataFrame | None = None,
        user_column: str = "user_id",
        item_column: str = "item_id",
    ) -> list[Any]:
        """Score candidates by the LightGCN user-item dot product.

        `self._embedding_` is already layer-propagated at fit time, so unlike
        the NumPy class there is no forward pass to repeat per user. Cold users
        and items unseen in training carry no signal and fall back to the seeded
        shuffle.
        """

        user_row = self._user_idx_.get(user_id)
        if user_row is None:
            return self._seeded_permutation(user_id, candidate_items)

        embedding_users = self._embedding_[: self._n_users_]
        embedding_items = self._embedding_[self._n_users_ :]
        user_vector = embedding_users[user_row]
        scores = {}
        for item in candidate_items:
            idx = self._item_idx_.get(item)
            if idx is not None:
                scores[item] = float(embedding_items[idx] @ user_vector)
        return self._order_by_score(user_id, candidate_items, scores)
