from __future__ import annotations

from dataclasses import dataclass

from typing import Optional, Tuple

import torch

import torch.nn as nn

import torch.nn.functional as F

from transformers import AutoModel

@dataclass

class PerColumnEncoderOutput:

    mu_t: torch.Tensor

    sigma_t: torch.Tensor

    mu_cols: torch.Tensor

    sigma_cols: torch.Tensor

    col_mask: torch.Tensor

    mu_agg: torch.Tensor

    sigma_agg: torch.Tensor

class PerColumnWassersteinTableEncoder(nn.Module):

    def __init__(

        self,

        model_name: str,

        projection_dim: int = 128,

        sep_token_id: Optional[int] = None,

        max_cols: int = 12,

        sigma_eps: float = 1e-6,

        use_variance: bool = True,

    ) -> None:

        super().__init__()

        self.backbone = AutoModel.from_pretrained(model_name)

        hidden = int(self.backbone.config.hidden_size)

        self.projection_dim = projection_dim

        self.sep_token_id = sep_token_id

        self.max_cols = max_cols

        self.sigma_eps = sigma_eps

        # Ablation switch for the "w/o Var" variant: when disabled, sigma is a

        # constant vector, so all variance terms in the losses and scores

        # vanish and the encoder degenerates to point embeddings.

        self.use_variance = use_variance

        self.mu_head_t = nn.Sequential(

            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, projection_dim)

        )

        self.sigma_head_t = nn.Sequential(

            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, projection_dim), nn.Softplus()

        )

        self.mu_head_c = nn.Sequential(

            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, projection_dim)

        )

        self.sigma_head_c = nn.Sequential(

            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, projection_dim), nn.Softplus()

        )

        self.attn_mu = nn.Linear(projection_dim, 1)

        self.attn_sigma = nn.Linear(projection_dim, 1)

    def _extract_column_embeddings(

        self,

        input_ids: torch.Tensor,

        attention_mask: torch.Tensor,

        last_hidden: torch.Tensor,

    ) -> Tuple[torch.Tensor, torch.Tensor]:

        bsz, seqlen, hidden = last_hidden.shape

        device = last_hidden.device

        dtype = last_hidden.dtype

        col_embs = torch.zeros((bsz, self.max_cols, hidden), device=device, dtype=dtype)

        col_mask = torch.zeros((bsz, self.max_cols), device=device, dtype=dtype)

        if self.sep_token_id is None:

            token_mask = attention_mask.clone()

            token_mask[:, 0] = 0

            denom = token_mask.sum(dim=1).clamp_min(1).to(dtype)

            pooled = (last_hidden * token_mask.unsqueeze(-1).to(dtype)).sum(dim=1) / denom.unsqueeze(-1)

            col_embs[:, 0, :] = pooled

            col_mask[:, 0] = 1.0

            return col_embs, col_mask

        for i in range(bsz):

            valid_len = int(attention_mask[i].sum().item())

            sep_positions = (input_ids[i, :valid_len] == self.sep_token_id).nonzero(as_tuple=False).flatten().tolist()

            start = 1

            col_id = 0

            for sep in sep_positions:

                if col_id >= self.max_cols:

                    break

                end = int(sep)

                if end > start:

                    span_mask = attention_mask[i, start:end].to(dtype)

                    if span_mask.sum() > 0:

                        span_hidden = last_hidden[i, start:end, :]

                        pooled = (span_hidden * span_mask.unsqueeze(-1)).sum(dim=0) / span_mask.sum().clamp_min(1e-9)

                        col_embs[i, col_id, :] = pooled

                        col_mask[i, col_id] = 1.0

                        col_id += 1

                start = end + 1

            if col_id < self.max_cols and start < valid_len:

                span_mask = attention_mask[i, start:valid_len].to(dtype)

                if span_mask.sum() > 0:

                    span_hidden = last_hidden[i, start:valid_len, :]

                    pooled = (span_hidden * span_mask.unsqueeze(-1)).sum(dim=0) / span_mask.sum().clamp_min(1e-9)

                    col_embs[i, col_id, :] = pooled

                    col_mask[i, col_id] = 1.0

        empty = col_mask.sum(dim=1) == 0

        if empty.any():

            cls_emb = last_hidden[empty, 0, :]

            col_embs[empty, 0, :] = cls_emb

            col_mask[empty, 0] = 1.0

        return col_embs, col_mask

    def _aggregate_columns(

        self,

        mu_cols: torch.Tensor,

        sigma_cols: torch.Tensor,

        col_mask: torch.Tensor,

    ) -> Tuple[torch.Tensor, torch.Tensor]:

        neg = torch.finfo(mu_cols.dtype).min / 4

        mu_logits = self.attn_mu(mu_cols).squeeze(-1).masked_fill(col_mask <= 0, neg)

        mu_w = torch.softmax(mu_logits, dim=1)

        mu_agg = torch.sum(mu_cols * mu_w.unsqueeze(-1), dim=1)

        mu_agg = F.normalize(mu_agg, p=2, dim=-1)

        log_sigma = torch.log(sigma_cols.clamp_min(self.sigma_eps))

        sig_logits = self.attn_sigma(log_sigma).squeeze(-1).masked_fill(col_mask <= 0, neg)

        sig_w = torch.softmax(sig_logits, dim=1)

        sigma_agg = torch.exp(torch.sum(log_sigma * sig_w.unsqueeze(-1), dim=1)).clamp_min(self.sigma_eps)

        return mu_agg, sigma_agg

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):

        is_views = input_ids.dim() == 3

        if is_views:

            b, v, s = input_ids.shape

            input_ids_flat = input_ids.reshape(b * v, s)

            attention_mask_flat = attention_mask.reshape(b * v, s)

        else:

            b = input_ids.shape[0]

            v = 1

            input_ids_flat = input_ids

            attention_mask_flat = attention_mask

        outputs = self.backbone(input_ids=input_ids_flat, attention_mask=attention_mask_flat)

        last_hidden = outputs.last_hidden_state

        cls_emb = last_hidden[:, 0, :]

        mu_t = F.normalize(self.mu_head_t(cls_emb), p=2, dim=-1)

        # phi_sigma(x) = softplus(x) + eps (Sec. 3.2); the head ends with Softplus.

        sigma_t = self.sigma_head_t(cls_emb) + self.sigma_eps

        if not self.use_variance:

            sigma_t = torch.ones_like(mu_t)

        col_embs, col_mask = self._extract_column_embeddings(input_ids_flat, attention_mask_flat, last_hidden)

        mu_cols = F.normalize(self.mu_head_c(col_embs), p=2, dim=-1)

        sigma_cols = self.sigma_head_c(col_embs) + self.sigma_eps

        if not self.use_variance:

            sigma_cols = torch.ones_like(mu_cols)

        mu_cols = mu_cols * col_mask.unsqueeze(-1)

        sigma_cols = sigma_cols * col_mask.unsqueeze(-1)

        mu_agg, sigma_agg = self._aggregate_columns(mu_cols, sigma_cols.clamp_min(self.sigma_eps), col_mask)

        if is_views:

            return PerColumnEncoderOutput(

                mu_t=mu_t.view(b, v, -1),

                sigma_t=sigma_t.view(b, v, -1),

                mu_cols=mu_cols.view(b, v, self.max_cols, -1),

                sigma_cols=sigma_cols.view(b, v, self.max_cols, -1),

                col_mask=col_mask.view(b, v, self.max_cols),

                mu_agg=mu_agg.view(b, v, -1),

                sigma_agg=sigma_agg.view(b, v, -1),

            )

        return PerColumnEncoderOutput(mu_t, sigma_t, mu_cols, sigma_cols, col_mask, mu_agg, sigma_agg)

class PerColumnWassersteinContrastiveLoss(nn.Module):

    """Gaussian contrastive objectives.

    L_t and L_c follow Eq. (2), a multi-positive contrastive loss over the

    compatibility score G of Eq. (3); L_align is the table-column alignment

    loss of Sec. 3.3. The total objective is Eq. (6):

    L = L_t + alpha * L_c + beta * L_align."""

    def __init__(

        self,

        lambda_reg: float = 0.5,

        alpha: float = 1.0,

        beta: float = 1.0,

        col_temperature: float = 0.05,

        col_match_mode: str = "logsumexp",

    ) -> None:

        super().__init__()

        self.lambda_reg = lambda_reg

        self.alpha = alpha

        self.beta = beta

        self.col_temperature = col_temperature

        self.col_match_mode = col_match_mode

    def g_score(self, mu1: torch.Tensor, sigma1: torch.Tensor, mu2: torch.Tensor, sigma2: torch.Tensor) -> torch.Tensor:

        """Eq. (3): G(q,t) = -||mu_q - mu_t||_2^2 - lambda * ||sigma_q - sigma_t||_2^2."""

        mean_dist = torch.cdist(mu1, mu2, p=2) ** 2

        var_dist = torch.cdist(sigma1, sigma2, p=2) ** 2

        return -(mean_dist + self.lambda_reg * var_dist)

    def _best_column_scores(self, score: torch.Tensor, mask_a: torch.Tensor, mask_b: torch.Tensor) -> torch.Tensor:

        # score: [N, M, Ca, Cb]; reduce each column of a to its best match in b.

        valid = (mask_a[:, None, :, None] > 0) & (mask_b[None, :, None, :] > 0)

        score = score.masked_fill(~valid, -1e4)

        if self.col_match_mode == "max":

            best = score.max(dim=-1).values

        else:

            tau = max(float(self.col_temperature), 1e-4)

            best = tau * torch.logsumexp(score / tau, dim=-1)

        best = best.masked_fill(mask_a[:, None, :] <= 0, 0.0)

        denom = mask_a.sum(dim=-1).clamp_min(1.0)[:, None]

        return best.sum(dim=-1) / denom

    def column_set_scores(self, mu_a, sigma_a, mask_a, mu_b, sigma_b, mask_b, chunk: int = 4) -> torch.Tensor:

        """Column-level compatibility between two sets of tables: each column

        of a table in `a` is matched to its best column of a table in `b`

        under G (Eq. 3), and the matched scores are averaged over the columns

        of the `a` table."""

        N, M = mu_a.shape[0], mu_b.shape[0]

        out = mu_a.new_zeros((N, M))

        for s0 in range(0, N, chunk):

            e0 = min(s0 + chunk, N)

            mean_dist = ((mu_a[s0:e0, None, :, None, :] - mu_b[None, :, None, :, :]) ** 2).sum(-1)

            var_dist = ((sigma_a[s0:e0, None, :, None, :] - sigma_b[None, :, None, :, :]) ** 2).sum(-1)

            score = -(mean_dist + self.lambda_reg * var_dist)

            out[s0:e0] = self._best_column_scores(score, mask_a[s0:e0], mask_b)

        return out

    def alignment_loss(self, mu_t, sigma_t, mu_agg, sigma_agg) -> torch.Tensor:

        """L_align = ||mu_t - mu_agg||_2^2 + lambda * ||sigma_t - sigma_agg||_2^2."""

        return (

            torch.linalg.norm(mu_t - mu_agg, dim=-1) ** 2

            + self.lambda_reg * torch.linalg.norm(sigma_t - sigma_agg, dim=-1) ** 2

        ).mean()

    @staticmethod

    def _multi_positive_loss(g_pos: torch.Tensor, g_neg: torch.Tensor, neg_mask: torch.Tensor) -> torch.Tensor:

        """Eq. (2): -log( sum_p exp(G(q,t_p+)) /

                        (sum_p exp(G(q,t_p+)) + sum_n exp(G(q,t_n-))) ).

        A per-anchor additive constant cancels in the ratio, so scores are

        shifted by their maximum for numerical stability."""

        neg_scores = g_neg.masked_fill(~neg_mask, float("-inf"))

        m = torch.maximum(g_pos.max(dim=1).values, neg_scores.max(dim=1).values)

        pos_sum = torch.exp(g_pos - m[:, None]).sum(dim=1)

        neg_sum = torch.exp(neg_scores - m[:, None]).sum(dim=1)

        loss = -torch.log(pos_sum / (pos_sum + neg_sum))

        loss = torch.where(neg_mask.any(dim=1), loss, torch.zeros_like(loss))

        return loss.mean()

    def forward(self, out: PerColumnEncoderOutput, pos_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        # View 0 is the anchor table; views 1..P are its sampled positive

        # views (filter-shuffle cross-table positives or self-augmentations).

        mu_t = out.mu_t.float()

        sigma_t = out.sigma_t.float()

        mu_cols = out.mu_cols.float()

        sigma_cols = out.sigma_cols.float()

        col_mask = out.col_mask

        B, V = mu_t.shape[0], mu_t.shape[1]

        if V < 2:

            raise ValueError("Training expects at least one positive view per sample: [B, 1+P, L].")

        P = V - 1

        device = mu_t.device

        a_mu, a_sig = mu_t[:, 0], sigma_t[:, 0]

        p_mu = mu_t[:, 1:].reshape(B * P, -1)

        p_sig = sigma_t[:, 1:].reshape(B * P, -1)

        if pos_mask is None:

            pos_mask = torch.zeros((B, B), dtype=torch.bool, device=device)

        neg_mask = (~torch.eye(B, dtype=torch.bool, device=device)) & (~pos_mask)

        # Table-level loss (Eqs. 2-3).

        # G between each anchor and its OWN positive views: [B, P].

        p_mu_v = p_mu.view(B, P, -1)

        p_sig_v = p_sig.view(B, P, -1)

        g_pos = -(((a_mu[:, None, :] - p_mu_v) ** 2).sum(-1) + self.lambda_reg * ((a_sig[:, None, :] - p_sig_v) ** 2).sum(-1))

        g_neg = self.g_score(a_mu, a_sig, a_mu, a_sig)

        loss_t = self._multi_positive_loss(g_pos, g_neg, neg_mask)

        # Column-level loss, defined analogously (Sec. 3.3).

        a_cols = (mu_cols[:, 0], sigma_cols[:, 0], col_mask[:, 0])

        p_cols = (

            mu_cols[:, 1:].reshape(B * P, *mu_cols.shape[2:]),

            sigma_cols[:, 1:].reshape(B * P, *sigma_cols.shape[2:]),

            col_mask[:, 1:].reshape(B * P, *col_mask.shape[2:]),

        )

        # Column-level scores between each anchor and its OWN positive views:

        # the pairwise matrix is [B, B*P]; keep the block-diagonal [B, P].

        s_pos_full = self.column_set_scores(*a_cols, *p_cols).view(B, B, P)

        idx = torch.arange(B, device=device)

        s_pos = s_pos_full[idx, idx]

        s_neg = self.column_set_scores(*a_cols, *a_cols)

        loss_c = self._multi_positive_loss(s_pos, s_neg, neg_mask)

        # Table-column alignment loss over all views.

        loss_align = self.alignment_loss(

            mu_t.reshape(B * V, -1),

            sigma_t.reshape(B * V, -1),

            out.mu_agg.reshape(B * V, -1).float(),

            out.sigma_agg.reshape(B * V, -1).float(),

        )

        return loss_t + self.alpha * loss_c + self.beta * loss_align

