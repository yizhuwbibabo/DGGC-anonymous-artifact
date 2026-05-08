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

    ) -> None:

        super().__init__()

        self.backbone = AutoModel.from_pretrained(model_name)

        hidden = int(self.backbone.config.hidden_size)

        self.projection_dim = projection_dim

        self.sep_token_id = sep_token_id

        self.max_cols = max_cols

        self.sigma_eps = sigma_eps

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

        sigma_t = self.sigma_head_t(cls_emb).clamp_min(self.sigma_eps)

        col_embs, col_mask = self._extract_column_embeddings(input_ids_flat, attention_mask_flat, last_hidden)

        mu_cols = F.normalize(self.mu_head_c(col_embs), p=2, dim=-1)

        sigma_cols = self.sigma_head_c(col_embs).clamp_min(self.sigma_eps)

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

    def __init__(

        self,

        temperature: float = 0.07,

        lambda_reg: float = 0.5,

        alpha: float = 1.0,

        beta: float = 1.0,

        col_temperature: float = 0.05,

        col_match_mode: str = "logsumexp",

    ) -> None:

        super().__init__()

        self.temperature = temperature

        self.lambda_reg = lambda_reg

        self.alpha = alpha

        self.beta = beta

        self.col_temperature = col_temperature

        self.col_match_mode = col_match_mode

        self.ce = nn.CrossEntropyLoss()

    def g_score(self, mu1: torch.Tensor, sigma1: torch.Tensor, mu2: torch.Tensor, sigma2: torch.Tensor) -> torch.Tensor:

        sim_mu = torch.matmul(mu1, mu2.T) / self.temperature

        dist_sigma = torch.cdist(sigma1, sigma2, p=2) * self.lambda_reg

        return sim_mu - dist_sigma

    def column_set_score(

        self,

        mu_a: torch.Tensor,

        sigma_a: torch.Tensor,

        mask_a: torch.Tensor,

        mu_b: torch.Tensor,

        sigma_b: torch.Tensor,

        mask_b: torch.Tensor,

    ) -> torch.Tensor:

        sim = torch.einsum("acd,bfd->abcf", mu_a, mu_b) / self.temperature

        dist = torch.linalg.norm(

            sigma_a[:, None, :, None, :] - sigma_b[None, :, None, :, :], dim=-1

        ) * self.lambda_reg

        score = sim - dist

        valid = (mask_a[:, None, :, None] > 0) & (mask_b[None, :, None, :] > 0)

        score = score.masked_fill(~valid, -1e4)

        if self.col_match_mode == "max":

            best = score.max(dim=-1).values

        else:

            tau = max(float(self.col_temperature), 1e-4)

            best = tau * torch.logsumexp(score / tau, dim=-1)

        best = best.masked_fill(mask_a[:, None, :] <= 0, 0.0)

        denom = mask_a.sum(dim=1).clamp_min(1.0)[:, None]

        return best.sum(dim=-1) / denom

    def alignment_loss(self, mu_t, sigma_t, mu_agg, sigma_agg) -> torch.Tensor:

        score = torch.sum(mu_t * mu_agg, dim=-1) / self.temperature

        penalty = torch.linalg.norm(sigma_t - sigma_agg, dim=-1) * self.lambda_reg

        return -(score - penalty).mean()

    def forward(self, out: PerColumnEncoderOutput) -> torch.Tensor:

        if out.mu_t.dim() != 3 or out.mu_t.shape[1] < 2:

            raise ValueError("Training expects two views per sample: [B, 2, L].")

        labels = torch.arange(out.mu_t.shape[0], device=out.mu_t.device)

        a_mu_t, p_mu_t = out.mu_t[:, 0, :], out.mu_t[:, 1, :]

        a_sig_t, p_sig_t = out.sigma_t[:, 0, :], out.sigma_t[:, 1, :]

        logits_t = self.g_score(a_mu_t, a_sig_t, p_mu_t, p_sig_t)

        loss_t = self.ce(logits_t, labels)

        a_mu_c, p_mu_c = out.mu_cols[:, 0], out.mu_cols[:, 1]

        a_sig_c, p_sig_c = out.sigma_cols[:, 0], out.sigma_cols[:, 1]

        a_mask, p_mask = out.col_mask[:, 0], out.col_mask[:, 1]

        logits_c = self.column_set_score(a_mu_c, a_sig_c, a_mask, p_mu_c, p_sig_c, p_mask)

        loss_c = self.ce(logits_c, labels)

        loss_align = 0.5 * (

            self.alignment_loss(out.mu_t[:, 0], out.sigma_t[:, 0], out.mu_agg[:, 0], out.sigma_agg[:, 0])

            + self.alignment_loss(out.mu_t[:, 1], out.sigma_t[:, 1], out.mu_agg[:, 1], out.sigma_agg[:, 1])

        )

        return loss_t + self.alpha * loss_c + self.beta * loss_align
