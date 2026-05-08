from __future__ import annotations

import argparse

import glob

import json

import os

import random

from typing import Any, Dict, List, Optional

import numpy as np

import pandas as pd

import torch

from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm

from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from model import PerColumnWassersteinTableEncoder, PerColumnWassersteinContrastiveLoss

def set_seed(seed: int) -> None:

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

def safe_read_csv(path: str, sample_rows: int = 50) -> Optional[pd.DataFrame]:

    try:

        df = pd.read_csv(path, nrows=sample_rows, low_memory=False)

        if df is not None and not df.empty and len(df.columns) > 0:

            return df

    except Exception:

        pass

    try:

        df = pd.read_csv(path, nrows=sample_rows, encoding="latin1", low_memory=False)

        if df is not None and not df.empty and len(df.columns) > 0:

            return df

    except Exception:

        return None

    return None

def serialize_table(df: pd.DataFrame, max_rows: int = 3, max_cols: int = 12) -> str:

    if df is None or df.empty or len(df.columns) == 0:

        return "empty_table [SEP]"

    df = df.astype(str).fillna("")

    parts: List[str] = []

    for col in list(df.columns)[:max_cols]:

        vals = [str(v).strip() for v in df[col].tolist()[:max_rows] if str(v).strip()]

        val_str = " | ".join(vals[:max_rows])

        parts.append(f"header: {col} values: {val_str} [SEP]")

    return " ".join(parts).strip() if parts else "empty_table [SEP]"

class PerColumnTableDataset(Dataset):

    def __init__(

        self,

        data_dir: str,

        relation_map_path: str,

        tokenizer,

        max_length: int = 160,

        max_rows: int = 3,

        max_cols: int = 12,

        csv_cache_rows: int = 50,

        min_semantic_positive_score: float = 0.35,

        self_aug_prob: float = 0.10,

    ) -> None:

        self.data_dir = data_dir

        self.tokenizer = tokenizer

        self.max_length = max_length

        self.max_rows = max_rows

        self.max_cols = max_cols

        self.csv_cache_rows = csv_cache_rows

        self.min_semantic_positive_score = min_semantic_positive_score

        self.self_aug_prob = self_aug_prob

        self.table_files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(data_dir, "*.csv")))

        self.relation_map: Dict[str, Any] = {}

        if relation_map_path and os.path.exists(relation_map_path):

            with open(relation_map_path, "r", encoding="utf-8") as f:

                obj = json.load(f)

            if isinstance(obj, dict):

                self.relation_map = obj

        print(f"Relation map keys: {len(self.relation_map)}", flush=True)

        print("Caching CSV data in memory...", flush=True)

        self.df_cache: Dict[str, pd.DataFrame] = {}

        bad = 0

        examples: List[str] = []

        for fn in tqdm(self.table_files, desc="Loading CSVs"):

            df = safe_read_csv(os.path.join(data_dir, fn), sample_rows=csv_cache_rows)

            if df is None or df.empty or len(df.columns) == 0:

                bad += 1

                if len(examples) < 5:

                    examples.append(fn)

                continue

            self.df_cache[fn] = df

        self.table_files = [f for f in self.table_files if f in self.df_cache]

        print(f"Loaded {len(self.table_files)} CSV files. Skipped {bad} bad files.", flush=True)

        if examples:

            print("First skipped examples: " + ", ".join(examples), flush=True)

        self.positive_index = self._build_positive_index()

        print(

            f"Positive index: anchors={len(self.positive_index)}, edges={sum(len(v) for v in self.positive_index.values())}",

            flush=True,

        )

    def _entry_score(self, entry: Any) -> float:

        if isinstance(entry, str):

            return 1.0

        if not isinstance(entry, dict):

            return 0.0

        for key in ("semantic_score", "score", "confidence_score"):

            if key in entry:

                try:

                    return float(entry[key])

                except Exception:

                    pass

        return 1.0

    def _entry_table(self, entry: Any) -> Optional[str]:

        if isinstance(entry, str):

            return os.path.basename(entry)

        if isinstance(entry, dict):

            for key in ("table", "target", "candidate"):

                if key in entry:

                    return os.path.basename(str(entry[key]))

        return None

    def _build_positive_index(self) -> Dict[str, List[str]]:

        out: Dict[str, List[str]] = {}

        valid = set(self.table_files)

        for anchor, entries in self.relation_map.items():

            a = os.path.basename(str(anchor))

            if a not in valid:

                continue

            if isinstance(entries, dict):

                entries = entries.get("positives", []) or entries.get("neighbors", [])

            if not isinstance(entries, list):

                continue

            pos: List[str] = []

            for e in entries:

                p = self._entry_table(e)

                if p and p in valid and p != a and self._entry_score(e) >= self.min_semantic_positive_score:

                    pos.append(p)

            if pos:

                out[a] = sorted(set(pos))

        return out

    def __len__(self) -> int:

        return len(self.table_files)

    def _augment_self(self, df: pd.DataFrame) -> pd.DataFrame:

        if df is None or df.empty:

            return df

        cols = list(df.columns)

        random.shuffle(cols)

        df2 = df[cols]

        if len(df2) > 1:

            df2 = df2.sample(n=min(len(df2), max(1, self.max_rows)), replace=False, random_state=random.randint(0, 10**6))

        return df2

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

        anchor = self.table_files[idx]

        anchor_df = self.df_cache[anchor]

        use_self = random.random() < self.self_aug_prob or anchor not in self.positive_index

        if use_self:

            pos_df = self._augment_self(anchor_df)

        else:

            pos_file = random.choice(self.positive_index[anchor])

            pos_df = self.df_cache[pos_file]

        texts = [serialize_table(anchor_df, self.max_rows, self.max_cols), serialize_table(pos_df, self.max_rows, self.max_cols)]

        enc = self.tokenizer(

            texts,

            max_length=self.max_length,

            padding="max_length",

            truncation=True,

            return_tensors="pt",

        )

        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

def save_checkpoint(model: torch.nn.Module, output_dir: str, epoch: int) -> None:

    os.makedirs(output_dir, exist_ok=True)

    state = {"epoch": epoch, "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}

    path = os.path.join(output_dir, f"epoch_{epoch}.pth")

    tmp = path + ".tmp"

    torch.save(state, tmp)

    os.replace(tmp, path)

    latest = os.path.join(output_dir, "latest.pth")

    torch.save(state, latest + ".tmp")

    os.replace(latest + ".tmp", latest)

    print(f"[Epoch {epoch}] saved {path}", flush=True)

def main() -> None:

    ap = argparse.ArgumentParser()

    ap.add_argument("--model_name", default="<MODEL_NAME>")

    ap.add_argument("--data_dir", required=True)

    ap.add_argument("--relation_map", required=True)

    ap.add_argument("--output_dir", required=True)

    ap.add_argument("--epochs", type=int, default=10)

    ap.add_argument("--batch_size", type=int, default=128)

    ap.add_argument("--lr", type=float, default=1e-5)

    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--projection_dim", type=int, default=128)

    ap.add_argument("--temperature", type=float, default=0.07)

    ap.add_argument("--lambda_reg", type=float, default=0.5)

    ap.add_argument("--alpha", type=float, default=1.0)

    ap.add_argument("--beta", type=float, default=1.0)

    ap.add_argument("--col_temperature", type=float, default=0.05)

    ap.add_argument("--col_match_mode", choices=["max", "logsumexp"], default="logsumexp")

    ap.add_argument("--max_length", type=int, default=160)

    ap.add_argument("--max_rows", type=int, default=3)

    ap.add_argument("--max_cols", type=int, default=12)

    ap.add_argument("--csv_cache_rows", type=int, default=50)

    ap.add_argument("--min_semantic_positive_score", type=float, default=0.35)

    ap.add_argument("--self_aug_prob", type=float, default=0.10)

    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--save_every", type=int, default=1)

    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--compile", action="store_true")

    args = ap.parse_args()

    set_seed(args.seed)

    torch.backends.cuda.matmul.allow_tf32 = True

    torch.backends.cudnn.allow_tf32 = True

    try:

        torch.set_float32_matmul_precision("high")

    except Exception:

        pass

    device = torch.device(args.device)

    use_cuda = device.type == "cuda" and torch.cuda.is_available()

    print(f"Using device: {device}", flush=True)

    print(f"Mixed precision enabled: {use_cuda}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    dataset = PerColumnTableDataset(

        data_dir=args.data_dir,

        relation_map_path=args.relation_map,

        tokenizer=tokenizer,

        max_length=args.max_length,

        max_rows=args.max_rows,

        max_cols=args.max_cols,

        csv_cache_rows=args.csv_cache_rows,

        min_semantic_positive_score=args.min_semantic_positive_score,

        self_aug_prob=args.self_aug_prob,

    )

    loader = DataLoader(

        dataset,

        batch_size=args.batch_size,

        shuffle=True,

        drop_last=True,

        num_workers=args.num_workers,

        pin_memory=use_cuda,

        persistent_workers=args.num_workers > 0,

    )

    if len(loader) == 0:

        raise RuntimeError("Empty dataloader. Reduce batch_size or check data.")

    model = PerColumnWassersteinTableEncoder(

        args.model_name,

        projection_dim=args.projection_dim,

        sep_token_id=tokenizer.sep_token_id,

        max_cols=args.max_cols,

    ).to(device)

    if args.compile and hasattr(torch, "compile"):

        print("Compiling model with torch.compile...", flush=True)

        model = torch.compile(model)

    loss_fn = PerColumnWassersteinContrastiveLoss(

        temperature=args.temperature,

        lambda_reg=args.lambda_reg,

        alpha=args.alpha,

        beta=args.beta,

        col_temperature=args.col_temperature,

        col_match_mode=args.col_match_mode,

    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    total_steps = len(loader) * args.epochs

    warmup_steps = min(total_steps // 10, len(loader) * 3)

    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    os.makedirs(args.output_dir, exist_ok=True)

    print("Starting per-column DGGC training...", flush=True)

    for epoch in range(1, args.epochs + 1):

        model.train()

        total = 0.0

        for batch in tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}"):

            ids = batch["input_ids"].to(device, non_blocking=use_cuda)

            mask = batch["attention_mask"].to(device, non_blocking=use_cuda)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):

                out = model(ids, mask)

                loss = loss_fn(out)

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)

            scaler.update()

            scheduler.step()

            total += float(loss.detach().cpu())

        print(f"Epoch {epoch}: loss={total / max(1, len(loader)):.6f}", flush=True)

        if epoch % args.save_every == 0 or epoch == args.epochs:

            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model

            save_checkpoint(raw_model, args.output_dir, epoch)

    print("Training finished successfully.", flush=True)

if __name__ == "__main__":

    main()
