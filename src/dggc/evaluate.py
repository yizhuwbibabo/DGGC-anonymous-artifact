from __future__ import annotations

import argparse

import glob

import json

import os

import re

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import pandas as pd

import torch

from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm

from transformers import AutoTokenizer

from model import PerColumnWassersteinTableEncoder

try:

    import faiss

except Exception:

    faiss = None

def norm_file(x: Any) -> str:

    return os.path.basename(str(x).strip())

def load_gt(path: str) -> Dict[str, List[str]]:

    with open(path, "r", encoding="utf-8") as f:

        raw = json.load(f)

    return {norm_file(q): [norm_file(v) for v in vals] if isinstance(vals, list) else [] for q, vals in raw.items()}

def safe_read_csv(path: str, sample_rows: int) -> Optional[pd.DataFrame]:

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

    parts = []

    for col in list(df.columns)[:max_cols]:

        vals = [str(v).strip() for v in df[col].tolist()[:max_rows] if str(v).strip()]

        parts.append(f"header: {col} values: {' | '.join(vals[:max_rows])} [SEP]")

    return " ".join(parts).strip() if parts else "empty_table [SEP]"

class CsvRecordDataset(Dataset):

    def __init__(self, records: Sequence[Tuple[str, str]], tokenizer, max_length: int, max_rows: int, max_cols: int, csv_nrows: int):

        self.records = list(records)

        self.tokenizer = tokenizer

        self.max_length = max_length

        self.max_rows = max_rows

        self.max_cols = max_cols

        self.csv_nrows = csv_nrows

    def __len__(self) -> int:

        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

        name, path = self.records[idx]

        df = safe_read_csv(path, self.csv_nrows)

        text = serialize_table(df, self.max_rows, self.max_cols) if df is not None else "empty_table [SEP]"

        enc = self.tokenizer(text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")

        return {"input_ids": enc["input_ids"].squeeze(0), "attention_mask": enc["attention_mask"].squeeze(0)}

def build_query_records(gt: Dict[str, List[str]], query_dir: str) -> List[Tuple[str, str]]:

    out = []

    missing = []

    for q in gt:

        path = os.path.join(query_dir, q)

        if os.path.exists(path):

            out.append((q, path))

        else:

            missing.append(q)

    if missing:

        print(f"WARNING: missing {len(missing)} query files, examples={missing[:5]}", flush=True)

    return out

def build_candidate_records(candidate_dir: str) -> List[Tuple[str, str]]:

    return [(os.path.basename(p), p) for p in sorted(glob.glob(os.path.join(candidate_dir, "*.csv")))]

def checkpoint_epoch(path: str) -> int:

    m = re.search(r"epoch_(\d+)\.pth", os.path.basename(path))

    return int(m.group(1)) if m else -1

def find_checkpoints(model_dir: str, eval_latest_only: bool = False) -> List[str]:

    if os.path.isfile(model_dir):

        return [model_dir]

    if eval_latest_only:

        p = os.path.join(model_dir, "latest.pth")

        return [p] if os.path.exists(p) else []

    paths = sorted(glob.glob(os.path.join(model_dir, "epoch_*.pth")), key=checkpoint_epoch)

    if not paths and os.path.exists(os.path.join(model_dir, "latest.pth")):

        paths = [os.path.join(model_dir, "latest.pth")]

    return paths

def load_checkpoint_into_model(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:

    ckpt = torch.load(ckpt_path, map_location=device)

    sd = ckpt.get("model_state_dict", ckpt)

    missing, unexpected = model.load_state_dict(sd, strict=False)

    if missing:

        print(f"WARNING missing keys: {missing[:5]} ...", flush=True)

    if unexpected:

        print(f"WARNING unexpected keys: {unexpected[:5]} ...", flush=True)

@torch.no_grad()

def encode_records(model, loader, device: torch.device, use_cuda: bool) -> Dict[str, np.ndarray]:

    outs = {"mu_t": [], "sigma_t": [], "mu_cols": [], "sigma_cols": [], "col_mask": []}

    model.eval()

    for batch in tqdm(loader, desc="Encoding", leave=False):

        ids = batch["input_ids"].to(device, non_blocking=use_cuda)

        mask = batch["attention_mask"].to(device, non_blocking=use_cuda)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):

            out = model(ids, mask)

        outs["mu_t"].append(out.mu_t.float().cpu().numpy())

        outs["sigma_t"].append(out.sigma_t.float().cpu().numpy())

        outs["mu_cols"].append(out.mu_cols.float().cpu().numpy())

        outs["sigma_cols"].append(out.sigma_cols.float().cpu().numpy())

        outs["col_mask"].append(out.col_mask.float().cpu().numpy())

    return {k: np.concatenate(v, axis=0).astype("float32") for k, v in outs.items()}

def save_cache(path: str, q, c, query_files: List[str], candidate_files: List[str], metadata: Dict[str, Any]) -> None:

    os.makedirs(os.path.dirname(path), exist_ok=True)

    np.savez_compressed(

        path,

        query_files=np.array(query_files, dtype=object),

        candidate_files=np.array(candidate_files, dtype=object),

        q_mu_t=q["mu_t"], q_sigma_t=q["sigma_t"], q_mu_cols=q["mu_cols"], q_sigma_cols=q["sigma_cols"], q_col_mask=q["col_mask"],

        c_mu_t=c["mu_t"], c_sigma_t=c["sigma_t"], c_mu_cols=c["mu_cols"], c_sigma_cols=c["sigma_cols"], c_col_mask=c["col_mask"],

        metadata=json.dumps(metadata),

    )

def load_cache(path: str):

    data = np.load(path, allow_pickle=True)

    q = {"mu_t": data["q_mu_t"].astype("float32"), "sigma_t": data["q_sigma_t"].astype("float32"),

         "mu_cols": data["q_mu_cols"].astype("float32"), "sigma_cols": data["q_sigma_cols"].astype("float32"), "col_mask": data["q_col_mask"].astype("float32")}

    c = {"mu_t": data["c_mu_t"].astype("float32"), "sigma_t": data["c_sigma_t"].astype("float32"),

         "mu_cols": data["c_mu_cols"].astype("float32"), "sigma_cols": data["c_sigma_cols"].astype("float32"), "col_mask": data["c_col_mask"].astype("float32")}

    qf = [str(x) for x in data["query_files"].tolist()]

    cf = [str(x) for x in data["candidate_files"].tolist()]

    return q, c, qf, cf

def batch_search(cand_mu_t: np.ndarray, query_mu_t: np.ndarray, top_n: int) -> np.ndarray:

    n = min(top_n, len(cand_mu_t))

    if faiss is not None:

        index = faiss.IndexFlatIP(cand_mu_t.shape[1])

        index.add(cand_mu_t.astype("float32"))

        _, idx = index.search(query_mu_t.astype("float32"), n)

        return idx.astype("int64")

    scores = query_mu_t @ cand_mu_t.T

    part = np.argpartition(-scores, kth=n - 1, axis=1)[:, :n]

    row = np.arange(scores.shape[0])[:, None]

    order = np.argsort(-scores[row, part], axis=1)

    return part[row, order].astype("int64")

def metrics(ranked: Sequence[str], positives: Sequence[str], k: int) -> Tuple[float, float]:

    gt = set(norm_file(x) for x in positives)

    if not gt:

        return 0.0, 0.0

    top = [norm_file(x) for x in ranked[:k]]

    hits = 0

    psum = 0.0

    for i, name in enumerate(top):

        if name in gt:

            hits += 1

            psum += hits / (i + 1)

    recall = hits / len(gt)

    ap = psum / min(len(gt), k)

    return recall, ap

def eval_ranking(all_indices: np.ndarray, query_files: List[str], candidate_files: List[str], gt: Dict[str, List[str]], ks: Sequence[int]) -> List[Dict[str, Any]]:

    q2i = {q: i for i, q in enumerate(query_files)}

    rows = []

    for k in ks:

        rec_sum = map_sum = cnt = 0.0

        for q, pos in gt.items():

            if q not in q2i:

                continue

            idxs = [int(x) for x in all_indices[q2i[q]] if int(x) >= 0]

            ranked = [candidate_files[i] for i in idxs if candidate_files[i] != q]

            rec, ap = metrics(ranked, pos, int(k))

            rec_sum += rec; map_sum += ap; cnt += 1

        rows.append({"mode": "faiss_only", "k": int(k), "queries": int(cnt), "map": map_sum / max(cnt, 1), "recall": rec_sum / max(cnt, 1)})

    return rows

def gpu_greedy_match_scores(q, c, all_indices: np.ndarray, device: torch.device, block_pairs: int = 131072, lambda_reg: float = 0.5) -> np.ndarray:

    n_q, top_n = all_indices.shape

    q_ids = np.repeat(np.arange(n_q), top_n)

    c_ids = all_indices.reshape(-1)

    valid = c_ids >= 0

    scores_out = np.full((len(c_ids),), -1e9, dtype="float32")

    q_ids = q_ids[valid]

    c_ids_valid = c_ids[valid]

    out_pos = np.where(valid)[0]

    q_mu_t = torch.from_numpy(q["mu_t"]).to(device)

    c_mu_t = torch.from_numpy(c["mu_t"]).to(device)

    q_sig_t = torch.from_numpy(q["sigma_t"]).to(device)

    c_sig_t = torch.from_numpy(c["sigma_t"]).to(device)

    q_mu = torch.from_numpy(q["mu_cols"]).to(device)

    c_mu = torch.from_numpy(c["mu_cols"]).to(device)

    q_sig = torch.from_numpy(q["sigma_cols"]).to(device)

    c_sig = torch.from_numpy(c["sigma_cols"]).to(device)

    q_mask = torch.from_numpy(q["col_mask"]).to(device)

    c_mask = torch.from_numpy(c["col_mask"]).to(device)

    with torch.no_grad():

        for s in tqdm(range(0, len(q_ids), block_pairs), desc="Per-column greedy rerank", leave=False):

            e = min(s + block_pairs, len(q_ids))

            qi = torch.from_numpy(q_ids[s:e].astype("int64")).to(device)

            ci = torch.from_numpy(c_ids_valid[s:e].astype("int64")).to(device)

            table_ip = (q_mu_t[qi] * c_mu_t[ci]).sum(dim=-1)

            table_sig = torch.linalg.norm(q_sig_t[qi] - c_sig_t[ci], dim=-1)

            table_score = table_ip - lambda_reg * table_sig

            sim = torch.einsum("pcd,ped->pce", q_mu[qi], c_mu[ci])

            dist = torch.linalg.norm(q_sig[qi, :, None, :] - c_sig[ci, None, :, :], dim=-1)

            mat = sim - lambda_reg * dist

            valid_mat = (q_mask[qi, :, None] > 0) & (c_mask[ci, None, :] > 0)

            mat = mat.masked_fill(~valid_mat, -1e4)

            P, Cq, Cc = mat.shape

            matched = []

            work = mat.clone()

            max_steps = min(Cq, Cc)

            for _ in range(max_steps):

                flat = work.view(P, -1)

                val, arg = flat.max(dim=1)

                ok = val > -1e3

                matched.append(torch.where(ok, val, torch.zeros_like(val)))

                row = arg // Cc

                col = arg % Cc

                ar = torch.arange(P, device=device)

                work[ar, row, :] = -1e4

                work[ar, :, col] = -1e4

            if matched:

                m = torch.stack(matched, dim=1)

                count = (m != 0).sum(dim=1).clamp_min(1)

                col_score = m.sum(dim=1) / q_mask[qi].sum(dim=1).clamp_min(1)

            else:

                col_score = torch.zeros_like(table_score)

            final = 0.5 * table_score + 0.5 * col_score

            scores_out[out_pos[s:e]] = final.float().cpu().numpy()

    return scores_out.reshape(n_q, top_n)

def main() -> None:

    ap = argparse.ArgumentParser()

    ap.add_argument("--model_dir", required=True)

    ap.add_argument("--candidate_dir", required=True)

    ap.add_argument("--query_dir", required=True)

    ap.add_argument("--gt_path", required=True)

    ap.add_argument("--model_name", default="<MODEL_NAME>")

    ap.add_argument("--cache_dir", required=True)

    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--eval_latest_only", action="store_true")

    ap.add_argument("--top_n", type=int, default=300)

    ap.add_argument("--ks", type=int, nargs="+", default=[10, 60, 300])

    ap.add_argument("--batch_size", type=int, default=256)

    ap.add_argument("--max_length", type=int, default=160)

    ap.add_argument("--max_rows", type=int, default=3)

    ap.add_argument("--max_cols", type=int, default=12)

    ap.add_argument("--csv_nrows", type=int, default=300)

    ap.add_argument("--projection_dim", type=int, default=128)

    ap.add_argument("--rerank", choices=["faiss_only", "percol_greedy"], default="faiss_only")

    ap.add_argument("--lambda_reg", type=float, default=0.5)

    ap.add_argument("--output_csv", required=True)

    args = ap.parse_args()

    device = torch.device(args.device)

    use_cuda = device.type == "cuda" and torch.cuda.is_available()

    gt = load_gt(args.gt_path)

    query_records = build_query_records(gt, args.query_dir)

    cand_records = build_candidate_records(args.candidate_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    q_files = [x[0] for x in query_records]

    c_files = [x[0] for x in cand_records]

    q_loader = DataLoader(CsvRecordDataset(query_records, tokenizer, args.max_length, args.max_rows, args.max_cols, args.csv_nrows), batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=use_cuda)

    c_loader = DataLoader(CsvRecordDataset(cand_records, tokenizer, args.max_length, args.max_rows, args.max_cols, args.csv_nrows), batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=use_cuda)

    ckpts = find_checkpoints(args.model_dir, args.eval_latest_only)

    print("Per-column cached evaluation configuration:", flush=True)

    print(f"  checkpoints: {len(ckpts)}", flush=True)

    print(f"  query files: {len(q_files)} candidate files: {len(c_files)}", flush=True)

    rows = []

    for ckpt_path in ckpts:

        epoch = checkpoint_epoch(ckpt_path)

        cache_path = os.path.join(args.cache_dir, os.path.basename(ckpt_path) + ".percol.npz")

        if os.path.exists(cache_path):

            print(f"Loading cache: {cache_path}", flush=True)

            q, c, q_files_cached, c_files_cached = load_cache(cache_path)

            q_files, c_files = q_files_cached, c_files_cached

        else:

            print(f"Encoding checkpoint: {ckpt_path}", flush=True)

            model = PerColumnWassersteinTableEncoder(args.model_name, args.projection_dim, tokenizer.sep_token_id, args.max_cols).to(device)

            load_checkpoint_into_model(model, ckpt_path, device)

            q = encode_records(model, q_loader, device, use_cuda)

            c = encode_records(model, c_loader, device, use_cuda)

            save_cache(cache_path, q, c, q_files, c_files, {"checkpoint": ckpt_path, "epoch": epoch, "max_cols": args.max_cols})

            print(f"Saved cache: {cache_path}", flush=True)

            del model

            if use_cuda:

                torch.cuda.empty_cache()

        idx = batch_search(c["mu_t"], q["mu_t"], args.top_n + 1)

        c_index = {n: i for i, n in enumerate(c_files)}

        cleaned = []

        for qi, qn in enumerate(q_files):

            arr = [int(x) for x in idx[qi] if int(x) >= 0 and c_files[int(x)] != qn]

            if len(arr) < args.top_n:

                arr += [-1] * (args.top_n - len(arr))

            cleaned.append(arr[:args.top_n])

        all_indices = np.asarray(cleaned, dtype="int64")

        if args.rerank == "faiss_only":

            eval_rows = eval_ranking(all_indices, q_files, c_files, gt, args.ks)

        else:

            scores = gpu_greedy_match_scores(q, c, all_indices, device, lambda_reg=args.lambda_reg)

            reranked = np.argsort(-scores, axis=1)

            reranked_indices = np.take_along_axis(all_indices, reranked, axis=1)

            eval_rows = eval_ranking(reranked_indices, q_files, c_files, gt, args.ks)

            for r in eval_rows:

                r["mode"] = "percol_greedy"

        for r in eval_rows:

            r.update({"epoch": epoch, "checkpoint": os.path.basename(ckpt_path), "top_n": args.top_n})

            rows.append(r)

            print(f"Epoch {epoch} {r['mode']} MAP@{r['k']}={r['map']:.6f}, Recall@{r['k']}={r['recall']:.6f}", flush=True)

    import pandas as pd

    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    df.to_csv(args.output_csv, index=False)

    print(f"Saved evaluation CSV: {args.output_csv}", flush=True)

    print("Best rows by k:", flush=True)

    for k in args.ks:

        sub = df[df["k"] == k].sort_values("map", ascending=False).head(10)

        print(sub.to_string(index=False), flush=True)

if __name__ == "__main__":

    main()
