#!/usr/bin/env python3
"""
run_nlp_practice_no_ppt.py

Versão do script focada apenas no pipeline NLP e nos experimentos:
- treina SentencePiece tokenizers com vários vocab sizes
- treina um modelo simples (Embedding -> mean pooling -> Linear)
- varía: vocab size, dataset size, embedding dim
- salva métricas, plots (learning curves, embeddings PCA+t-SNE)
- salva tabela CSV com resultados

Ajuste EXPERIMENT_SETTINGS, NUM_EPOCHS, etc. conforme sua máquina.
"""
import os
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import sentencepiece as spm
from datasets import load_dataset

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# -------------------------
# Config (edite conforme necessário)
# -------------------------
OUTDIR = Path("outputs")
FIGS_DIR = OUTDIR / "figs"
SP_MODELS_DIR = OUTDIR / "spm_models"
os.makedirs(FIGS_DIR, exist_ok=True)
os.makedirs(SP_MODELS_DIR, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# dataset configuration
LOAD_DATASET_NAME = "imdb"  # HuggingFace dataset name
MAX_SEQ_LEN = 256  # truncation/padding length
BATCH_SIZE = 64
NUM_EPOCHS = 8
LR = 1e-3

# plotting / embedding projection
EMBED_PLOT_SAMPLE = 1000  # pontos para t-SNE
TSNE_PERPLEXITY = 30

# experiment grid (mude conforme desejar)
EXPERIMENT_SETTINGS = {
    "vocab_sizes": [2000, 10000, 30000],
    "dataset_sizes": [2000, 10000, 25000],
    "embed_dims": [16, 64, 256]
}

RUNS_PER_SETTING = 1

# -------------------------
# Utils
# -------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def sample_dataset_split(ds, n_total, seed=SEED):
    n_total = min(n_total, len(ds))
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    sel = idxs[:n_total]
    texts = [ds[i]["text"] for i in sel]
    labels = [ds[i]["label"] for i in sel]
    return texts, labels

# -------------------------
# SentencePiece helpers
# -------------------------
def train_sentencepiece_from_texts(texts, model_prefix, vocab_size, model_type="unigram"):
    tmp_file = SP_MODELS_DIR / f"{model_prefix}_corpus.txt"
    with open(tmp_file, "w", encoding="utf8") as f:
        for t in texts:
            f.write(t.replace("\n", " ") + "\n")
    model_prefix_path = SP_MODELS_DIR / model_prefix
    spm.SentencePieceTrainer.Train(
        input=str(tmp_file),
        model_prefix=str(model_prefix_path),
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=0.9995,
        pad_id=0, unk_id=1, bos_id=-1, eos_id=-1,
    )
    return str(model_prefix_path) + ".model"

def load_sentencepiece_model(model_path):
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    return sp

def encode_and_pad(sp, texts, max_len, pad_id=0):
    out = np.full((len(texts), max_len), pad_id, dtype=np.int64)
    for i, t in enumerate(texts):
        ids = sp.EncodeAsIds(t)[:max_len]
        out[i, :len(ids)] = ids
    return out

# -------------------------
# Dataset / Model
# -------------------------
class TextDataset(Dataset):
    def __init__(self, ids_array, labels):
        self.ids = ids_array
        self.labels = np.array(labels, dtype=np.int64)
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, idx):
        return torch.tensor(self.ids[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float32)

class SimpleClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, padding_idx=0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.fc = nn.Linear(embed_dim, 1)
    def forward(self, x):
        emb = self.embed(x)  # B,L,D
        mask = (x != 0).unsqueeze(-1).float()  # B,L,1
        sum_emb = (emb * mask).sum(dim=1)  # B,D
        denom = mask.sum(dim=1).clamp(min=1.0)
        doc_emb = sum_emb / denom
        logits = self.fc(doc_emb).squeeze(-1)
        return logits, doc_emb

# -------------------------
# Train / Eval
# -------------------------
def train_one_epoch(model, dataloader, opt, device):
    model.train()
    total_loss = 0.0
    preds = []
    trues = []
    for x, y in dataloader:
        x = x.to(device); y = y.to(device)
        opt.zero_grad()
        logits, _ = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * x.size(0)
        with torch.no_grad():
            prob = torch.sigmoid(logits).cpu().numpy()
            preds.extend((prob >= 0.5).astype(int).tolist())
            trues.extend(y.cpu().numpy().astype(int).tolist())
    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(trues, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(trues, preds, average="binary", zero_division=0)
    return avg_loss, acc, prec, rec, f1

def eval_model(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    preds = []
    trues = []
    embeddings = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device); y = y.to(device)
            logits, doc_emb = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            total_loss += loss.item() * x.size(0)
            prob = torch.sigmoid(logits).cpu().numpy()
            preds.extend((prob >= 0.5).astype(int).tolist())
            trues.extend(y.cpu().numpy().astype(int).tolist())
            embeddings.append(doc_emb.cpu().numpy())
    embeddings = np.vstack(embeddings) if len(embeddings) else np.zeros((0, model.embed.embedding_dim))
    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(trues, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(trues, preds, average="binary", zero_division=0)
    return avg_loss, acc, prec, rec, f1, embeddings, np.array(trues)

# -------------------------
# Plot helpers
# -------------------------
def plot_learning_curves(history, title, outpath):
    epochs = list(range(1, len(history['train_loss'])+1))
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, history['train_loss'], label='train_loss')
    plt.plot(epochs, history['val_loss'], label='val_loss')
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss"); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(epochs, history['train_acc'], label='train_acc')
    plt.plot(epochs, history['val_acc'], label='val_acc')
    plt.xlabel("Epoch"); plt.ylabel("Acc"); plt.title("Accuracy"); plt.legend()
    plt.suptitle(title)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig(outpath)
    plt.close()

def plot_embedding_2d(embeddings, labels, title, outpath, sample_size=EMBED_PLOT_SAMPLE):
    n = embeddings.shape[0]
    if n == 0:
        print("No embeddings to plot for", title)
        return
    idx = np.arange(n)
    if n > sample_size:
        np.random.seed(SEED)
        idx = np.random.choice(np.arange(n), size=sample_size, replace=False)
    X = embeddings[idx]
    y = labels[idx]
    pca = PCA(n_components=min(50, X.shape[1]))
    X_p = pca.fit_transform(X)
    tsne = TSNE(n_components=2, perplexity=min(30, max(5, len(X_p)//3)), init="random", random_state=SEED)
    X2 = tsne.fit_transform(X_p)
    plt.figure(figsize=(6,6))
    for lab in np.unique(y):
        sel = y==lab
        plt.scatter(X2[sel,0], X2[sel,1], label=str(lab), alpha=0.6, s=8)
    plt.legend(title="label")
    plt.title(title + " (t-SNE)")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

# -------------------------
# Experiment runner
# -------------------------
def run_single_experiment(texts_train, labels_train, texts_val, labels_val, sp_model_path,
                          vocab_size, embed_dim, max_len=MAX_SEQ_LEN, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS):
    sp = load_sentencepiece_model(sp_model_path)
    pad_id = 0
    train_ids = encode_and_pad(sp, texts_train, max_len=max_len, pad_id=pad_id)
    val_ids = encode_and_pad(sp, texts_val, max_len=max_len, pad_id=pad_id)
    train_ds = TextDataset(train_ids, labels_train)
    val_ds = TextDataset(val_ids, labels_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    model = SimpleClassifier(vocab_size, embed_dim, padding_idx=pad_id).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    history = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}
    best_val_f1 = -1.0
    best_state = None
    for epoch in range(1, num_epochs+1):
        tr_loss, tr_acc, tr_p, tr_r, tr_f1 = train_one_epoch(model, train_loader, opt, DEVICE)
        val_loss, val_acc, val_p, val_r, val_f1, val_embs, val_labels = eval_model(model, val_loader, DEVICE)
        history["train_loss"].append(tr_loss); history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc); history["val_acc"].append(val_acc)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k:v.cpu().clone() if isinstance(v, torch.Tensor) else v for k,v in model.state_dict().items()}
        print(f"Epoch {epoch}/{num_epochs} | tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    # final eval
    val_loss, val_acc, val_p, val_r, val_f1, val_embs, val_labels = eval_model(model, val_loader, DEVICE)
    train_loss, train_acc, train_p, train_r, train_f1, train_embs, train_labels = eval_model(model, train_loader, DEVICE)
    metrics = {
        "vocab_size": vocab_size,
        "embed_dim": embed_dim,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "train_f1": train_f1,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_f1": val_f1
    }
    return metrics, history, (train_embs, train_labels), (val_embs, val_labels)

# -------------------------
# Main
# -------------------------
def main():
    print("Loading dataset:", LOAD_DATASET_NAME)
    ds = load_dataset(LOAD_DATASET_NAME)
    train_split = ds["train"]
    test_split = ds["test"]

    results = []
    fig_paths = []

    tokenizer_corpus_texts = [train_split[i]["text"] for i in range(min(20000, len(train_split)))]
    print("Prepared tokenizer corpus size:", len(tokenizer_corpus_texts))

    fixed_vocab = EXPERIMENT_SETTINGS["vocab_sizes"][1] if len(EXPERIMENT_SETTINGS["vocab_sizes"])>1 else EXPERIMENT_SETTINGS["vocab_sizes"][0]
    fixed_ds = EXPERIMENT_SETTINGS["dataset_sizes"][1] if len(EXPERIMENT_SETTINGS["dataset_sizes"])>1 else EXPERIMENT_SETTINGS["dataset_sizes"][0]
    fixed_dim = EXPERIMENT_SETTINGS["embed_dims"][1] if len(EXPERIMENT_SETTINGS["embed_dims"])>1 else EXPERIMENT_SETTINGS["embed_dims"][0]

    # Pre-sample datasets
    pre_sampled = {}
    for ds_size in set(EXPERIMENT_SETTINGS["dataset_sizes"]):
        texts, labels = sample_dataset_split(train_split, ds_size, seed=SEED)
        n_val = max(50, int(0.1 * len(texts)))
        texts_train, labels_train = texts[n_val:], labels[n_val:]
        texts_val, labels_val = texts[:n_val], labels[:n_val]
        pre_sampled[ds_size] = (texts_train, labels_train, texts_val, labels_val)
        print(f"Prepared sample for ds_size={ds_size}: train={len(texts_train)} val={len(texts_val)}")

    # 1) Vary vocab sizes
    for vocab_size in EXPERIMENT_SETTINGS["vocab_sizes"]:
        sp_model_name = f"spm_vocab{vocab_size}"
        sp_model_path = Path(SP_MODELS_DIR) / (sp_model_name + ".model")
        if not sp_model_path.exists():
            print("Training SentencePiece vocab_size=", vocab_size)
            model_path = train_sentencepiece_from_texts(tokenizer_corpus_texts, sp_model_name, vocab_size=vocab_size)
            sp_model_path = Path(model_path)
        else:
            print("Found existing SP model:", sp_model_path)
        texts_train, labels_train, texts_val, labels_val = pre_sampled[fixed_ds]
        for run in range(RUNS_PER_SETTING):
            print(f"Experiment: vocab={vocab_size} ds={fixed_ds} dim={fixed_dim} run={run}")
            metrics, history, train_embs_pack, val_embs_pack = run_single_experiment(
                texts_train, labels_train, texts_val, labels_val,
                str(sp_model_path), vocab_size, fixed_dim, max_len=MAX_SEQ_LEN, num_epochs=NUM_EPOCHS
            )
            metrics.update({"experiment":"vocab_variation", "dataset_size": fixed_ds, "run": run})
            results.append(metrics)
            curve_path = FIGS_DIR / f"curve_vocab{vocab_size}_ds{fixed_ds}_dim{fixed_dim}_run{run}.png"
            plot_learning_curves(history, f"vocab={vocab_size} dim={fixed_dim}", curve_path)
            fig_paths.append(curve_path)
            train_embs, train_labels = train_embs_pack
            val_embs, val_labels = val_embs_pack
            emb_path = FIGS_DIR / f"emb_vocab{vocab_size}_ds{fixed_ds}_dim{fixed_dim}_run{run}.png"
            plot_embedding_2d(np.vstack([train_embs, val_embs]), np.hstack([train_labels, val_labels]), f"vocab={vocab_size}", emb_path)
            fig_paths.append(emb_path)

    # 2) Vary dataset size
    for ds_size in EXPERIMENT_SETTINGS["dataset_sizes"]:
        texts_train, labels_train, texts_val, labels_val = pre_sampled[ds_size]
        sp_model_name = f"spm_vocab{fixed_vocab}"
        sp_model_path = Path(SP_MODELS_DIR) / (sp_model_name + ".model")
        if not sp_model_path.exists():
            train_sentencepiece_from_texts(tokenizer_corpus_texts, sp_model_name, vocab_size=fixed_vocab)
        for run in range(RUNS_PER_SETTING):
            print(f"Experiment: ds={ds_size} vocab={fixed_vocab} dim={fixed_dim} run={run}")
            metrics, history, train_embs_pack, val_embs_pack = run_single_experiment(
                texts_train, labels_train, texts_val, labels_val,
                str(sp_model_path), fixed_vocab, fixed_dim, max_len=MAX_SEQ_LEN, num_epochs=NUM_EPOCHS
            )
            metrics.update({"experiment":"dataset_variation", "dataset_size": ds_size, "run": run})
            results.append(metrics)
            curve_path = FIGS_DIR / f"curve_ds{ds_size}_vocab{fixed_vocab}_dim{fixed_dim}_run{run}.png"
            plot_learning_curves(history, f"ds={ds_size} vocab={fixed_vocab}", curve_path)
            fig_paths.append(curve_path)
            train_embs, train_labels = train_embs_pack
            val_embs, val_labels = val_embs_pack
            emb_path = FIGS_DIR / f"emb_ds{ds_size}_vocab{fixed_vocab}_dim{fixed_dim}_run{run}.png"
            plot_embedding_2d(np.vstack([train_embs, val_embs]), np.hstack([train_labels, val_labels]), f"ds={ds_size}", emb_path)
            fig_paths.append(emb_path)

    # 3) Vary embedding dims
    sp_model_name = f"spm_vocab{fixed_vocab}"
    sp_model_path = Path(SP_MODELS_DIR) / (sp_model_name + ".model")
    if not sp_model_path.exists():
        train_sentencepiece_from_texts(tokenizer_corpus_texts, sp_model_name, vocab_size=fixed_vocab)
    texts_train, labels_train, texts_val, labels_val = pre_sampled[fixed_ds]
    for embed_dim in EXPERIMENT_SETTINGS["embed_dims"]:
        for run in range(RUNS_PER_SETTING):
            print(f"Experiment: dim={embed_dim} vocab={fixed_vocab} ds={fixed_ds} run={run}")
            metrics, history, train_embs_pack, val_embs_pack = run_single_experiment(
                texts_train, labels_train, texts_val, labels_val,
                str(sp_model_path), fixed_vocab, embed_dim, max_len=MAX_SEQ_LEN, num_epochs=NUM_EPOCHS
            )
            metrics.update({"experiment":"embeddim_variation", "dataset_size": fixed_ds, "run": run})
            results.append(metrics)
            curve_path = FIGS_DIR / f"curve_dim{embed_dim}_vocab{fixed_vocab}_ds{fixed_ds}_run{run}.png"
            plot_learning_curves(history, f"dim={embed_dim} vocab={fixed_vocab}", curve_path)
            fig_paths.append(curve_path)
            train_embs, train_labels = train_embs_pack
            val_embs, val_labels = val_embs_pack
            emb_path = FIGS_DIR / f"emb_dim{embed_dim}_vocab{fixed_vocab}_ds{fixed_ds}_run{run}.png"
            plot_embedding_2d(np.vstack([train_embs, val_embs]), np.hstack([train_labels, val_labels]), f"dim={embed_dim}", emb_path)
            fig_paths.append(emb_path)

    # Save results DataFrame
    df = pd.DataFrame(results)
    df_path = OUTDIR / "experiment_results.csv"
    df.to_csv(df_path, index=False)
    print("Saved results to", df_path)
    print("Figures saved to", FIGS_DIR)
    print("Done.")

if __name__ == "__main__":
    main()
