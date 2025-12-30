import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


SEED = 7
DEVICE = "cpu"
OUT_DIR = "wyniki"

EPOCHS_BIN = 120
LR_BIN = 0.05
BATCH_BIN = 16

EPOCHS_MLP = 180
LR_MLP = 0.01
BATCH_MLP = 16

H_LIST = [2, 4, 8, 16, 32, 64]

BINARY_PAIR = (1, 2)
TRAIN_FRACS_BIN = [0.8, 0.7, 0.6]

REPEATS = 10


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def to_loader(X: np.ndarray, y: np.ndarray, batch: int, shuffle: bool):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)


def standardize(X_train: np.ndarray, X_test: np.ndarray):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train).astype(np.float32)
    X_test = sc.transform(X_test).astype(np.float32)
    return X_train, X_test


def acc_binary(logits: torch.Tensor, y: torch.Tensor):
    pred = (torch.sigmoid(logits) >= 0.5).long()
    return (pred == y.long()).float().mean().item()


def acc_multiclass(logits: torch.Tensor, y: torch.Tensor):
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()


def confusion_matrix_3(y_true: np.ndarray, y_pred: np.ndarray):
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def train_binary_model(train_loader, test_loader):
    model = nn.Linear(4, 1).to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=LR_BIN)
    loss_fn = nn.BCEWithLogitsLoss()

    hist = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}

    for _ in range(EPOCHS_BIN):
        model.train()
        tl, ta, n = 0.0, 0.0, 0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE).float()

            logits = model(xb).squeeze(1)
            loss = loss_fn(logits, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            bsz = xb.size(0)
            tl += loss.item() * bsz
            ta += acc_binary(logits.detach(), yb.detach()) * bsz
            n += bsz

        model.eval()
        with torch.no_grad():
            vl, va, m = 0.0, 0.0, 0
            for xb, yb in test_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE).float()

                logits = model(xb).squeeze(1)
                loss = loss_fn(logits, yb)

                bsz = xb.size(0)
                vl += loss.item() * bsz
                va += acc_binary(logits, yb) * bsz
                m += bsz

        hist["train_loss"].append(tl / n)
        hist["train_acc"].append(ta / n)
        hist["test_loss"].append(vl / m)
        hist["test_acc"].append(va / m)

    return model, hist


def train_mlp_model(train_loader, test_loader, hidden: int):
    model = nn.Sequential(
        nn.Linear(4, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 3)
    ).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=LR_MLP)
    loss_fn = nn.CrossEntropyLoss()

    hist = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}

    for _ in range(EPOCHS_MLP):
        model.train()
        tl, ta, n = 0.0, 0.0, 0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            logits = model(xb)
            loss = loss_fn(logits, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            bsz = xb.size(0)
            tl += loss.item() * bsz
            ta += acc_multiclass(logits.detach(), yb.detach()) * bsz
            n += bsz

        model.eval()
        with torch.no_grad():
            vl, va, m = 0.0, 0.0, 0
            for xb, yb in test_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)

                logits = model(xb)
                loss = loss_fn(logits, yb)

                bsz = xb.size(0)
                vl += loss.item() * bsz
                va += acc_multiclass(logits, yb) * bsz
                m += bsz

        hist["train_loss"].append(tl / n)
        hist["train_acc"].append(ta / n)
        hist["test_loss"].append(vl / m)
        hist["test_acc"].append(va / m)

    return model, hist


def plot_curves(hist, title, path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].plot(hist["train_loss"], label="train")
    ax[0].plot(hist["test_loss"], label="test")
    ax[0].set_title("loss")
    ax[0].set_xlabel("epoch")
    ax[0].legend()

    ax[1].plot(hist["train_acc"], label="train")
    ax[1].plot(hist["test_acc"], label="test")
    ax[1].set_title("accuracy")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylim(0.0, 1.05)
    ax[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_acc_vs_hidden(xs, mean, std, path):
    plt.figure(figsize=(6, 4))
    plt.errorbar(xs, mean, yerr=std, marker="o", capsize=4)
    plt.xscale("log", base=2)

    y_min = min(mean) - max(std)
    y_max = max(mean) + max(std)

    y0 = max(0.0, y_min - 0.02)
    y1 = min(1.0, y_max + 0.02)

    if y1 - y0 < 0.12:
        mid = (y0 + y1) / 2
        y0 = max(0.0, mid - 0.06)
        y1 = min(1.0, mid + 0.06)

    plt.ylim(y0, y1)

    start = np.floor(y0 * 100) / 100
    stop = np.ceil(y1 * 100) / 100
    ticks = np.arange(start, stop + 1e-9, 0.02)
    plt.yticks(ticks)

    plt.xlabel("H")
    plt.ylabel("test accuracy (mean ± std)")
    plt.title("MLP: accuracy vs hidden neurons")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_cm(cm, names, path):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.xticks([0, 1, 2], names, rotation=30, ha="right")
    plt.yticks([0, 1, 2], names)
    for i in range(3):
        for j in range(3):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.title("Confusion matrix (best MLP)")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    set_seed(SEED)
    ensure_dir(OUT_DIR)

    iris = load_iris()
    X = iris.data.astype(np.float32)
    y = iris.target.astype(np.int64)
    names = list(iris.target_names)

    a, b = BINARY_PAIR
    mask = (y == a) | (y == b)
    Xb = X[mask]
    yb = (y[mask] == b).astype(np.int64)

    bin_results = []
    bin_hist_80 = None

    for frac in TRAIN_FRACS_BIN:
        Xtr, Xte, ytr, yte = train_test_split(
            Xb, yb, train_size=frac, random_state=SEED, stratify=yb
        )
        Xtr, Xte = standardize(Xtr, Xte)

        tr_loader = to_loader(Xtr, ytr, batch=BATCH_BIN, shuffle=True)
        te_loader = to_loader(Xte, yte, batch=BATCH_BIN, shuffle=False)

        _, hist = train_binary_model(tr_loader, te_loader)

        bin_results.append({
            "train_frac": float(frac),
            "test_size": int(len(yte)),
            "final_test_acc": float(hist["test_acc"][-1]),
            "final_test_loss": float(hist["test_loss"][-1]),
        })

        if abs(frac - 0.8) < 1e-9:
            bin_hist_80 = hist

    if bin_hist_80 is not None:
        plot_curves(
            bin_hist_80,
            f"Binary perceptron: {names[a]} vs {names[b]} (train=80/test=20)",
            os.path.join(OUT_DIR, "binary_curves.png")
        )

    all_runs = []
    for r in range(REPEATS):
        seed_r = SEED + 1000 * r
        set_seed(seed_r)

        Xtr, Xte, ytr, yte = train_test_split(
            X, y, train_size=0.8, random_state=seed_r, stratify=y
        )
        Xtr, Xte = standardize(Xtr, Xte)

        tr_loader = to_loader(Xtr, ytr, batch=BATCH_MLP, shuffle=True)
        te_loader = to_loader(Xte, yte, batch=BATCH_MLP, shuffle=False)

        run = []
        for h in H_LIST:
            _, hist = train_mlp_model(tr_loader, te_loader, hidden=h)
            run.append(float(hist["test_acc"][-1]))
        all_runs.append(run)

    all_runs = np.array(all_runs, dtype=np.float32)
    mean_acc = all_runs.mean(axis=0)
    std_acc = all_runs.std(axis=0)

    best_idx = int(np.argmax(mean_acc))
    best_H = int(H_LIST[best_idx])

    set_seed(SEED)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, train_size=0.8, random_state=SEED, stratify=y
    )
    Xtr, Xte = standardize(Xtr, Xte)

    tr_loader = to_loader(Xtr, ytr, batch=BATCH_MLP, shuffle=True)
    te_loader = to_loader(Xte, yte, batch=BATCH_MLP, shuffle=False)

    best_model, best_hist = train_mlp_model(tr_loader, te_loader, hidden=best_H)

    plot_acc_vs_hidden(
        H_LIST,
        mean_acc.tolist(),
        std_acc.tolist(),
        os.path.join(OUT_DIR, "mlp_acc_vs_hidden.png")
    )

    plot_curves(
        best_hist,
        f"Best MLP (4->{best_H}->3)",
        os.path.join(OUT_DIR, "mlp_best_curves.png")
    )

    best_model.eval()
    with torch.no_grad():
        xb = torch.tensor(Xte, dtype=torch.float32).to(DEVICE)
        logits = best_model(xb).cpu().numpy()
        y_pred = logits.argmax(axis=1)

    cm = confusion_matrix_3(yte, y_pred)
    plot_cm(cm, names, os.path.join(OUT_DIR, "mlp_best_cm.png"))

    sweep = []
    for i, h in enumerate(H_LIST):
        sweep.append({
            "H": int(h),
            "mean_test_acc": float(mean_acc[i]),
            "std_test_acc": float(std_acc[i])
        })

    summary = {
        "device": DEVICE,
        "repeats": REPEATS,
        "binary": {
            "pair": [names[a], names[b]],
            "results": bin_results
        },
        "mlp": {
            "hidden_list": H_LIST,
            "sweep": sweep,
            "best_H": best_H,
            "best_mean_test_acc": float(mean_acc[best_idx]),
            "best_std_test_acc": float(std_acc[best_idx]),
            "confusion_matrix": cm.tolist()
        }
    }

    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
