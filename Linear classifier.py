#!/usr/bin/env python3

import argparse, os, time, random, numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def standardize(train, *others):
    mean = train.mean(0, keepdims=True)
    std  = train.std(0, unbiased=False, keepdims=True).clamp(min=1e-6)
    out = [(x - mean) / std for x in (train, *others)]
    return out if len(out) > 1 else out[0]

def sgd_logistic_L1(Xtr, ytr, Xvl, yvl,
                    C=1.0, fit_intercept=True,
                    max_iter=500, batch=1024, tol=1e-4):
    D = Xtr.shape[1]
    model = torch.nn.Linear(D, 1, bias=fit_intercept).to(DEVICE)
    torch.nn.init.zeros_(model.weight)
    if fit_intercept:
        torch.nn.init.zeros_(model.bias)

    lam = 1.0 / C
    opt  = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    sch  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

    pos_w = (1 - ytr.mean()) / ytr.mean()
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)

    best_f1, best_state = -1, None

    for epoch in range(max_iter):
        model.train()
        for xb, yb in zip(Xtr.split(batch), ytr.split(batch)):
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            l1 = sum(p.abs().sum() for p in model.parameters())
            loss = loss + lam * l1
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            preds = (model(Xvl).sigmoid() > 0.5).float()
            f1 = f1_score(yvl.cpu(), preds.cpu())
        sch.step(-f1)

        if f1 > best_f1:
            best_f1, best_state = f1, {k: v.clone() for k, v in model.state_dict().items()}

        if opt.param_groups[0]["lr"] < tol:
            break

    model.load_state_dict(best_state)
    return model, best_f1

def print_source_breakdown(meta, idxs, name):
   
    subset = meta.iloc[idxs]
    print(f"\n{name.upper()} (tot: {len(subset)})")
    if "source" not in subset:
        print("  Nessuna colonna source trovata!")
        return
    source_counts = subset['source'].value_counts()
    for src, count in source_counts.items():
        perc = 100 * count / len(subset)
        print(f"  {src:15s}: {count:5d} ({perc:5.1f}%)")
    print("")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",   default="probing_records_multilayer.csv")
    ap.add_argument("--npz",   default="probing_embeddings_multilayer.npz")
    ap.add_argument("--out",   default="layer_scores_multilayer_gpu.csv")
    ap.add_argument("--seed",  type=int, default=42)
    ap.add_argument("--layers", type=int, nargs="*",
                    help="Indici layer da usare (es. --layers 1 7 14)")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    meta   = pd.read_csv(args.csv)
    y      = meta["label"].values.astype(np.int8)
    groups = meta["id"].values

    X = np.load(args.npz)["embeddings"]
    if X.ndim == 2:
        X = X[:, None, :]
    X = np.nan_to_num(X, nan=0.0,
                      posinf=np.finfo(np.float32).max,
                      neginf=np.finfo(np.float32).min).astype(np.float32)
    N, L, H = X.shape
    print(f"Embeddings shape: N={N}, L={L}, H={H}", flush=True)

    # SPLIT BY GROUP
    maj = {}
    for gid, lbl in zip(groups, y):
        maj.setdefault(gid, []).append(lbl)
    maj_lbl = {gid: int(np.mean(v) >= .5) for gid, v in maj.items()}

    ids = np.array(list(maj_lbl))
    labs = np.array(list(maj_lbl.values()))

    ids_tr, ids_tmp, _, _ = train_test_split(
        ids, labs, test_size=0.25, random_state=args.seed, stratify=labs)
    ids_vl, ids_ts, _, _ = train_test_split(
        ids_tmp, labs[[np.where(ids==i)[0][0] for i in ids_tmp]],
        test_size=0.4, random_state=args.seed,
        stratify=labs[[np.where(ids==i)[0][0] for i in ids_tmp]])

    def mask(idset): return np.isin(groups, list(idset))

    idx_train = np.where(mask(ids_tr))[0]
    idx_val   = np.where(mask(ids_vl))[0]
    idx_test  = np.where(mask(ids_ts))[0]

    X_tr, y_tr = X[idx_train], y[idx_train]
    X_vl, y_vl = X[idx_val], y[idx_val]
    X_ts, y_ts = X[idx_test], y[idx_test]

    print(f"Split → train {len(X_tr)} | val {len(X_vl)} | test {len(X_ts)}", flush=True)

    # Breakdown delle sources nei vari split
    print_source_breakdown(meta, idx_train, "train")
    print_source_breakdown(meta, idx_val,   "val")
    print_source_breakdown(meta, idx_test,  "test")

    # HYPERPARAMS & LAYER LIST
    C_grid            = [0.01, 1, 10, 100]
    fit_int_grid      = [True, False]
    layers = args.layers if args.layers is not None else list(range(L))
    print("Layers to probe:", layers, flush=True)

    results = []

    # LOOP SUI LAYER
    for layer in layers:
        print(f"\n[Layer {layer}] start", flush=True)
        Xtr = torch.tensor(X_tr[:, layer, :], device=DEVICE)
        Xvl = torch.tensor(X_vl[:, layer, :], device=DEVICE)
        Xts = torch.tensor(X_ts[:, layer, :], device=DEVICE)
        ytr_t = torch.tensor(y_tr, device=DEVICE, dtype=torch.float32).unsqueeze(1)
        yvl_t = torch.tensor(y_vl, device=DEVICE, dtype=torch.float32).unsqueeze(1)

        Xtr, Xvl, Xts = standardize(Xtr, Xvl, Xts)

        best_f1, best_cfg, best_model = -1, None, None
        for C in C_grid:
            for fi in fit_int_grid:
                print(f"  try C={C}, fit_int={fi}", flush=True)
                t0 = time.time()
                mdl, f1_val = sgd_logistic_L1(Xtr, ytr_t, Xvl, yvl_t,
                                              C=C, fit_intercept=fi)
                print(f"    val_f1={f1_val:.3f} in {time.time()-t0:.1f}s", flush=True)
                if f1_val > best_f1:
                    best_f1, best_cfg, best_model = f1_val, (C, fi), mdl

        with torch.no_grad():
            probs = best_model(Xts).sigmoid().cpu().numpy().ravel()
        preds = (probs > 0.5).astype(int)
        acc   = accuracy_score(y_ts, preds)
        f1_sc = f1_score(y_ts, preds)
        auroc = roc_auc_score(y_ts, probs)

        results.append({
            "npz": os.path.basename(args.npz),
            "layer": layer,
            "C": best_cfg[0],
            "penalty": "l1",
            "solver": "sgd",
            "fit_intercept": best_cfg[1],
            "acc": acc,
            "f1": f1_sc,
            "auroc": auroc
        })
        pd.DataFrame(results).to_csv(args.out, index=False)

        print(f"Layer {layer} best C={best_cfg[0]} fit_int={best_cfg[1]} | "
              f"Acc={acc:.3f} F1={f1_sc:.3f} AUROC={auroc:.3f}", flush=True)

    print(f"\n[✓] Risultati finali salvati → {args.out}", flush=True)

if __name__ == "__main__":
    main()
