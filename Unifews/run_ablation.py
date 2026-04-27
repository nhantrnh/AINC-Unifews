import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import ptflops
import torch
import torch.nn as nn
import torch.optim as optim

from utils.logger import Logger, ModelLogger, prepare_opt
from utils.loader import load_edgelist
import utils.metric as metric
from archs import identity_n_norm, flops_modules_dict
import archs.models as models


np.set_printoptions(
    linewidth=160,
    edgeitems=5,
    threshold=20,
    formatter={"float": lambda x: "% 9.3e" % x},
)
torch.set_printoptions(linewidth=160, edgeitems=5)


def parse_float_list(v: str):
    if v is None:
        return []
    vals = []
    for tok in v.split(","):
        tok = tok.strip()
        if tok:
            vals.append(float(tok))
    return vals


def build_model(args, nfeat, nclass):
    algo_head = args.algo.split("_")[0]
    if algo_head in ["gcn2"]:
        model = models.SandwitchThr(
            nlayer=args.layer,
            nfeat=nfeat,
            nhidden=args.hidden,
            nclass=nclass,
            thr_a=args.thr_a,
            thr_w=args.thr_w,
            thr_mode=args.thr_mode,
            dropout=args.dropout,
            layer=args.algo,
        )
    elif algo_head in ["mlp"]:
        model = models.MLP(
            nlayer=args.layer,
            nfeat=nfeat,
            nhidden=args.hidden,
            nclass=nclass,
            thr_w=args.thr_w,
            dropout=args.dropout,
            layer="mlp",
        )
    elif "sgc" in args.algo or "appnp" in args.algo:
        model = models.MLP(
            nlayer=args.layer,
            nfeat=nfeat,
            nhidden=args.hidden,
            nclass=nclass,
            dropout=args.dropout,
            thr_w=args.thr_w,
            layer=args.algo,
        )
    else:
        model = models.GNNThr(
            nlayer=args.layer,
            nfeat=nfeat,
            nhidden=args.hidden,
            nclass=nclass,
            thr_a=args.thr_a,
            thr_w=args.thr_w,
            thr_mode=args.thr_mode,
            dropout=args.dropout,
            layer=args.algo,
        )
    model.reset_parameters()
    return model


def run_one(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.dev >= 0:
        with torch.cuda.device(args.dev):
            torch.cuda.manual_seed(args.seed)

    if "_" not in args.algo:
        args.thr_a, args.thr_w = 0.0, 0.0

    flag_run = f"ablation-{args.seed}-{args.thr_a:.1e}-{args.thr_w:.1e}-{args.thr_mode}"
    logger = Logger(args.data, args.algo, flag_run=flag_run)
    logger.save_opt(args)
    model_logger = ModelLogger(
        logger,
        patience=args.patience,
        cmp="max",
        prefix="model" + args.suffix,
        storage="state_ram" if args.data in ["cs", "physics", "arxiv"] else "state_gpu",
    )
    stopwatch = metric.Stopwatch()

    adj, feat, labels, idx, nfeat, nclass = load_edgelist(
        datastr=args.data,
        datapath=args.path,
        inductive=args.inductive,
        multil=args.multil,
        seed=args.seed,
    )

    model = build_model(args, nfeat, nclass)
    model.kwargs["diag"] = None
    diag = model.kwargs["diag"]
    adj["train"] = identity_n_norm(
        adj["train"],
        edge_weight=None,
        num_nodes=feat["train"].shape[0],
        rnorm=model.kwargs["rnorm"],
        diag=diag,
    )

    if args.dev >= 0:
        model = model.cuda(args.dev)

    # Register model so ModelLogger can save/load best checkpoints.
    model_logger.register(model, save_init=False)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, threshold=1e-4, patience=15
    )
    loss_fn = nn.BCEWithLogitsLoss() if args.multil else nn.CrossEntropyLoss()

    def train(epoch):
        model.train()
        if epoch < args.epochs // 2:
            model.set_scheme("pruneall", "pruneall")
        else:
            model.set_scheme("pruneall", "pruneinc")

        x, y = feat["train"].cuda(args.dev), labels["train"].cuda(args.dev)
        edge_idx = adj["train"]
        if isinstance(edge_idx, tuple):
            edge_idx = (edge_idx[0].cuda(args.dev), edge_idx[1].cuda(args.dev))
        else:
            edge_idx = edge_idx.cuda(args.dev)

        stopwatch.reset()
        stopwatch.start()
        optimizer.zero_grad()
        output = model(x, edge_idx, node_lock=torch.Tensor([]))[idx["train"]]
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        stopwatch.pause()
        return loss.item(), stopwatch.time

    def evaluate(split):
        model.eval()
        model.set_scheme("keep", "keep")

        x = feat["train"].cuda(args.dev)
        y = labels[split].cuda(args.dev)
        edge_idx = adj["train"] if split != "test" else adj["test"]
        if isinstance(edge_idx, tuple):
            edge_idx = (edge_idx[0].cuda(args.dev), edge_idx[1].cuda(args.dev))
        else:
            edge_idx = edge_idx.cuda(args.dev)

        idx_split = idx[split]
        calc = metric.F1Calculator(nclass)
        stopwatch.reset()
        with torch.no_grad():
            stopwatch.start()
            output = model(x, edge_idx, node_lock=idx_split)[idx_split]
            stopwatch.pause()
            output = output.cpu().detach()
            ylabel = y.cpu().detach()
            if args.multil:
                output = torch.where(output > 0, torch.tensor(1), torch.tensor(0))
            else:
                output = output.argmax(dim=1)
            calc.update(ylabel, output)
        res = calc.compute(("macro" if args.multil else "micro"))
        return res, stopwatch.time

    def cal_flops(split):
        model.eval()
        model.set_scheme("keep", "keep")
        x = feat["train"].cuda(args.dev) if split != "test" else feat["test"].cuda(args.dev)
        edge_idx = adj["train"] if split != "test" else adj["test"]
        if isinstance(edge_idx, tuple):
            edge_idx = (edge_idx[0].cuda(args.dev), edge_idx[1].cuda(args.dev))
        else:
            edge_idx = edge_idx.cuda(args.dev)

        handle = model.register_forward_hook(models.GNNThr.batch_counter_hook)
        model.__batch_counter_handle__ = handle
        macs, _ = ptflops.get_model_complexity_info(
            model,
            (1, 1, 1),
            input_constructor=lambda _: {"x": x, "edge_idx": edge_idx},
            custom_modules_hooks=flops_modules_dict,
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
        )
        return macs / 1e9

    if args.dev >= 0:
        with torch.cuda.device(args.dev):
            torch.cuda.empty_cache()

    time_tol, macs_tol = metric.Accumulator(), metric.Accumulator()
    epoch_conv, acc_best = 0, 0

    for epoch in range(1, args.epochs + 1):
        loss_train, time_epoch = train(epoch)
        time_tol.update(time_epoch)
        acc_val, _ = evaluate("val")
        scheduler.step(acc_val)
        macs_epoch = cal_flops("train")
        macs_tol.update(macs_epoch)

        print(
            f"Epoch:{epoch:04d} | train loss:{loss_train:.4f}, "
            f"val acc:{acc_val:.4f}, time:{time_tol.val:.4f}, macs:{macs_tol.val:.4f}",
            flush=True,
        )

        acc_best = model_logger.save_best(acc_val, epoch=epoch)
        if not model_logger.is_early_stop(epoch=epoch):
            epoch_conv = max(0, epoch - model_logger.patience)

    model = model_logger.load("best")
    if args.dev >= 0:
        model = model.cuda(args.dev)

    adj["test"] = identity_n_norm(
        adj["test"],
        edge_weight=None,
        num_nodes=feat["test"].shape[0],
        rnorm=model.kwargs["rnorm"],
        diag=model.kwargs["diag"],
    )

    acc_test, time_test = evaluate("test")
    macs_test = cal_flops("test")
    numel_a, numel_w = model.get_numel()

    return {
        "seed": args.seed,
        "thr_a": args.thr_a,
        "thr_w": args.thr_w,
        "thr_mode": args.thr_mode,
        "acc_val_best": float(acc_best),
        "acc_test": float(acc_test),
        "conv_epoch": int(epoch_conv),
        "epoch": int(args.epochs),
        "time_train_s": float(time_tol.val),
        "time_train_avg_ms": float(time_tol.avg * 1000),
        "macs_train_g": float(macs_tol.val),
        "macs_train_avg_g": float(macs_tol.avg),
        "time_test_s": float(time_test),
        "macs_test_g": float(macs_test),
        "numel_a_k": float(numel_a),
        "numel_w_k": float(numel_w),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--seed", type=int, default=11, help="Random seed.")
    parser.add_argument("-v", "--dev", type=int, default=0, help="Device id.")
    parser.add_argument("-c", "--config", type=str, default="cora", help="Config file name.")
    parser.add_argument("-m", "--algo", type=str, default="gcn_thr", help="Model name")
    parser.add_argument("-n", "--suffix", type=str, default="", help="Save name suffix.")
    parser.add_argument("-a", "--thr_a", type=float, default=None, help="Threshold of adj.")
    parser.add_argument("-w", "--thr_w", type=float, default=None, help="Threshold of weight.")
    parser.add_argument("-l", "--layer", type=int, default=None, help="Layer.")

    parser.add_argument("--thr_mode", type=str, default="fixed", choices=["adaptive", "fixed"])
    parser.add_argument("--thr_a_list", type=str, default="0.2,0.5,1.0")
    parser.add_argument("--thr_w_list", type=str, default="0.5")
    parser.add_argument("--pair_mode", type=str, default="cartesian", choices=["cartesian", "zip"])
    parser.add_argument("--seed_list", type=str, default=None, help="Comma list, e.g. 11,12,13")
    parser.add_argument("--result_file", type=str, default="")

    base_args = prepare_opt(parser)

    if base_args.algo != "gcn_thr":
        raise ValueError("run_ablation.py is intended for gcn_thr only. Set --algo gcn_thr.")

    thr_a_list = parse_float_list(base_args.thr_a_list)
    thr_w_list = parse_float_list(base_args.thr_w_list)
    if not thr_a_list:
        thr_a_list = [float(base_args.thr_a)] if base_args.thr_a is not None else [0.5]
    if not thr_w_list:
        thr_w_list = [float(base_args.thr_w)] if base_args.thr_w is not None else [0.5]

    if base_args.seed_list:
        seed_list = [int(x.strip()) for x in base_args.seed_list.split(",") if x.strip()]
    else:
        seed_list = [int(base_args.seed)]

    grid = []
    if base_args.pair_mode == "zip":
        for a, w in zip(thr_a_list, thr_w_list):
            grid.append((a, w))
    else:
        for a in thr_a_list:
            for w in thr_w_list:
                grid.append((a, w))

    all_results = []
    print("=== Ablation Start ===")
    print(f"algo={base_args.algo}, mode={base_args.thr_mode}, data={base_args.data}")
    print(f"grid={grid}, seeds={seed_list}")

    for seed in seed_list:
        for thr_a, thr_w in grid:
            args = copy.deepcopy(base_args)
            args.seed = seed
            args.thr_a = float(thr_a)
            args.thr_w = float(thr_w)
            print(f"\n[RUN] seed={seed} thr_a={thr_a:.3e} thr_w={thr_w:.3e} mode={args.thr_mode}")
            res = run_one(args)
            all_results.append(res)
            print(
                "[DONE] "
                f"acc_test={res['acc_test']:.5f}, "
                f"macs_test={res['macs_test_g']:.4f}G, "
                f"numel_a={res['numel_a_k']:.3f}k, numel_w={res['numel_w_k']:.3f}k"
            )

    print("\n=== Ablation Summary ===")
    for r in all_results:
        print(
            f"seed={r['seed']:>3d} | thr_a={r['thr_a']:.3e} | thr_w={r['thr_w']:.3e} | "
            f"acc_test={r['acc_test']:.5f} | macs_test={r['macs_test_g']:.4f}G | "
            f"numel_a={r['numel_a_k']:.3f}k | numel_w={r['numel_w_k']:.3f}k"
        )

    # Print compact table like run_fb.py logs.
    print("\nModel|  Seed|     ThA|     ThW|    Acc|  Cn|  EP|  Ttrain|  Ctrain|   Ttest|   CTest|  NumelA|  NumelW")
    for r in all_results:
        print(
            f"{base_args.algo:10s},{r['seed']:6d},{r['thr_a']:7.2e},{r['thr_w']:7.2e},"
            f"{r['acc_test']:7.5f},{r['conv_epoch']:4d},{r['epoch']:4d},"
            f"{r['time_train_s']:8.4f},{r['macs_train_g']:8.3f},"
            f"{r['time_test_s']:8.4f},{r['macs_test_g']:8.4f},{r['numel_a_k']:8.3f},{r['numel_w_k']:8.3f}"
        )

    if base_args.result_file:
        out_path = Path(base_args.result_file)
    else:
        out_path = Path("save") / base_args.data / base_args.algo / f"ablation_{base_args.thr_mode}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()
