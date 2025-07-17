#!/usr/bin/env python
import argparse
import os, time
import yaml
import torch
import numpy as np
import GPy
from model.models import predict_bnn, predict_gpr

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["bnn","gpr"], required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--iteration", type=int, default=0)
    p.add_argument("--n_new_samples", type=int, default=5)
    p.add_argument("--pipeline_dir", required=True)
    args = p.parse_args()

#    args = argparse.Namespace(
#            model     = "gpr",
#            config    = '/pscratch/sd/u/usatlas/globus-compute-test/Tianle_test/nano-confinement/rose_exp_1/config/gpr.yaml',
#            iteration = 1,
#            data_dir  = '/pscratch/sd/u/usatlas/globus-compute-test/Tianle_test/nano-confinement/rose_exp_1/experiment/test_01/'
#            )

    cfg = yaml.safe_load(open(args.config))
    in_dir   = os.path.join(args.pipeline_dir, f"iter_{args.iteration:03d}")
    out_dir  = os.path.join(args.pipeline_dir, f"iter_{args.iteration+1:03d}")
    os.makedirs(out_dir, exist_ok=False)

    x_train  = np.load(os.path.join(in_dir,  'scaled_x_train.npy'),  mmap_mode='r')
    y_train  = np.load(os.path.join(in_dir,  'scaled_y_train.npy'),  mmap_mode='r')
    x_remain = np.load(os.path.join(in_dir,  'scaled_x_remain.npy'), mmap_mode='r')
    y_remain = np.load(os.path.join(in_dir,  'scaled_y_remain.npy'), mmap_mode='r')

    if args.model == "gpr":
        model = GPy.load(os.path.join(in_dir, "model.pkl"))
        _, std_remain = predict_gpr(model, x_remain)
        idx = np.argsort(std_remain.flatten())[-args.n_new_samples:]
    else:
        model = torch.load(os.path.join(in_dir, "model.pkl"))
        _, std_remain = predict_bnn(model, torch.FloatTensor(x_remain), n_samples=cfg["n_mc_samples"])
        idx = np.argsort(std_remain.cpu().numpy())[-args.n_samples:]

    x_train_new = np.vstack([x_train, x_remain[idx]])
    y_train_new = np.vstack([y_train, y_remain[idx]])
    mask = np.ones(len(x_remain),bool)
    mask[idx]=False
    x_remain_new = x_remain[mask]
    y_remain_new = y_remain[mask]

    np.save(os.path.join(out_dir, 'scaled_x_train.npy'),  x_train_new)
    np.save(os.path.join(out_dir, 'scaled_y_train.npy'),  y_train_new)
    np.save(os.path.join(out_dir, 'scaled_x_remain.npy'), x_remain_new)
    np.save(os.path.join(out_dir, 'scaled_y_remain.npy'), y_remain_new)

    print(f"[active] done iteration {args.iteration}")

if __name__=="__main__":
    main()
