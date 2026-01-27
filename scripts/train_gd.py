import protax.model as model
from protax import protax_utils
from protax.taxonomy import CSRWrapper

import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import sparse

import scipy.sparse as sp
import time
import pandas as pd

from pathlib import Path
import random
import matplotlib.pyplot as plt
from functools import partial
import argparse

from tqdm import tqdm
import sys

from scripts.calibration import evaluate
from protax.classify import classify_file

def CE_loss(log_probs, y_ind):
    """
    Computes the cross-entropy loss between the log_probs and
    labels y_ind.

    Args:
        log_probs: Log probabilities returned by the model, shape (N, D)
        y_ind: Integer array of true class indices, shape (N,)
    """
    return -jnp.sum(jnp.take(log_probs, y_ind, axis=0))


def forward(query, ok, tree, beta, sc_mean, sc_var, N, segnum, y_ind, lvl, q_param, tree_prior, train_with_q):
    beta = jnp.take(beta, lvl, axis=0)
    X = model.get_X(query, ok, tree, N, sc_mean, sc_var)
    log_probs = model.fill_log_bprob(X, beta, tree, segnum)

    if train_with_q:
        q = jax.nn.sigmoid(q_param)
        species_prior = tree_prior[y_ind]
        row_log_probs = log_probs[y_ind]
        species_predicted = jnp.exp(row_log_probs[-1])

        species_adjusted = (q * species_prior) + ((1 - q) * species_predicted)
        species_adjusted_log = jnp.log(jnp.clip(species_adjusted, 1e-10, 1.0))

        row_log_probs = row_log_probs.at[-1].set(species_adjusted_log)
        log_probs = log_probs.at[y_ind].set(row_log_probs)

    return CE_loss(log_probs, y_ind)

f_grad_beta = jax.jit(jax.grad(forward, argnums=(3)), static_argnums=(6, 7, 12))
f_grad_q = jax.jit(jax.grad(forward, argnums=(10)), static_argnums=(6, 7, 12))
forward_jit = jax.jit(forward, static_argnums=(6, 7, 12))

def get_targ(target_dir):
    """
    Get node id for each reference sequence at lowest level
    """
    targ = pd.read_csv(target_dir)
    targ = targ.to_numpy()[:, 1:].T

    res = np.zeros((targ.shape[0],), dtype=np.int32)

    for i in range(len(targ)):
        old = -1
        for j in range(targ.shape[1]):
            if targ[i][j] == -1:
                res[i] = old
            elif j == targ.shape[1] - 1:
                res[i] = targ[i][j]
            old = targ[i][j]

    return jnp.array(res)


def mask_n2s(n2s, node_state, i):
    """
    Remove a column in node2seq
    """
    ref_mask = np.ones((n2s.shape[1],), dtype=np.int32)
    ref_mask[i] = 0
    n2s = n2s @ sp.diags(ref_mask)

    has_refs = np.array(n2s.sum(axis=1)) > 0
    empty = np.logical_not(has_refs)

    # update empty but known entries
    node_state = np.logical_or(node_state, empty)
    node_state = np.concatenate((node_state, has_refs), axis=1)

    n2s = CSRWrapper(
        data=jnp.array(n2s.data),
        indices=jnp.array(n2s.indices),
        indptr=jnp.array(n2s.indptr),
        shape=n2s.shape,
    )

    return n2s, jnp.array(node_state)


def load_params(pdir, tdir):
    par_dir = Path(pdir)
    tax_dir = Path(tdir)

    tax = np.load(tax_dir.resolve())
    par = np.load(par_dir.resolve())

    beta = par["beta"]
    sc = par["scalings"]
    lvl = tax["node_layer"]

    tax = np.load(tax_dir.resolve())
    par = np.load(par_dir.resolve())

    return beta, lvl, sc


def train(train_config, train_dir, targ_dir, model_id = ""):

    tree, params, N, segnum = protax_utils.read_model_jax(
        "models/params/model.npz", "models/ref_db/taxonomy37k.npz"
    )

    pkey = jax.random.PRNGKey(0)
    key_beta, key_q = jax.random.split(pkey)

    sigma_beta = 10.0
    sigma_q = 1.0
    beta = jax.random.normal(key_beta, (7, 4)) * sigma_beta
    q_param = jax.random.normal(key_q, ()) * sigma_q

    lr = train_config["learning_rate"]
    train_with_q = train_config["train_with_q"]
    tree_prior = tree.prior

    n2s = sp.csr_matrix(
        (tree.node2seq.data, tree.node2seq.indices, tree.node2seq.indptr),
        shape=tree.node2seq.shape,
    )
    targ = get_targ(targ_dir)
    seq_list, ok_list = protax_utils.read_refs(train_dir)
    print("Read reference sequences successfully")

    node_state = np.expand_dims(np.array(tree.node_state)[:, 0], 1)
    print("Read node state successfully")

    # params and node lvl
    _, lvl, sc = load_params("models/params/model.npz", "models/ref_db/taxonomy37k.npz")
    epoch_loss_hist = []

    print("Training model...")
    start_time = time.time()

    for e in range(train_config["num_epochs"]):
        print(f"epoch {e+1} / {train_config['num_epochs']}")

        beta_grad = 0
        q_grad = 0
        loss_sum = 0
        batch_loss = 0

        traversal = list(range(seq_list.shape[0]))
        random.shuffle(traversal)

        # minibatch
        for i in tqdm(traversal, file=sys.stderr, dynamic_ncols=True, mininterval=5):
            # mask out tree
            q_seq = seq_list[i]
            ok = ok_list[i]

            # masks out a node2seq column given reference index (~1.2 ms)
            new_node2seq, new_node_state = mask_n2s(n2s, node_state, i)     # Edited by Mohamed
            tree = tree._replace(node2seq=new_node2seq, node_state=new_node_state)

            beta_grad += f_grad_beta(
                q_seq,
                ok,
                tree,
                beta,
                params.sc_mean,
                params.sc_var,
                N,
                segnum,
                targ.at[i].get(),
                lvl,
                q_param, 
                tree_prior,
                train_with_q,
            )

            if train_with_q:
                q_grad += f_grad_q(
                    q_seq,
                    ok,
                    tree,
                    beta,
                    params.sc_mean,
                    params.sc_var,
                    N,
                    segnum,
                    targ.at[i].get(),
                    lvl,
                    q_param, 
                    tree_prior,
                    train_with_q,
                )

            batch_loss += forward_jit(
                q_seq,
                ok,
                tree,
                beta,
                params.sc_mean,
                params.sc_var,
                N,
                segnum,
                targ.at[i].get(),
                lvl,
                q_param, 
                tree_prior,
                train_with_q,
            )


            if i % train_config["batch_size"] == 0:
                beta = beta - lr * beta_grad
                beta_grad = 0

                if train_with_q:
                    # q_grad = jnp.clip(q_grad, -1.0, 1.0)  # Clip to prevent vanishing gradients
                    q_param = q_param - (lr * 5 * q_grad)
                    q_grad = 0

                loss_sum += batch_loss
                batch_loss = 0

                # grad norm
                # bflat = beta_grad.reshape(beta_grad.shape[0] * beta_grad.shape[1])    # Edited by Mohamed

        if train_with_q:
            print("q_percentage: ", jax.nn.sigmoid(q_param).item())

        epoch_loss = loss_sum / seq_list.shape[0]
        epoch_loss_hist.append(epoch_loss)
        print("total loss: ", epoch_loss)

        # save checkpoint
        mf = Path(f"models/params/model_{model_id}.npz")
        np.savez_compressed(mf.resolve(), beta=np.array(beta), scalings=sc)

        if train_config["evaluate_during_training"]:
            if e % 3 == 0:
                classify_file("models/ref_db/refs.aln", f"models/params/model_{model_id}.npz", "models/ref_db/taxonomy37k.npz", e)
                evaluate(f"training_results_{e}.csv", "models/ref_db/list_of_labels_nodeIDs.txt", e)


    plt.plot(epoch_loss_hist)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.savefig(f"loss_curve_{model_id}.png", dpi=300, bbox_inches="tight")
    plt.close()

    end_time = time.time()
    print(f"Time taken to train: {end_time - start_time} seconds")


if __name__ == "__main__":
    # parse config from command line
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--train_dir", type=str, help="Path to training data")
    parser.add_argument("--targ_dir", type=str, help="Path to target data")
    parser.add_argument("--model_id", type=str, help="Model identifier")
    args = parser.parse_args()
    train_dir = Path(args.train_dir)
    targ_dir = Path(args.targ_dir)
    model_id = args.model_id

    # training config
    tc = {
        "learning_rate": 0.03,
        "batch_size": 500,
        "num_epochs": 30,
        "train_with_q": True,
        "evaluate_during_training": False,
    }

    train(tc, train_dir, targ_dir, model_id)  # train the model
