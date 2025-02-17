"""Super basic script to fit a GP on logP data."""

import argparse
import random
import urllib.request
from functools import lru_cache

import jax
import jax.numpy as jnp
import optax
from rdkit import Chem
from rdkit.Chem import Crippen, rdFingerprintGenerator

import tanimoto_gp


def main(rng: random.Random, fp_size: int, N_train: int, N_test: int):

    # Read a list of SMILES strings from my mol_ga github repo
    with urllib.request.urlopen(
        "https://raw.githubusercontent.com/AustinT/mol_ga/refs/heads/main/mol_ga/data/zinc250k.smiles"
    ) as f:
        smiles = [line.strip() for line in f.readlines()]

    # Split a random subset into train and test sets
    smiles = rng.sample(smiles, N_train + N_test)
    logp_val = [Crippen.MolLogP(Chem.MolFromSmiles(s)) for s in smiles]  # thing we are regressing
    smiles_train = smiles[:N_train]
    y_train = logp_val[:N_train]
    smiles_test = smiles[N_train:]
    y_test = logp_val[N_train:]

    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=1, fpSize=fp_size)

    @lru_cache(maxsize=100_000)
    def smiles_to_fp(smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        return mfpgen.GetSparseCountFingerprint(
            mol
        )  # replace with "GetCountFingerprint" for fingerprint of size fpSize

    gp = tanimoto_gp.ZeroMeanTanimotoGP(smiles_to_fp, smiles_train, y_train)
    gp_params = tanimoto_gp.TanimotoGP_Params(raw_amplitude=jnp.asarray(1.0), raw_noise=jnp.asarray(1e-2))

    print(f"Start MLL: {gp.marginal_log_likelihood(params=gp_params)}")

    @jax.jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(lambda x: -gp.marginal_log_likelihood(x))(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(gp_params)

    # Run optimization loop
    for _ in range(100):
        gp_params, opt_state, loss = step(gp_params, opt_state)

    print(f"End MLL (after optimization): {gp.marginal_log_likelihood(params=gp_params)}")
    print(f"End GP parameters (after optimization): {gp_params}")

    # Test set
    mu_test, sigma_test = gp.predict_y(gp_params, smiles_test, full_covar=False)
    print(f"Test MSE: {jnp.mean((mu_test - jnp.asarray(y_test))**2)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fp_size", type=int, default=2048)
    parser.add_argument("--N_train", type=int, default=2000)
    parser.add_argument("--N_test", type=int, default=1000)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    main(rng=rng, fp_size=args.fp_size, N_train=args.N_train, N_test=args.N_test)
