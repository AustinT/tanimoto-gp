from typing import Any, Callable, NamedTuple

import kern_gp as kgp
from jax import numpy as jnp
from jax.nn import softplus
from rdkit import DataStructs

TRANSFORM = softplus  # fixed transform function


class TanimotoGP_Params(NamedTuple):
    # Inverse softplus of GP parameters
    raw_amplitude: jnp.ndarray
    raw_noise: jnp.ndarray


class ZeroMeanTanimotoGP:
    def __init__(self, fp_func: Callable[[str], Any], smiles_train: list[str], y_train):
        super().__init__()
        self._fp_func = fp_func
        self.set_training_data(smiles_train, y_train)

    def set_training_data(self, smiles_train: list[str], y_train: jnp.ndarray):
        self._smiles_train = smiles_train
        self._y_train = jnp.asarray(y_train)
        self._fp_train = [self._fp_func(smiles) for smiles in smiles_train]
        self._K_train_train = jnp.asarray(
            [DataStructs.BulkTanimotoSimilarity(fp, self._fp_train) for fp in self._fp_train]
        )
        self._L_cached = None

    def marginal_log_likelihood(self, params: TanimotoGP_Params) -> jnp.ndarray:
        return kgp.mll_train(
            a=TRANSFORM(params.raw_amplitude),
            s=TRANSFORM(params.raw_noise),
            k_train_train=self._K_train_train,
            y_train=self._y_train,
        )

    def predict_f(self, params: TanimotoGP_Params, smiles_test: list[str], full_covar: bool = True) -> jnp.ndarray:

        # Construct kernel matrices
        fp_test = [self._fp_func(smiles) for smiles in smiles_test]
        K_test_train = jnp.asarray([DataStructs.BulkTanimotoSimilarity(fp, self._fp_train) for fp in fp_test])
        if full_covar:
            K_test_test = jnp.asarray([DataStructs.BulkTanimotoSimilarity(fp, fp_test) for fp in fp_test])
        else:
            K_test_test = jnp.ones((len(smiles_test)), dtype=float)

        return kgp.noiseless_predict(
            a=TRANSFORM(params.raw_amplitude),
            s=TRANSFORM(params.raw_noise),
            k_train_train=self._K_train_train,
            k_test_train=K_test_train,
            k_test_test=K_test_test,
            y_train=self._y_train,
            full_covar=full_covar,
        )

    def predict_y(self, params: TanimotoGP_Params, smiles_test: list[str], full_covar: bool = True) -> jnp.ndarray:
        mean, covar = self.predict_f(params, smiles_test, full_covar)
        if full_covar:
            covar = covar + jnp.eye(len(smiles_test)) * TRANSFORM(params.raw_noise)
        else:
            covar += TRANSFORM(params.raw_noise)
        return mean, covar

    def _get_L(self, params: TanimotoGP_Params) -> jnp.ndarray:
        """Get cached or compute new Cholesky factorization"""
        if self._L_cached is None:
            a = TRANSFORM(params.raw_amplitude)
            s = TRANSFORM(params.raw_noise)
            L = kgp._k_cholesky(
                self._K_train_train,
                s / a
            )
            self._L_cached = L
        
        return self._L_cached

    def add_observation(self, params: TanimotoGP_Params, new_smiles: str, new_y: float) -> None:
        """
        Add a single observation and efficiently update cached Cholesky factorization.
        Avoids recomputing K_train_train and Cholesky factorization repeatedly during BO.
        """
        new_fp = self._fp_func(new_smiles)

        # Compute Tanimoto similarity between new observation and each existing training point
        k_new = jnp.asarray(DataStructs.BulkTanimotoSimilarity(new_fp, self._fp_train))

        # Diagonal term is Tanimoto simiarity of new observation w.r.t. itself
        k_new_new = jnp.array(1.0)

        # Update training data
        self._smiles_train.append(new_smiles)
        self._y_train = jnp.append(self._y_train, new_y)
        self._fp_train.append(new_fp)

        # Update kernel matrix
        n = len(k_new)
        top_block = jnp.concatenate([self._K_train_train, k_new.reshape(n, 1)], axis=1)
        bottom_row = jnp.concatenate([k_new, k_new_new.reshape(1,)])
        self._K_train_train = jnp.concatenate([top_block, bottom_row.reshape(1, n+1)], axis=0)

        # If we have cached L, update it efficiently
        if self._L_cached is not None:
            a = TRANSFORM(params.raw_amplitude)
            s = TRANSFORM(params.raw_noise)
            self._L_cached = kgp.update_cholesky(
                self._L_cached,
                k_new,
                k_new_new + (s/a)
            )