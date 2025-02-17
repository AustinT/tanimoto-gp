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


class BaseTanimotoGP:
    """Base class for Tanimoto kernel Gaussian Process implementation."""

    def __init__(self, fp_func: Callable[[str], Any], smiles_train: list[str], y_train) -> None:
        self._fp_func = fp_func
        self.set_training_data(smiles_train, y_train)

    def _setup_kernel(self, smiles_train: list[str]) -> None:
        self._fp_train = [self._fp_func(smiles) for smiles in smiles_train]
        self._K_train_train = jnp.asarray(
            [DataStructs.BulkTanimotoSimilarity(fp, self._fp_train) for fp in self._fp_train]
        )

    def marginal_log_likelihood(self, params: TanimotoGP_Params) -> jnp.ndarray:
        return kgp.mll_train(
            a=TRANSFORM(params.raw_amplitude),
            s=TRANSFORM(params.raw_noise),
            k_train_train=self._K_train_train,
            y_train=self._get_training_targets(),
        )

    def predict_f(self, params: TanimotoGP_Params, smiles_test: list[str], full_covar: bool = True) -> jnp.ndarray:
        fp_test = [self._fp_func(smiles) for smiles in smiles_test]
        K_test_train = jnp.asarray([DataStructs.BulkTanimotoSimilarity(fp, self._fp_train) for fp in fp_test])
        K_test_test = (
            jnp.asarray([DataStructs.BulkTanimotoSimilarity(fp, fp_test) for fp in fp_test])
            if full_covar
            else jnp.ones((len(smiles_test)), dtype=float)
        )

        return kgp.noiseless_predict(
            a=TRANSFORM(params.raw_amplitude),
            s=TRANSFORM(params.raw_noise),
            k_train_train=self._K_train_train,
            k_test_train=K_test_train,
            k_test_test=K_test_test,
            y_train=self._get_training_targets(),
            full_covar=full_covar,
        )

    def _get_training_targets(self) -> jnp.ndarray:
        raise NotImplementedError

    def predict_y(self, params: TanimotoGP_Params, smiles_test: list[str], full_covar: bool = True) -> jnp.ndarray:
        raise NotImplementedError


class ZeroMeanTanimotoGP(BaseTanimotoGP):
    """Tanimoto GP implementation with zero mean function."""

    def set_training_data(self, smiles_train: list[str], y_train: jnp.ndarray) -> None:
        self._smiles_train = smiles_train
        self._y_train = jnp.asarray(y_train)
        self._setup_kernel(smiles_train)

    def _get_training_targets(self) -> jnp.ndarray:
        """Get uncentered training targets"""
        return self._y_train

    def predict_y(self, params: TanimotoGP_Params, smiles_test: list[str], full_covar: bool = True) -> jnp.ndarray:
        mean, covar = self.predict_f(params, smiles_test, full_covar)
        if full_covar:
            covar = covar + jnp.eye(len(smiles_test)) * TRANSFORM(params.raw_noise)
        else:
            covar += TRANSFORM(params.raw_noise)
        return mean, covar


class TanimotoGP(BaseTanimotoGP):
    """Tanimoto GP implementation with mean function set to training data mean."""

    def set_training_data(self, smiles_train, y_train) -> None:
        self._smiles_train = smiles_train
        self._y_train = jnp.asarray(y_train)
        self._y_mean = jnp.mean(self._y_train)  # Compute mean of training data
        self._y_centered = self._y_train - self._y_mean  # Center training data by subtracting mean
        self._setup_kernel(smiles_train)

    def _get_training_targets(self) -> jnp.ndarray:
        """Get uncentered training targets"""
        return self._y_centered

    def predict_y(self, params: TanimotoGP_Params, smiles_test: list[str], full_covar: bool = True) -> jnp.ndarray:
        """Predict observed values for test points with added mean."""

        mean, covar = self.predict_f(params, smiles_test, full_covar)
        if full_covar:
            covar = covar + jnp.eye(len(smiles_test)) * TRANSFORM(params.raw_noise)
        else:
            covar += TRANSFORM(params.raw_noise)
        return mean + self._y_mean, covar
