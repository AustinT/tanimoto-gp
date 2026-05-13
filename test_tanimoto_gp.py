import jax.numpy as jnp
import pytest
from rdkit.Chem import rdFingerprintGenerator

import tanimoto_gp

_mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512)


def fp_func(smiles: str):
    from rdkit import Chem

    return _mfpgen.GetSparseCountFingerprint(Chem.MolFromSmiles(smiles))


SMILES_TRAIN = ["C", "CC", "CCC", "CCCC"]
SMILES_TEST = ["CCCCC", "c1ccccc1"]
Y_TRAIN = jnp.array([1.0, 2.0, 3.0, 4.0])
PARAMS = tanimoto_gp.TanimotoGP_Params(
    raw_amplitude=jnp.array(1.0),
    raw_noise=jnp.array(0.1),
    mean=jnp.array(2.5),
)


@pytest.fixture
def gp():
    return tanimoto_gp.ConstantMeanTanimotoGP(fp_func, SMILES_TRAIN, Y_TRAIN)


def test_instantiation(gp):
    assert gp is not None


def test_mll_is_scalar(gp):
    mll = gp.marginal_log_likelihood(PARAMS)
    assert mll.shape == ()


def test_predict_f_full_covar(gp):
    mean, covar = gp.predict_f(PARAMS, SMILES_TEST, full_covar=True)
    n = len(SMILES_TEST)
    assert mean.shape == (n,)
    assert covar.shape == (n, n)


def test_predict_f_diag(gp):
    mean, var = gp.predict_f(PARAMS, SMILES_TEST, full_covar=False)
    n = len(SMILES_TEST)
    assert mean.shape == (n,)
    assert var.shape == (n,)


def test_predict_y_adds_noise(gp):
    mean_f, var_f = gp.predict_f(PARAMS, SMILES_TEST, full_covar=False)
    mean_y, var_y = gp.predict_y(PARAMS, SMILES_TEST, full_covar=False)
    assert jnp.allclose(mean_f, mean_y)
    assert jnp.all(var_y > var_f)
