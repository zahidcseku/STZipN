import numpy as np

from Codebase.data import dataprocessing
import pytest
import logging
from sklearn.metrics.pairwise import haversine_distances
from math import radians


logger = logging.getLogger(__name__)


def test_initializations(PoiD1GCNNoSaveNoNormalize_fixture):
    """test initializations are corrct"""
    logger.info(
        "Testing initial conditions of data object with Poisson = true, "
        "Fdim=1, GAT=false, save=false, normalize=false"
    )
    assert PoiD1GCNNoSaveNoNormalize_fixture.poissonmodel == True
    assert PoiD1GCNNoSaveNoNormalize_fixture.fdim == 1
    assert PoiD1GCNNoSaveNoNormalize_fixture.gat == False
    assert PoiD1GCNNoSaveNoNormalize_fixture.savefeatures == False
    assert PoiD1GCNNoSaveNoNormalize_fixture.normalize == False


@pytest.mark.parametrize("idx", [0, 100, 927])
def test_shape_dataccess_byid(PoiD1GCNNoSaveNoNormalize_fixture, idx):
    logger.info("Testing the shapes of an instance of X and Y are same")
    assert (
        PoiD1GCNNoSaveNoNormalize_fixture[idx]["x"].shape[0]
        == PoiD1GCNNoSaveNoNormalize_fixture[idx]["y"].shape[0]
    )


@pytest.mark.parametrize("idx", [20, 100, 200, 500, 750, 927])
def test_yisnexttimestep_byid(PoiD1GCNNoSaveNoNormalize_fixture, idx):
    logger.info("Testing that for any ts Y is equal to X(ts+1) when accessed by id")
    logger.debug(
        f"Shape of X: {PoiD1GCNNoSaveNoNormalize_fixture[idx]['x'].squeeze().shape}"
    )
    logger.debug(
        f"Shape of Y: {PoiD1GCNNoSaveNoNormalize_fixture[idx]['y'].squeeze().shape}"
    )
    logger.debug(
        f"Values in X: {PoiD1GCNNoSaveNoNormalize_fixture[idx]['x'].squeeze()[:20]}"
    )
    logger.debug(
        f"values in Y: {PoiD1GCNNoSaveNoNormalize_fixture[idx]['y'].squeeze()[:20]}"
    )
    logger.debug(type(PoiD1GCNNoSaveNoNormalize_fixture[idx]["x"].squeeze()))
    assert np.array_equal(
        PoiD1GCNNoSaveNoNormalize_fixture[idx]["x"].squeeze(),
        PoiD1GCNNoSaveNoNormalize_fixture[idx - 1]["y"].squeeze(),
    )


@pytest.mark.parametrize("idx", [20, 100, 200, 500, 750, 927])
def test_yisnexttimestep_byXY(PoiD1GCNNoSaveNoNormalize_fixture, idx):
    logger.info(
        "Testing that for any ts Y is equal to X(ts+1) when accessed by X and Y"
    )
    assert np.array_equal(
        PoiD1GCNNoSaveNoNormalize_fixture.X[idx + 1, :, 0],
        PoiD1GCNNoSaveNoNormalize_fixture.Y[idx, :],
    )


@pytest.mark.parametrize("idx", [20, 100, 200, 500, 750, 927])
def test_dimensionsbyID(PoiD1GCNNoSaveNoNormalize_fixture, idx):
    logger.info(
        "Testing that for any ts the dimension of X and Y are correct when accessed by id"
    )
    assert PoiD1GCNNoSaveNoNormalize_fixture[idx]["x"].shape == (
        PoiD1GCNNoSaveNoNormalize_fixture.get_nsites(),
        PoiD1GCNNoSaveNoNormalize_fixture.fdim,
    )
    assert PoiD1GCNNoSaveNoNormalize_fixture[idx]["y"].shape == (
        PoiD1GCNNoSaveNoNormalize_fixture.get_nsites(),
    )


# TODO: add more comprehensive tests gradually.


def test_haversineFdim1(PoiD1GCNNoSaveNoNormalize_fixture):
    locs = PoiD1GCNNoSaveNoNormalize_fixture.locations
    sites = PoiD1GCNNoSaveNoNormalize_fixture.sites
    X = []
    for _idx in sites:
        X.append([radians(_) for _ in locs[_idx]])
    X = np.array(X)
    # X = X * np.pi / 180
    # X = [radians(_) for _ in X]
    dist_lib = haversine_distances(X)
    dist_lib = dist_lib * 6371e3
    logger.info(f"{dist_lib.max()}, {PoiD1GCNNoSaveNoNormalize_fixture.D.max()}")
    logger.info(type(dist_lib))
    logger.info(type(PoiD1GCNNoSaveNoNormalize_fixture.D))
    logger.info(f"{dist_lib.min()}, {PoiD1GCNNoSaveNoNormalize_fixture.D.min()}")
    logger.info(f"{dist_lib.mean()}, {PoiD1GCNNoSaveNoNormalize_fixture.D.mean()}")
    logger.info(f"{dist_lib.sum()}, {PoiD1GCNNoSaveNoNormalize_fixture.D.sum()}")
    logger.info(f"{dist_lib[dist_lib != PoiD1GCNNoSaveNoNormalize_fixture.D]}")
    logger.info(
        f"{PoiD1GCNNoSaveNoNormalize_fixture.D[dist_lib != PoiD1GCNNoSaveNoNormalize_fixture.D]}"
    )
    logger.info(
        f"{PoiD1GCNNoSaveNoNormalize_fixture.D[dist_lib != PoiD1GCNNoSaveNoNormalize_fixture.D].shape}"
    )

    # TODO: This test looks funny everything looks the same yet the test fails
    # assert np.array_equal(dist_lib, PoiD1GCNNoSaveNoNormalize_fixture.D)
    assert (dist_lib == PoiD1GCNNoSaveNoNormalize_fixture.D).all()


def test_haversineFdim2():
    # TODO: test haversive distance using fdim=2
    pass
