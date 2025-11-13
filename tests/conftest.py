""" This file include fixtures to test several modules of the project.
"""
import pytest
from Codebase.data.dataprocessing import GBRDataset

"""Fixtures for testing data package
Initializations params:
self.poissonmodel = kwargs["poissonmodel"]
self.fdim = kwargs["fdim"]
self.gat = kwargs["gat"]
self.normalize = kwargs["normalize"]
self.savefeatures = kwargs["savefeatures"]
"""
@pytest.fixture(scope="class")
def PoiD1GCNNoSaveNoNormalize_fixture():
    return GBRDataset(fileloc="data/rawdata/Cull Data 211011.xlsx",
                          poissonmodel=True,
                          fdim=1,
                          gat=False,
                          normalize=False,
                          savefeatures=False
                          )

@pytest.fixture(scope="class")
def PoissonFdim2GatFalseSaveFalse_fixture():
    data_obj = GBRDataset(fileloc="rawdata/Cull Data 211011.xlsx",
                          poissonmodel=True,
                          fdim=2,
                          gat=False,
                          savefeatures=False
                          )

@pytest.fixture(scope="class")
def BinomialFdim1GatFalseSaveFalse_fixture():
    data_obj = GBRDataset(fileloc="rawdata/Cull Data 211011.xlsx",
                          poissonmodel=True,
                          fdim=2,
                          gat=False,
                          savefeatures=False
                          )

@pytest.fixture(scope="class")
def cullfile_limited_data():
    pass
"""End data fixtures"""