import pytest
from bayesrul import __version__
from bayesrul.utils.lmdb_utils import make_slice


def test_version():
    assert __version__ == '0.1.0'


def test_make_slice():
    s = [x for x in make_slice(6,2,2)]
    true_s = [
        slice(0,2,None),
        slice(2,4,None),
        slice(4,6,None)
    ]
    
    assert all([s[i] == true_s[i] for i in range(len(s))])