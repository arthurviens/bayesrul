import pytest

from pathlib import Path
from bayesrul import __version__
from bayesrul.utils.lmdb_utils import make_slice
from bayesrul.utils.plotting import PredLogger


class DangerousOperationError(Exception):
    pass


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


def test_PredLogger():
    try:
        preddict = {'preds': [1,2,3], 'labels': [4,5,6]}
        
        base_log_dir = './tests/'

        predLog = PredLogger(base_log_dir)
        predLog.save(preddict)

        retrieved = predLog.load()
        
        for key in retrieved:
            for i in range(len(retrieved[key])):
                assert preddict[key][i] == retrieved[key][i]

        with pytest.raises(KeyError):
            predLog.save({'yolo': [1,2], 'random': [3, 4]})
        
    finally: # Cleaning created file and directory

        # Careful rmdir !
        if 'tests/predictions' not in predLog.path.absolute().as_posix(): 
            raise DangerousOperationError("You are potentially trying to "
                "remove a directory you would not want to remove")
        else:
            Path(predLog.path, 'preds.npy').unlink()
            predLog.path.rmdir()