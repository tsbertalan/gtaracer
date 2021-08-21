import os
import time
from contextlib import contextmanager

home = os.path.expanduser("~")

def mkdir_p(path):
    try:
        print("Creating directory: {}".format(path))
        os.makedirs(path)
    except FileExistsError:
        pass
    except OSError as exc:  # Python >2.5
        # Check if the path exists (Windows)
        if os.path.exists(path) and os.path.isdir(path):
            pass
        else:
            raise

@contextmanager
def timeit(label=None):
    '''Context manager to print elapsed time.
    
    Use like:
    >>> with timeit('waiting'):
    ...     sleep(1.0)
    1.0 sec elapsed waiting.
    '''
    if label is not None:
        print('%s ... ' % label, end='')
    
    s = time.time()
    yield
    e = time.time()
    out = '%.1f sec elapsed.' % (e - s)
    print(out)
