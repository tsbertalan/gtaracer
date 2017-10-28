import os
import time
from contextlib import contextmanager

home = os.path.expanduser("~")

def mkdir_p(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
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
