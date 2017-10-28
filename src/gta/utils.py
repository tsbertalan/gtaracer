import os
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