'''The internet is a series of tubes.

Remote:
>>> source = pipe.Source()

Local:
>>> sink = pipe.Sink('remote.vpn.tomsb.net')
>>> import numpy as np
>>> big = np.arange(1000*1000).reshape((1000, 1000))
>>> big[-1, -1]
999999

>>> sink.put(big)

Remote:
>>> big = source.get()
>>> type(big)
numpy.ndarray
>>> big[-1, -1]
999999

'''
# https://stackoverflow.com/questions/34653875/python-how-to-send-data-over-tcp
# https://wiki.python.org/moin/TcpCommunication

import socket, time, pickle, struct
from contextlib import contextmanager

@contextmanager
def timeit(label=None):
    if label is not None: print('%s ... ' % label, end='')
    s = time.time()
    yield
    e = time.time()
    print('(%.1f sec).' % (e - s))

def _compress(data):
    import zlib
    with timeit('Compressing %d bytes' % len(data)):
        return zlib.compress(data)

def _decompress(data):
    import zlib
    with timeit('Decompressing %d bytes' % len(data)):
        return zlib.decompress(data)


class Sink(object):
    def __init__(self, 
        # The same port as used by the server
        host=socket.gethostname(),
        port=44444,
        ):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((host, port))

    def __del__(self):
        self.s.close()

    def put(self, msg, compress=False):
        msg = pickle.dumps(msg)
        if compress:
            msg = _compress(msg)
        # Prefix each message with a 4-byte length (network byte order)
        self.s.sendall(
            struct.pack('>I', len(msg))
            +
            msg
        )

    def putArray(self, arr, recursiveHalving=0, **kwargs):
        kwargs.setdefault('compress', True)
        def halves(x):
            l = len(x)
            s = int(float(l) / 2)
            return x[:s], x[s:]
        assert recursiveHalving >= 0
        try:    
            if recursiveHalving == 0:
                self.put(arr, **kwargs)
            else:
                [self.putArray(x, recursiveHalving-1) for x in halves(arr)]
        except MemoryError:
            self.putArray(arr, recursiveHalving+1)


class Source(object):

    def __init__(self,
        port=44444,     # Arbitrary non-privileged port
        host='',        # Symbolic name meaning all available interfaces
        ):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind((host, port))
        self.s.listen(1)
        
    def _connect(self):
        if not hasattr(self, 'conn'):
            self.conn, self.connAddr = self.s.accept()
            print('Connected by', self.connAddr)

    def get(self, sleep=1, connect=True, pbar=False, decompress=False):
        if connect:
            self._connect()
        while True:
            # Read message length and unpack it into an integer
            raw_msglen = self._recvall(4)
            if not raw_msglen:
                if sleep or sleep == 0:
                    time.sleep(sleep)
                else:
                    return None
            else:
                msglen = struct.unpack('>I', raw_msglen)[0]
                out = self._recvall(msglen, pbar=pbar)
                break
        if out is not None:
            if decompress:
                out = _decompress(out)
            out = pickle.loads(out)
        return out

    def _recvall(self, n, pbar=False):
        '''Helper function to recv n bytes or return None if EOF is hit'''
        data = b''
        if pbar:
            import tqdm
            pbar = tqdm.tqdm(total=n, unit='ibibyte', unit_scale=True)
        while len(data) < n:
            packet = self.conn.recv(n - len(data))
            if pbar: pbar.update(len(packet))
            if not packet:
                return None
            data += packet
        if pbar: 
            pbar.update(n - pbar.n)
            pbar.refresh()
        return data

    def __del__(self):
        if hasattr(self, 'conn'):
            self.conn.close()
        self.s.close()


class ArraySaver(Source):

    def __init__(self, pathFormat='./message_%s', **kwargs):
        Source.__init__(self, **kwargs)
        self.pathFormat = pathFormat

    def record(self, compressed=False, **getKwargs):
        getKwargs.setdefault('decompress', True)
        import numpy as np
        while True:
            try:
                msg = self.get(**getKwargs)
                print('Got msg ...', end=' ')
                if isinstance(msg, np.ndarray):
                    print('array shape is %s.' % (msg.shape,))
                    fpath = self.pathFormat % time.time()
                    with timeit('Saving to %s' % fpath):
                        if compressed:
                            np.savez_compressed(fpath, msg)
                        else:
                            np.save(fpath, msg)
                else:
                    from warnings import warn
                    warn('got %s, not ndarray:\n%s' % (type(msg), msg))
            except KeyboardInterrupt:
                print('Caught KeyboardInterrupt; stopping %s.' % type(self).__name__)
                break

