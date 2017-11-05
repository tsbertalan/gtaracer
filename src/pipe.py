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
import socket, time, pickle, struct

# https://stackoverflow.com/questions/34653875/python-how-to-send-data-over-tcp
# https://wiki.python.org/moin/TcpCommunication


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

    def put(self, msg):
        msg = pickle.dumps(msg)
        # Prefix each message with a 4-byte length (network byte order)
        self.s.sendall(
            struct.pack('>I', len(msg))
            +
            msg
        )


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

    def get(self, sleep=1, connect=True, pbar=False):
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
        if pbar: pbar.update(n - pbar.n)
        return data

    def __del__(self):
        if hasattr(self, 'conn'):
            self.conn.close()
        self.s.close()

