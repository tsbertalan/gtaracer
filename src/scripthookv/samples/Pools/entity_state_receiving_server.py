import socketserver, time, numbers

from ctypes import c_uint, c_double, c_float, c_bool, Structure, c_char, sizeof


class EntityState(Structure):


    _fields_ = [
        ("m1", c_char),
        ("m2", c_char),
        ("m3", c_char),
       
        ("id", c_uint),           # 4 bytes

        ("posx", c_float),        # 4 bytes
        ("posy", c_float),        # 4 bytes 
        ("posz", c_float),        # 4 bytes

        ("roll", c_float),        # 4 bytes
        ("pitch", c_float),       # 4 bytes
        ("yaw", c_float),         # 4 bytes

        ("velx", c_float),        # 4 bytes
        ("vely", c_float),        # 4 bytes
        ("velz", c_float),        # 4 bytes

        ("rvelx", c_float),       # 4 bytes
        ("rvely", c_float),       # 4 bytes
        ("rvelz", c_float),       # 4 bytes

        ("wall_time", c_double),  # 8 bytes

        ("screenx", c_float),     # 4 bytes
        ("screeny", c_float),     # 4 bytes 

        ("occluded", c_bool),     # 1 byte
        ("is_vehicle", c_bool),   # 1 byte

    ]

    structure_code = 'c'*3 + 'I' + 'fff'*4 + 'dff??'

    # or, 
    # ccc               marker
    # I fff fff fff fff d ff ??   data
    # c                           checksum


from collections import namedtuple
EntityStateTuple = namedtuple("EntityStateTuple", [field_name for field_name, unused_field_type in EntityState._fields_])


def xorchecksum(data):
    checksum = 0
    for el in data:
        checksum ^= el
    return checksum

def is_valid(entity_state, max_abs=1e10):
    if isinstance(entity_state, EntityState):
        for field_name, unused_field_type in entity_state._fields_:
            if abs(getattr(entity_state, field_name)) > max_abs:
                return False
    else:
        assert isinstance(entity_state, EntityStateTuple)
        for val in entity_state:
            if isinstance(val, numbers.Number) and abs(val) > max_abs:
                return False
    return True


class EntityStateReceivingHandler(socketserver.BaseRequestHandler):

    def handle(self):
        bufsize = sizeof(EntityState)
        print('bufsize:', bufsize)
        char_data = self.request.recv(bufsize)

        # Unpack the data into the data structure
        self.data = EntityState.from_buffer_copy(char_data)

        print('[{time:01f}] received from {addr}: "{data}"'.format(
            time=time.time()-self.server.t0, 
            addr=self.client_address[0],
            data=self.data
        ))

        for field_name, field_type in self.data._fields_:
            print("{ty} {name} = {val}".format(
                ty=field_type,
                name=field_name,
                val=getattr(self.data, field_name)
            ))

        # Flush stdout.
        from sys import stdout
        stdout.flush()


class EntityStateReceivingServer(socketserver.TCPServer):

    def __init__(self, *a, **k):
        socketserver.TCPServer.__init__(self, *a, **k)
        self.t0 = time.time()


def UDP_main():
    with EntityStateReceivingServer(("localhost", 27015), EntityStateReceivingHandler) as server:
        server.serve_forever()


def read_data_main(fname="C:\Program Files (x86)\Steam\SteamApps\common\Grand Theft Auto V\GTA_recording.bin"):

    from tqdm.auto import tqdm
    import struct
    data = []
    marker = b'GTA'
    with open(fname, "rb") as f:
        bytes = f.read()
        item_size = struct.calcsize(EntityState.structure_code)
        skip_size = sizeof(EntityState)
        i = 0
        pbar = tqdm(total=len(bytes)/1024., unit='kbytes')
        while True:
            pbar_jump = int(i - pbar.n*1024.) // 1024
            if pbar_jump > 0:
                pbar.update(pbar_jump)
            
            full_chunk = bytes[i:i+skip_size]

            if not full_chunk.startswith(marker):
                # Skip ahead until the next start marker.
                offset = bytes[i:].find(marker)
                if offset == -1:
                    break
                else:
                    i += offset
            
            else:

                item_chunk = full_chunk[:item_size]
                datum = struct.unpack(EntityState.structure_code, item_chunk)
                datum = EntityStateTuple(*datum)
                checksum = xorchecksum(full_chunk[:-1])
                target_checksum = full_chunk[-1]

                #alt_datum = EntityState.from_buffer_copy(full_chunk)

                if checksum == target_checksum and is_valid(datum):
                    data.append(datum)

                i += skip_size
        

        # total_bytes = len(bytes)
        # pbar = tqdm(total=total_bytes, unit='bytes')

        # while len(bytes) > len(marker):
        #     pbar_position = total_bytes - pbar.n
        #     if pbar_position != len(bytes):
        #         pbar.update(pbar_position - len(bytes))

        #     imarker = bytes.find(marker)

        #     if imarker == -1:
        #         break

        #     i1 = imarker + len(marker)
        #     i2 = i1 + sizeof(EntityState) + 1
        #     chunk = bytes[i1:i2]

        #     target_checksum = chunk[-1]
        #     chunk = chunk[:-1]
        #     checksum = xorchecksum(chunk)

        #     if len(chunk) == sizeof(EntityState):
        #         datum = EntityState.from_buffer_copy(chunk)
        #         if is_valid(datum) and checksum == target_checksum:
        #             data.append(datum)
        #     else:
        #         break

        #     bytes = bytes[i2:]
        #     bytes = bytes[bytes.find(marker):]
        #     assert len(bytes) < len(marker) or bytes.startswith(marker)

    ego_peds = [d for d in data if not d.is_vehicle and d.id == 0]

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    x = [d.posx for d in ego_peds]
    y = [d.posy for d in ego_peds]
    get_spd = lambda d: d.velx**2 + d.vely**2 + d.velz**2
    spd = [get_spd(d) for d in ego_peds]
    t = [d.wall_time for d in ego_peds]
    ax.plot(x, y, 'k-')
    fig.colorbar(
        ax.scatter(x, y, c=t, alpha=.5),
        ax=ax,
        label="wall time"
    )
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_aspect('equal')
    duration = max(t) - min(t)
    ax.set_title('Ego Ped over %s seconds' % duration)

    fig, ax = plt.subplots()
    ax.plot(t, x, label='$x(t)$')
    ax.plot(t, y, label='$y(t)$')
    ax.plot(t, spd, label='$||v(t)||$')
    ax.set_xlabel('wall time')
    ax.set_ylabel('position')
    ax.legend()
    ax.set_title('Ego Ped over %s seconds' % duration)


    fig, ax = plt.subplots()
    for ent_data, ent_type in zip(
        [[d for d in data if d.is_vehicle], 
         [d for d in data if not d.is_vehicle]],
        ["vehicle", "non-vehicle"]
        ):
        x = [d.posx for d in ent_data]
        y = [d.posy for d in ent_data]
        t = [d.wall_time for d in ent_data]
        fig.colorbar(
            ax.scatter(x, y, c=t, marker='o' if ent_type == "vehicle" else 'x'),
            ax=ax,
            label='[s] (%s)' % ent_type
        )
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_aspect('equal')
    duration = max(t) - min(t)
    ax.set_title('Entities over %s seconds' % duration)


    

    plt.show()


if __name__ == "__main__":
    read_data_main()
