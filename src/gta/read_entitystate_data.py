"""
Read a stream of entity state structs from a file.

Later, we might want to read from a network socket instead.
"""
import numbers, numpy as np
from collections import namedtuple
from ctypes import c_uint, c_int, c_double, c_float, c_bool, Structure, c_char, sizeof
from tqdm.auto import tqdm
import struct
import matplotlib.pyplot as plt
# Allow for 3D matplotlib plots.
from mpl_toolkits.mplot3d import Axes3D


class EntityState(Structure):
    """
    This is how we will receive data from the C++ side.

    Even though we write sizeof(c_struct) bytes from the C++ side, 
    it ends up writing a few extra for padding. Doing sizeof(EntityState) from
    the Python side correctly predicts this, whereas just doing 
    struct.calcsize(EntityState.structure_code) doesn't.
    So, we need to read chunks of the former, larger size, 
    but then feed only the smaller first part of these chunks
    into struct.unpack(), which is *much* faster than EntityState.from_buffer_copy().

    Still, we'll keep EntityState around as a useful list of explicit class names, 
    from which to build our namedtuple.
    """

    _fields_ = [
        # A start marker for the beginning of the struct. Always the bytes b'GTA'.
        ("m1", c_char),
        ("m2", c_char),
        ("m3", c_char),
       
        # The entity's non-unique ID in this timestep--just an index 
        # in the list of either vehicles or pedestrians.
        # That is, the combination of ID and the later is_vehicle is unique per time step.
        ("id", c_int),

        # Absolute position in the world--not yet sure about the units.
        # TODO: check units.
        ("posx", c_float),
        ("posy", c_float),
        ("posz", c_float),

        # Euler angles in the world frame (right)?
        # TODO: check frames.
        ("roll", c_float),
        ("pitch", c_float),
        ("yaw", c_float),

        # Velocity
        ("velx", c_float),
        ("vely", c_float),
        ("velz", c_float),

        # Rotational velocity
        ("rvelx", c_float),
        ("rvely", c_float),
        ("rvelz", c_float),

        # What's supposed to be wall-clock time, given by something like
        # (double)time.QuadPart / freq.QuadPart
        # However, I'm not sure if this really corresponds to Python's time.time()
        # TODO: Get a reasonably corresponding wall-clock time from C++ to match Python's.
        ("wall_time", c_double),

        # If the entity is on-screen, its coordinates in screen space (pixels?).
        ("screenx", c_float),
        ("screeny", c_float),

        # Is the entity on-screen? This tends to err on the side of reporting False,
        # even if something like a tree is almost completely blocking it.
        # Or, maybe it just doesn't even count trees as occluding the entity.
        ("occluded", c_bool),

        # Whether the entity is a vehicle (vs a pedestrian).
        ("is_vehicle", c_bool),

        # Whether the entity is the player or player's vehicle.
        ("is_player", c_bool),
    ]

# This structure code is what we'll *actually* use to read the struct.
structure_code = ''
for field_name, field_type in EntityState._fields_:
    structure_code += {
        c_uint: 'I',
        c_float: 'f',
        c_double: 'd',
        c_bool: '?',
        c_char: 'c',
        c_int: 'i',
    }[field_type]
    
EntityStateTuple = namedtuple("EntityStateTuple", [field_name for field_name, unused_field_type in EntityState._fields_])


def xorchecksum(data):
    """
    Verify that the checksum of a struct is correct.

    I'm not sure how really necessary this is, since we're writing directly to disk.
    But there might be some race conditions in e.g. how update() is called in the C++ code
    that corrupt data, or we might seek to a start-sequence that is just a coincidence,
    or we might later start sending this same data stream over the wire instead of reading from disk.
    """
    checksum = 0
    for el in data:
        checksum ^= el
    return checksum


def is_valid(entity_state, max_abs=1e10):
    """A crude additional check of whether a packet is reasonable.
    
    TODO: Figure out what's causing unreasonable packets, that still have a valid checksum.
    """
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



def read_data(fname):
    """Scan through the data and make structs."""

    # We'll accumulate structs here.
    data = []

    # Every struct starts with a 3-byte marker.
    marker = b'GTA'

    # As explained above, we'll read chunks of the larger size,
    # but then feed only the smaller first part into struct.unpack().
    item_size = struct.calcsize(structure_code)
    skip_size = sizeof(EntityState)
    assert skip_size >= item_size

    with open(fname, "rb") as f:
        bytes = f.read()

        pbar = tqdm(total=len(bytes)/1024., unit='kbytes', desc='Reading loaded file')
        i = 0
        while True:

            # Step the progressbar in units of kilobytes.
            pbar_jump = int(i - pbar.n * 1024.) // 1024
            if pbar_jump > 0:
                pbar.update(pbar_jump)
            
            # Take a bigger bite.
            full_chunk = bytes[i:i+skip_size]

            # Check for a marker.
            if not full_chunk.startswith(marker):
                # Skip ahead until the next start marker.
                offset = bytes[i:].find(marker)
                if offset != -1:
                    i += offset
                else:
                    # No more markers; we're done.
                    break
            
            else:
                # Unpack from the smaller first part.
                item_chunk = full_chunk[:item_size]
                datum = struct.unpack(structure_code, item_chunk)

                # Construct a more convenient form.
                datum = EntityStateTuple(*datum)

                # Check the checksum. Note that the checksum is an
                # extra byte appended *after* even the larger chunk.
                # Because the larger and smaller chunk differ only by zero padding,
                # they should have the same (xor) checksum.
                # [sssssssssssssssslllllc]
                #  ^^^^^^^^^^^^^^^^
                #    smaller chunk
                #  ^^^^^^^^^^^^^^^^^^^^^
                #       larger chunk
                #                       ^
                #                checksum
                checksum = xorchecksum(full_chunk[:-1])
                target_checksum = full_chunk[-1]

                # If the packets passes our test, we'll add it to the list.
                if checksum == target_checksum and is_valid(datum):
                    data.append(datum)

                # Continue to the next packet.
                i += skip_size
    return data
        

def read_data_main(plot_3d=False, fname="C:\Program Files (x86)\Steam\SteamApps\common\Grand Theft Auto V\GTA_recording.bin"):
    data = read_data(fname)

    # TODO: Save real ego packets from C++ and read them here. id==0 cannot be trusted.
    ego_peds = [d for d in data if not d.is_vehicle and d.is_player]
    x_ego = [d.posx for d in ego_peds]
    y_ego = [d.posy for d in ego_peds]
    z_ego = [d.posz for d in ego_peds]

    # Make some pretty plots.
    # fig, ax = plt.subplots()
    # get_spd = lambda d: d.velx**2 + d.vely**2 + d.velz**2
    # spd = [get_spd(d) for d in ego_peds]
    # t = [d.wall_time for d in ego_peds]
    # ax.plot(x_ego, y_ego, 'k-')
    # fig.colorbar(
    #     ax.scatter(x_ego, y_ego, c=t, alpha=.5),
    #     ax=ax,
    #     label="wall time"
    # )
    # ax.set_xlabel('$x$')
    # ax.set_ylabel('$y$')
    # ax.set_aspect('equal')
    # duration = max(t) - min(t)
    # ax.set_title('Ego Ped over %s seconds' % duration)

    # fig, ax = plt.subplots()
    # ax.plot(t, x_ego, label='$x(t)$')
    # ax.plot(t, y_ego, label='$y(t)$')
    # ax.plot(t, spd, label='$||v(t)||$')
    # ax.set_xlabel('wall time')
    # ax.set_ylabel('position')
    # ax.legend()
    # ax.set_title('Ego Ped over %s seconds' % duration)


    

    if plot_3d:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig, ax = plt.subplots(figsize=(8, 6))

    ent_datas = {
        'vehicle': [d for d in data if d.is_vehicle], 
        'non-vehicle': [d for d in data if not d.is_vehicle],
    }

    unique_ids = {
        k: list(sorted(set([d.id for d in ent_datas[k]])))
        for k in ent_datas.keys()
    }
    entity_ids = {
        k: [unique_ids[k].index(d.id) for d in ent_datas[k]]
        for k in ent_datas.keys()
    }

    for ent_type in ["vehicle", "non-vehicle"]:
        # x = [d.posx for d in ent_data]
        # y = [d.posy for d in ent_data]
        # if plot_3d:
        #     z = [d.posz for d in ent_data]
        ent_data = ent_datas[ent_type]
        t = [d.wall_time for d in ent_data]
        
        veh = ent_type == "vehicle"
        offset_per_id = np.random.normal(scale=.1, size=(len(unique_ids[ent_type]), 3))
        x = [d.posx + offset_per_id[unique_ids[ent_type].index(d.id)][0] for d in ent_data]
        y = [d.posy + offset_per_id[unique_ids[ent_type].index(d.id)][1] for d in ent_data]
        z = [d.posz + offset_per_id[unique_ids[ent_type].index(d.id)][2] for d in ent_data]
        args = (x, y, z) if plot_3d else (x, y)
        fig.colorbar(
            ax.scatter(*args, 

                #c=t,
                #cmap='viridis' if veh else 'plasma',

                c=entity_ids[ent_type],
                cmap='jet',

                marker='o' if veh else 's',
                s=1 if veh else 8,
                alpha=.9 if veh else .4,
            ),
            ax=ax,
            label='[s] (%s)' % ent_type
        )
        if plot_3d:
            center_x = np.mean(x_ego)
            center_y = np.mean(y_ego)
            center_z = np.mean(z_ego)
            radius = 150
            ax.set_xlim(center_x - radius, center_x + radius)
            ax.set_ylim(center_y - radius, center_y + radius)
            ax.set_zlim(center_z - radius, center_z + radius)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    if plot_3d:
        ax.set_zlabel('$z$')
    else:
        ax.set_aspect('equal')
    duration = max(t) - min(t)
    ax.set_title('Entities over %s seconds' % duration)
    
    from os.path import join, dirname
    HERE = dirname(__file__)
    fig_dir = join(HERE, '..', '..', 'doc')
    fig.tight_layout()
    fig.savefig(join(fig_dir, 'entity_tracks.png'))

    plt.show()



if __name__ == "__main__":
    read_data_main()
