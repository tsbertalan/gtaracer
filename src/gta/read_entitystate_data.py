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
from warnings import warn
from os.path import join, expanduser, dirname
HOME = expanduser("~")
HERE = dirname(__file__)
DATA_DIR = join(HOME, 'data', 'gta', 'velocity_prediction')

from joblib import Memory
memory = Memory(location=DATA_DIR, verbose=1)


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
        # A start marker for the beginning of the struct. Always the bytes b'GTAGTA'.
        ("m1", c_char),
        ("m2", c_char),
        ("m3", c_char),
        ("m4", c_char),
        ("m5", c_char),
        ("m6", c_char),

       
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
        ("roll",  c_float),
        ("pitch", c_float),
        ("yaw",   c_float),

        # Velocity
        ("velx", c_float),
        ("vely", c_float),
        ("velz", c_float),

        # Rotational velocity
        ("rvelx", c_float),
        ("rvely", c_float),
        ("rvelz", c_float),

        # Corresponds to Python's time.monotonic
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

constant_fields = 'is_player', 'is_vehicle', 'm1', 'm2', 'm3'

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


def is_valid(entity_state, max_abs=1e7, max_abs_xy=1e4):
    """A crude additional check of whether a packet is reasonable.
    
    TODO: Figure out what's causing unreasonable packets, that still have a valid checksum.
    """
    # if isinstance(entity_state, EntityState):
    #     for field_name, unused_field_type in entity_state._fields_:
    #         if abs(getattr(entity_state, field_name)) > max_abs:
    #             return False
    # else:
    assert isinstance(entity_state, EntityStateTuple)
    if entity_state.wall_time < 1:
        return False
    for val in entity_state:
        if isinstance(val, numbers.Number) and abs(val) > max_abs:
            return False
    if abs(entity_state.posx) > max_abs_xy or abs(entity_state.posy) > max_abs_xy:
        return False
    return True


@memory.cache
def read_data(fname, scan=False):
    """Scan through the data and make structs."""

    # We'll accumulate structs here.
    entities = []

    # Every struct starts with a 3-byte marker.
    marker = b'GTAGTA'

    # As explained above, we'll read chunks of the larger size,
    # but then feed only the smaller first part into struct.unpack().
    item_size = struct.calcsize(structure_code)
    window_size = sizeof(EntityState)
    assert window_size >= item_size
    
    num_skips = 0
    bytes_skipped = 0
    num_checkfails = 0
    num_validfails = 0
    num_packets_attempted = 0
    warnable_skipsize = max(abs(window_size - item_size), 3)
    failures = []

    with open(fname, "rb") as f:
        bytes = f.read()

    pbar = tqdm(total=len(bytes)/1024., unit='kbytes', desc='Reading loaded file')
    if scan:
        i = 0
        while True:

            # Step the progressbar in units of kilobytes.
            pbar_jump = int(i - pbar.n * 1024.) // 1024
            if pbar_jump > 0:
                pbar.update(pbar_jump)
            
            # Take a bigger bite.
            full_chunk = bytes[i:i+window_size]

            # Check for a marker.
            if not full_chunk.startswith(marker):
                # Skip ahead until the next start marker.
                num_skips += 1
                skip_size = bytes[i:].find(marker)
                if skip_size > warnable_skipsize:
                    warn('Larger byte skip encountered (%d bytes).' % (skip_size,))
                bytes_skipped += skip_size
                if skip_size != -1:
                    i += skip_size
                else:
                    # No more markers; we're done.
                    break
            
            else:
                # Unpack from the smaller first part.
                item_chunk = full_chunk[:item_size]
                if len(item_chunk) < item_size:
                    break
                datum_struct = struct.unpack(structure_code, item_chunk)

                # Construct a more convenient form.
                entity = EntityStateTuple(*datum_struct)
                num_packets_attempted += 1

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
                checksum_target = full_chunk[-1]

                # If the packets passes our test, we'll add it to the list.
                if checksum == checksum_target:
                    if is_valid(entity):
                        entities.append(entity)
                        failures.append(0)
                    else:
                        warn('Entity packet failed validation.')
                        num_validfails += 1
                        failures.append(1)
                else:
                    warn('Entity packet failed checksum.')
                    num_checkfails += 1
                    failures.append(2)

                # Continue to the next packet.
                i += window_size


    else:
        bytes_to_read = bytes
        while True:

            # Skip ahead until the next start marker.
            inext = bytes_to_read.find(marker)
            if inext == -1:
                break
            bytes_to_read = bytes_to_read[inext:]
            if len(bytes_to_read) < item_size:
                break
            
            # Step the progressbar in units of kilobytes.
            i = len(bytes) - len(bytes_to_read)
            pbar_jump = int(i - pbar.n * 1024.) // 1024
            if pbar_jump > 0:
                pbar.update(pbar_jump)

            # Compute and check the checksum.
            checksum_chunk = bytes_to_read[:window_size-1]
            checksum_target = bytes_to_read[window_size]
            checksum = xorchecksum(checksum_chunk)
            num_packets_attempted += 1

            # If the packets passes checksum and our test, we'll add it to the list.
            item_chunk = bytes_to_read[:item_size]
            datum_struct = struct.unpack(structure_code, item_chunk)
            entity = EntityStateTuple(*datum_struct)
            if checksum == checksum_target:
                # if is_valid(entity):
                entities.append(entity)
                failures.append(0)
                # else:
                #     num_validfails += 1
                #     failures.append(1)
            else:
                num_checkfails += 1
                failures.append(2)

            # Whatever happened, step forward one byte so this marker will be skipped.
            bytes_to_read = bytes_to_read[1:]

    if num_skips > 0 or num_checkfails > 0 or num_validfails > 0:
        msg = '''Some problems encountered while reading file:
        - Skipped forward %d times (%d cumulative bytes from %d bytes total, or %.2f%%).
        - %d packets had checksum failures (%.2f%%).
        - %d packets failed validation despite valid checksum (%.2f%%).''' % (
            num_skips, bytes_skipped, len(bytes), 100. * bytes_skipped / len(bytes),
            num_checkfails, 100. * num_checkfails / num_packets_attempted,
            num_validfails,  100. * num_validfails / num_packets_attempted,
        )
        warn(msg)

        # Show the pattern of failures.
        fig, ax = plt.subplots(figsize=(20, 3))

        # Jitter the locations to make them show up more clearly.
        failures = np.array(failures)
        jittered_failures = failures + np.random.normal(loc=0, scale=0.25, size=len(failures))
        attempt_num = np.arange(len(failures)).astype(float)
        jittered_attempt_num = attempt_num# + np.random.normal(loc=0, scale=1.0, size=len(attempt_num))

        # Plot all data.
        ax.scatter(jittered_attempt_num, jittered_failures, marker='|', alpha=.25, color='black')

        # Replot the invalid ones, since they're too few to be seen with light alpha.
        where_baddat = np.argwhere(failures == 1)
        ax.scatter(jittered_attempt_num[where_baddat], jittered_failures[where_baddat], color='black', marker='|', alpha=1.0)

        # Make the plot mobeta, and save.
        ax.set_xlim(0, len(failures))
        ax.grid(True)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['Success', 'Invalid Data', 'Invalid Checksum'])
        ax.set_xlabel('Packet Attempt #')
        ax.set_ylabel('Error Type')

        fig.tight_layout()
        fig.savefig('%s-data_read_failures.png' % (fname,))
        plt.close(fig)

    return entities


class Track:
    def __init__(self, unaffinity_threshold=4.0, merge_threshold=0.1, **interpolation_options):
        interpolation_options.setdefault('kind', 'cubic')
        interpolation_options.setdefault('fill_value', 'extrapolate')
        self.unaffinity_threshold = unaffinity_threshold
        self.merge_threshold = merge_threshold
        self.interpolation_options = interpolation_options
        self._reset_data()

    def _reset_data(self):
        self.interpolators = {}
        self._entities = []

    @property
    def _nonduplicate_time_indices(self):
        t = self.times
        # make sure it's sorted
        t_s = np.array(sorted(t))
        assert np.all(t == t_s)
        indices = []
        last_time = None
        for i, ti in enumerate(t):
            if last_time is None or ti != last_time:
                indices.append(i)
            last_time = ti
        return indices

    @property
    def duration(self):
        return self.tmax - self.tmin

    def __len__(self):
        return self.count

    @property
    def count(self):
        return len(self._entities)

    def get_interpolated_state(self, wall_time):
        return EntityStateTuple(*[self.get_interpolator(k)(wall_time) for (k, unused_field_type) in EntityState._fields_])

    def get(self, field, time):
        return self.get_interpolator(field)(time)

    @property
    def has_constant_position(self):
        return np.unique(self._get_data('posx')).size == 1 and np.unique(self._get_data('posy')).size == 1

    def associate(self, entity):
        """Associate an entity with this track."""
        if self.unaffinity(entity) > self.unaffinity_threshold:
            return False
        else:
            self._entities.append(entity)
            self._entities.sort(key=lambda e: e.wall_time)
            self._clear_caches()
            return True

    def unaffinity(self, entity, which=-1):
        """Return the distance between this entity and a chosen entity in this track."""
        if len(self._entities) == 0:
            return 0.0
        if isinstance(which, str) and which == 'min':
            return min(entity_unaffinity(e, entity) for e in self._entities)
        else:
            assert isinstance(which, int)
            return entity_unaffinity(self._entities[which], entity)

    def _get_data(self, field):
        return np.array([getattr(entity_state, field) for entity_state in self._entities])

    @property
    def times(self):
        return self._get_data('wall_time')

    @property
    def tmin(self):
        return min(self.times)

    @property
    def tmax(self):
        return max(self.times)

    def get_interpolator(self, field):
        """Return an interpolator for the given field."""
        check_datum = self._get_data(field)[0]
        if isinstance(check_datum, numbers.Number):
            from scipy.interpolate  import interp1d
            Interpolator = interp1d
        else:
            Interpolator = NearestFetcher

        if field not in self.interpolators:
            if field in constant_fields:
                assert len(list(set(self._get_data(field)))) == 1
            self.interpolators[field] = ScalarInterpolator(
                Interpolator, self.times, self._get_data(field), **self.interpolation_options
            )
        return self.interpolators[field]

    def _clear_caches(self):
        self.interpolators = {}

    @property
    def is_player(self):
        return self._entities[0].is_player

    @property
    def is_vehicle(self):
        return self._entities[0].is_vehicle


def entity_unaffinity(entity_state1, entity_state2):
    """Return the distance between two entity states."""
    checked_fields = 'wall_time', 'posx', 'posy', 'posz', 'is_vehicle', 'is_player'
    weights        = np.array([1.0, 1.0,    1.0,    1.0,    10.0,    10.0])
    v1 = np.array([getattr(entity_state1, field) for field in checked_fields])
    v2 = np.array([getattr(entity_state2, field) for field in checked_fields])
    e = v1 - v2
    return np.linalg.norm(e * weights)
    

class NearestFetcher:

    def __init__(self, x, y, **ignored_kwargs):
        self.x = np.array(x)
        order = np.argsort(self.x)
        self.x = self.x[order]
        self.y = [y[i] for i in order]

    def __call__(self, new_x):
        closest = np.argmin(np.abs(new_x - self.x))
        return self.y[closest]


class ScalarInterpolator:

    def __init__(self, BaseInterpolator, x, y, **kw):
        if len(x) < 2:
            BaseInterpolator = NearestFetcher
        else:
            min_points = {1: 'nearest', 2: 'linear', 3: 'quadratic', 4: 'cubic'}
            kw.setdefault('kind', 'cubic')
            n = len(x)
            assert n == len(y)
            kind = min_points.get(n, kw['kind'])
            if kind != kw['kind']:
                from warnings import warn
                warn("ScalarInterpolator: changing interpolation kind from %s to %s because we only have %d points." % (kw['kind'], kind, n))
            
            kw['kind'] = kind
        
        
        assert len(x) == len(y)
        self.base_interpolator = BaseInterpolator(x, y, **kw)
        
    def __call__(self, new_x):
        new_y = self.base_interpolator(new_x)
        if not isinstance(new_x, np.ndarray) and isinstance(new_y, np.ndarray):
            new_y = np.atleast_1d(new_y)[0]
        return new_y


class TrackManager:

    def __init__(self, entities=None, pbar=True):
        self.trackgroups = {}
        if entities is not None:
            for entity in entities if not pbar else tqdm(entities, unit='entities', desc='Associating entities with tracks'):
                self.associate(entity)

    def associate(self, entity):
        associated = False
        if entity.id not in self.trackgroups:
            self.trackgroups[entity.id] = [Track()]
        for track in self.trackgroups[entity.id]:
            associated = track.associate(entity)
            if associated:
                break
        if not associated:
            new_track = Track()
            associated = new_track.associate(entity)
            self.trackgroups[entity.id].append(new_track)
        assert associated

    @property
    def tracks(self):
        out = []
        for trackgroup in self.trackgroups.values():
            out.extend(trackgroup)
        return out

    def get_active_tracks(self, time):
        """Return a list of tracks that are active at the given time."""
        active_tracks = []
        for track in self.tracks:
            if track.tmin <= time < track.tmax:
                active_tracks.append(track)
        return active_tracks

    @property
    def tmin(self):
        return min([track.tmin for track in self.tracks])

    @property
    def tmax(self):
        return max([track.tmax for track in self.tracks])

    @property
    def player_ped(self):
        return self._get_first_track_by_prop(test=lambda track: track.is_player and not track.is_vehicle)

    @property
    def player_veh(self):
        return self._get_first_track_by_prop(test=lambda track: track.is_player and track.is_vehicle)
    
    def _get_first_track_by_prop(self, test):
        player_ped_tracks = [track for track in self.tracks if test(track)]
        if len(player_ped_tracks) == 0:
            return None
        elif len(player_ped_tracks) > 1:
            from warnings import warn
            warn(
                "Multiple %s tracks found. Using the first one. " % test
            )
        return player_ped_tracks[0]


def read_data_main(plot_3d=False, fname="C:\Program Files (x86)\Steam\SteamApps\common\Grand Theft Auto V\\"):
    if fname.endswith('\\') or fname.endswith('/'):
        from glob import glob
        binfiles = list(sorted(glob(fname + "GTA_recording*bin")))
        fname = binfiles[-1]
        print('Found most recent binfile:', fname)

    data = read_data(fname)
    track_manager = TrackManager(data)
    t_mean = (track_manager.tmin + track_manager.tmax) / 2.
    #active_tracks = track_manager.get_active_tracks(t_mean)
 


    # Make some pretty plots.
    from os.path import join, dirname
    HERE = dirname(__file__)
    fig_dir = join(HERE, '..', '..', 'doc')

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect("equal")
    for track in track_manager.tracks:

        # Plot the spline.
        T = np.linspace(track.tmin, track.tmax, 1024)
        X = track.get('posx', T)
        Y = track.get('posy', T)
        style = dict(alpha=.7)
        if track.is_player:
            style['linewidth'] = 8
        if track.is_vehicle:
            style['linestyle'] = '--'
        else:
            style['linestyle'] = '-'
        line = ax.plot(X, Y, **style)[0]

        # Plot the underlying data.
        X = track._get_data('posx')
        Y = track._get_data('posy')
        ax.scatter(X, Y, color=line._color, s=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    vehicle_tracks = [track for track in track_manager.tracks if track.is_vehicle]
    pla_veh_tracks = [track for track in vehicle_tracks if track.is_player]
    ped_tracks = [track for track in track_manager.tracks if not track.is_vehicle]
    pla_ped_tracks = [track for track in ped_tracks if track.is_player]
    print('Lengths of player ped tracks:', [
        '%d ents, %.1f sec' % (len(track._entities), track.tmax - track.tmin) for track in pla_ped_tracks
    ])
    print('Lengths of player veh tracks:', [
        '%d ents, %.1f sec' % (len(track._entities), track.tmax - track.tmin) for track in pla_veh_tracks
    ])
    fig.suptitle(
        '{nveh} vehicle tracks, incl. {nvehpla} player\n{nped} pedestrian tracks, incl. {npedpla} player'.format(
            nveh=len(vehicle_tracks), nvehpla=len(pla_veh_tracks), 
            nped=len(ped_tracks), npedpla=len(pla_ped_tracks)
        )
    )
    fig.tight_layout()
    fig.subplots_adjust(top=.9)
    center_x = np.median([np.mean(track._get_data('posx')) for track in vehicle_tracks])
    center_y = np.median([np.mean(track._get_data('posy')) for track in vehicle_tracks])
    radius = 1000
    existing_lowx, existing_highx = ax.get_xlim()
    existing_lowy, existing_highy = ax.get_ylim()
    new_lowx = max(center_x - radius, existing_lowx)
    new_highx = min(center_x + radius, existing_highx)
    new_lowy = max(center_y - radius, existing_lowy)
    new_highy = min(center_y + radius, existing_highy)
    ax.set_xlim(new_lowx, new_highx)
    ax.set_ylim(new_lowy, new_highy)
    for ext in ['png', 'pdf']:
        fig.savefig(join(fig_dir, 'associated_tracks.%s' % ext))


    player_veh = track_manager.player_veh
    if player_veh is not None:
        fig, ax = plt.subplots()
        T = np.linspace(player_veh.tmin, player_veh.tmax, 1200)
        ax.plot(T, player_veh.get('posx', T), label='x')
        ax.plot(T, player_veh.get('posy', T), label='y')
        ax.plot(T, player_veh.get('posz', T), label='z')
        velx = player_veh.get('velx', T)
        vely = player_veh.get('vely', T)
        velz = player_veh.get('velz', T)
        ax.plot(T, velx, label='velx')
        ax.plot(T, vely, label='vely')
        ax.plot(T, velz, label='velz')
        spd = np.sqrt(velx**2 + vely**2 + velz**2)
        ax.plot(T, spd, label='overall speed (vel mag)')
        ax.set_xlabel('Time (s)')
        ax.legend()
        fig.tight_layout()
        ax.set_xlim(new_lowx, new_highx)
        ax.set_ylim(new_lowy, new_highy)
        for ext in ['png', 'pdf']:
            fig.savefig(join(fig_dir, 'player_vehicle.%s' % ext))

        x_ego = [d.posx for d in player_veh._entities]
        y_ego = [d.posy for d in player_veh._entities]
        z_ego = [d.posz for d in player_veh._entities]




    

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
        if plot_3d and player_veh is not None:
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
    
    ax.set_xlim(new_lowx, new_highx)
    ax.set_ylim(new_lowy, new_highy)
    fig.tight_layout()
    fig.savefig(join(fig_dir, 'entity_tracks.png'))

    plt.show()



if __name__ == "__main__":
    read_data_main()
