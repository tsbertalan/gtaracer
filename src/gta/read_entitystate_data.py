"""
Read a stream of entity state structs from a file.

Later, we might want to read from a network socket instead.
"""
import numbers, numpy as np
from collections import namedtuple
from tqdm.auto import tqdm
import struct
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
# Allow for 3D matplotlib plots.
from mpl_toolkits.mplot3d import Axes3D
from warnings import warn
from os.path import join, expanduser, dirname
from packaging import version
HOME = expanduser("~")
HERE = dirname(__file__)
DATA_DIR = join(HOME, 'data', 'gta', 'velocity_prediction')

from joblib import Memory
memory = Memory(location=DATA_DIR, verbose=11)

try:
    from . import protocol_versions
except ImportError:
    import protocol_versions

from ctypes import c_uint, c_int, c_double, c_float, c_bool, Structure, c_char, sizeof


class ProtocolDefinition:
    
    def __init__(self, version_string):
        self.version_string = version_string
        self.version = version.parse(version_string)
        v1 = version.parse('1')
        v2 = version.parse('2')
        
        if self.version == v2:
            self.EntityState = protocol_versions.EntityStateV2
        elif self.version == v1:
            self.EntityState = protocol_versions.EntityStateV1
        elif self.version > v2:
            raise ValueError('Version {} is not supported.'.format(version_string))
        else:
            assert isinstance(self.version, version.LegacyVersion)
            warn("Unknown version string: {}; defaulting to V1.".format(version_string))
            self.EntityState = protocol_versions.EntityStateV1
        self.constant_fields = 'is_player', 'is_vehicle', 'm1', 'm2', 'm3'

        # This structure code is what we'll *actually* use to read the struct.
        self.structure_code = ''
        for field_name, field_type in self.EntityState._fields_:
            self.structure_code += {
                c_uint: 'I',
                c_float: 'f',
                c_double: 'd',
                c_bool: '?',
                c_char: 'c',
                c_int: 'i',
            }[field_type]
        self.EntityStateTuple = namedtuple(
            "EntityStateTuple", 
            [field_name for field_name, unused_field_type in self.EntityState._fields_] + ['protocol_definition'],
            defaults=([None]*len(self.EntityState._fields_)) + [self]
        )


def get_bbox(entity_state_tuple):
    """
    Get the bounding box of an entity.
    """
    before_rotation = np.array([
        [
            -entity_state_tuple.bboxdx/2,
            -entity_state_tuple.bboxdy/2,
        ],
        [
            entity_state_tuple.bboxdx/2,
            -entity_state_tuple.bboxdy/2,
        ],
        [
            entity_state_tuple.bboxdx/2,
            entity_state_tuple.bboxdy/2,
        ],
        [
            -entity_state_tuple.bboxdx/2,
            entity_state_tuple.bboxdy/2,
        ],
    ])

    rotation_angle = np.radians(entity_state_tuple.yaw)
    rotation_matrix = np.array([
        [np.cos(rotation_angle), -np.sin(rotation_angle)],
        [np.sin(rotation_angle), np.cos(rotation_angle)],
    ])
    offset = np.array([entity_state_tuple.posx, entity_state_tuple.posy])
    after_rotation = np.dot(rotation_matrix, before_rotation.T).T + offset
    return after_rotation


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


def is_valid(EntityStateTuple, entity_state, max_abs=1e7, max_abs_xy=1e4):
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
def read_data(fname):
    """Scan through the data and make structs."""

    # We'll accumulate structs here.
    entities = []
    i = 0

    # Every struct starts with a 3-byte marker.
    marker = b'GTAGTA'

    # Select a protocol version.
    with open(fname, "rb") as f:
        bytes = f.read()
    pv_tok1 = b'PROTOCOL_VERSION>>'
    pv_tok2 = b'<<PROTOCOL_VERSION'
    if not (pv_tok1 in bytes[:100] and pv_tok2 in bytes[:100]):
        protocol_version = 'undefined'
    else:
        # This is a protocol version header.
        # Read it so we know which data definition to use, and then skip it.
        i += len(pv_tok1)
        pv_bstr_len = bytes[i:].find(pv_tok2)
        protocol_version = bytes[i:i+pv_bstr_len].decode('ascii')
        print('Protocol version: {}'.format(protocol_version))
        i += pv_bstr_len + len(pv_tok2)
        
    # As explained above, we'll read chunks of the larger size,
    # but then feed only the smaller first part into struct.unpack().
    protocol_definition = ProtocolDefinition(protocol_version)
    item_size = struct.calcsize(protocol_definition.structure_code)
    window_size = sizeof(protocol_definition.EntityState)
    assert window_size >= item_size
    
    num_skips = 0
    bytes_skipped = 0
    num_checkfails = 0
    num_validfails = 0
    num_packets_attempted = 0
    warnable_skipsize = max(abs(window_size - item_size), 3)
    failures = []
        
    pbar = tqdm(total=int(np.math.ceil(len(bytes)/1024.)), unit='kbytes', desc='Reading loaded file')
    while True:

        # Step the progressbar in units of kilobytes.
        pbar_jump = int(i - pbar.n * 1024.) // 1024
        if pbar_jump > 0:
            pbar.update(pbar_jump)
        
        # Take a bigger bite, for checksumming purposes.
        full_chunk = bytes[i:i+window_size+1]

        # Check for a marker.
        if not full_chunk.startswith(marker):
            # Skip ahead until the next start marker.
            num_skips += 1
            skip_size = bytes[i:].find(marker)
            if skip_size > warnable_skipsize:
                warn('Larger byte skip encountered (%d bytes).' % (skip_size,))
            if skip_size > 0:
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
            datum_struct = struct.unpack(protocol_definition.structure_code, item_chunk)

            # Construct a more convenient form.
            entity = protocol_definition.EntityStateTuple(*datum_struct)
            num_packets_attempted += 1

            # Check the checksum. Note that the checksum is an
            # extra byte appended *after* even the larger chunk.
            # Because the larger and smaller chunk differ only by zero padding,
            # they should have the same (xor) checksum.
            # [sssssssssssssssslllllc]
            #  ^^^^^^^^^^^^^^^^
            #    item chunk
            #  ^^^^^^^^^^^^^^^^^^^^^
            #       full chunk
            #                       ^
            #                checksum
            checksum = xorchecksum(full_chunk[:-2])
            checksum_target = full_chunk[-1]

            # If the packets passes our test, we'll add it to the list.
            if checksum == checksum_target:
                if is_valid(protocol_definition.EntityStateTuple, entity):
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
            i += window_size+1


    if bytes_skipped > 0 or num_checkfails > 0 or num_validfails > 0:
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
    def __init__(self, unaffinity_threshold=4.0, merge_threshold=0.5, **interpolation_options):
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

    @property
    def EntityStateTuple(self):
        return self.protocol_definition.EntityStateTuple

    @property
    def protocol_definition(self):
        return self._entities[0].protocol_definition

    def get_interpolated_state(self, wall_time):
        return self.EntityStateTuple(*[self.get_interpolator(k)(wall_time) for k in self.EntityStateTuple._fields])

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

    def __add__(self, track):
        """Merge two tracks together."""
        first, second = self._select_dominant_track(self, track)

        first_entities = first._entities
        second_entities = second._entities
        final_time = first_entities[-1].wall_time

        new_entites = first_entities
        for e in second_entities:
            if e.wall_time > final_time:
                new_entites.append(e)
        first._reset_data()
        first._entities = new_entites
        return first

    def can_merge(self, track, do_plot=False):
        """Return True if the two tracks can be merged."""
        first = self if self.tmin < track.tmin else track
        second = self if self is not first else track
        unaffinity = min(
            first.unaffinity(second.get_interpolated_state(first.tmax), which=-1), 
            second.unaffinity(first.get_interpolated_state(second.tmin), which=0)
        )
        possible = unaffinity < first.unaffinity_threshold
        if possible:
            print(self, 'can be merged with', track)
        if do_plot and (
            (not first.has_constant_position)
            and
            (not second.has_constant_position)
        ) and possible:
            fig, axbx = first.show_track()
            fig.suptitle('Can Merge' if possible else 'Cannot Merge')
            fig.subplots_adjust(top=0.92)
            second.show_track(axbx=axbx, linewidth=8, alpha=.75)
            plt.show()
            plt.close(fig)
        return possible

    @property
    def ids(self):
        return sorted(list(set([e.id for e in self._entities])))

    def show_track(self, axbx=None, do_legend=True, **kw_plot):
        if axbx is None:
            unused_fig, axbx = plt.subplots(ncols=2)

        ax, bx = axbx

        kw_plot.setdefault('label', str(self.ids))
        line = ax.plot(self._get_data('posx'), self._get_data('posy'), **kw_plot)[0]
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        kw_plot['color'] = line.get_color()
        bx.plot(self.times, self._get_data('posx'), **kw_plot)
        bx.set_xlabel('Time')
        bx.set_ylabel('X')

        if do_legend:
            ax.legend()
            bx.legend()
            
        fig = ax.get_figure()
        fig.tight_layout()

        return fig, axbx

    @staticmethod
    def _select_dominant_track(track1, track2):
        """Return the dominant track."""
        if track1.tmin < track2.tmin:
            first = track1
        else:
            if track1.tmin == track2.tmin:
                if track1.duration > track2.duration:
                    first = track1
                else:
                    first = track2
            else:
                first = track2
        return first, track2 if first is track1 else track1

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
        force_nearest = 'id', 'm1', 'm2', 'm3', 'is_occluded', 'is_player', 'is_vehicle'
        if (field not in force_nearest) and isinstance(check_datum, numbers.Number):
            from scipy.interpolate  import interp1d
            Interpolator = interp1d
        else:
            Interpolator = NearestFetcher

        if field not in self.interpolators:
            if field in self.protocol_definition.constant_fields:
                assert len(list(set(self._get_data(field)))) == 1
            x = self.times
            y = self._get_data(field)
            which = self._nonduplicate_time_indices
            x = np.array(x)[which]
            y = np.array(y)[which]
            self.interpolation_options.setdefault('kind', 'cubic')
            self.interpolation_options.setdefault('fill_value', 'extrapolate')
            self.interpolators[field] = ScalarInterpolator(
                Interpolator, x, y, **self.interpolation_options
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
    checked_fields = list(AFFINITY_KEYS) + ['is_vehicle', 'is_player']
    weights = list(AFFINITY_KEY_WEIGHTS) + [10.0, 10.0]
    # checked_fields = 'wall_time', 'posx', 'posy', 'posz', 'is_vehicle', 'is_player'
    # weights        = np.array([1.0, 1.0,    1.0,    1.0,    10.0,    10.0])
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


AFFINITY_KEYS = 'posx', 'posy', 'posz', 'wall_time'
AFFINITY_KEY_WEIGHTS = 1.0, 1.0, 1.0, 100.0


class ScalarInterpolator:

    def __init__(self, BaseInterpolator, x, y, **kw):
        self._kw = kw
        if len(x) < 2:
            BaseInterpolator = NearestFetcher
        else:
            min_points = {1: 'nearest', 2: 'linear', 3: 'quadratic', 4: 'cubic'}
            kw.setdefault('kind', 'cubic')
            n = len(x)
            assert n == len(y)
            kind = min_points.get(n, kw['kind'])
            if kind != kw['kind']:
                warn("ScalarInterpolator: changing interpolation kind from %s to %s because we only have %d points." % (kw['kind'], kind, n))
            
            kw['kind'] = kind
        
        assert len(x) == len(y)
        self.base_interpolator = BaseInterpolator(x, y, **kw)
        
    def __call__(self, new_x):
        new_y = self.base_interpolator(new_x)
        if not isinstance(new_x, np.ndarray) and isinstance(new_y, np.ndarray):
            new_y = np.atleast_1d(new_y)[0]
        return new_y


@memory.cache
def cached_TrackManager_fetch(binfpath):
    return TrackManager(binfpath)


class TrackManager:

    def show_occupancy(self, t, show_tracks=True):
        active_tracks = self.get_active_tracks(t)
        fig, ax = plt.subplots()
        for track in active_tracks:
            entity_state_tuple = track.get_interpolated_state(t)
            points = get_bbox(entity_state_tuple)
            # ax.scatter(*points.T, s=2, c='k')
            c = 'red' if track.is_player else ('blue' if track.is_vehicle else 'green')
            ax.add_patch(Polygon(points, closed=True, fill=True, alpha=.9, color=c))
            points_plus = np.vstack([points, points[:1]])
            ax.plot(*points_plus.T, color='k', alpha=.5)
        ax.set_aspect('equal')
        if show_tracks:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            for track in active_tracks:
                c = 'red' if track.is_player else ('blue' if track.is_vehicle else 'green')
                lw = 4 if track.is_player else 2
                linestyle = '--' if track.is_player else '-'
                ax.plot(track._get_data('posx'), track._get_data('posy'), color=c, alpha=.25, linewidth=lw, linestyle=linestyle)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        ax.set_xlabel('$x$ ($[m]$)')
        ax.set_ylabel('$y$ ($[m]$)')
        return fig, ax

    def local_occupancy_grid(self, t, resolution=(100, 100), radii=(30, 30), focal_track=None, z_radius=8.):

        if focal_track is None:
            focal_track = self.player_veh

        active_tracks = self.get_active_tracks(t)

        if focal_track not in active_tracks:
            raise ValueError

        R = lambda angle_deg: np.array([
            [np.cos(np.radians(angle_deg)), -np.sin(np.radians(angle_deg))],
            [np.sin(np.radians(angle_deg)), np.cos(np.radians(angle_deg))]
        ])

        def local_to_global(local_points):
            Rback = R(entity.yaw)
            offset = np.array([entity.posx, entity.posy])
            return (Rback @ local_points.T).T + offset

        # def global_to_local(global_points):
        #     Rto = R(-entity.yaw)
        #     offset = np.array([entity.posx, entity.posy])
        #     return (Rto @ (global_points - offset).T).T

        def points_in_polygon(polygon, pts):
            pts = np.asarray(pts,dtype='float32')
            polygon = np.asarray(polygon,dtype='float32')
            contour2 = np.vstack((polygon[1:], polygon[:1]))
            test_diff = contour2-polygon
            mask1 = (pts[:,None] == polygon).all(-1).any(-1)
            m1 = (polygon[:,1] > pts[:,None,1]) != (contour2[:,1] > pts[:,None,1])
            slope = ((pts[:,None,0]-polygon[:,0])*test_diff[:,1])-(test_diff[:,0]*(pts[:,None,1]-polygon[:,1]))
            m2 = slope == 0
            mask2 = (m1 & m2).any(-1)
            m3 = (slope < 0) != (contour2[:,1] < polygon[:,1])
            m4 = m1 & m3
            count = np.count_nonzero(m4,axis=-1)
            mask3 = ~(count%2==0)
            mask = mask1 | mask2 | mask3
            return mask

        def intersects_with_point(entity, XY):
            poly = get_bbox(entity)
            return points_in_polygon(poly, XY)

            

        entity = focal_track.get_interpolated_state(t)

        xe, ye = entity.posx, entity.posy
        left = -radii[0]
        right = radii[0]
        bottom = -radii[1]
        top = radii[1]
        points_in_focal_frame = np.hstack([
            a.ravel().reshape((-1, 1))
            for a in np.meshgrid(
                np.linspace(left, right, resolution[0]),
                np.linspace(bottom, top, resolution[1]),
                indexing='ij', # Returns arrays of shape Nx, Ny
            )
        ])

        points = local_to_global(points_in_focal_frame)

        mask = np.zeros(resolution, dtype=bool)

        for track in active_tracks:
            other_entity = track.get_interpolated_state(t)
            if abs(other_entity.posz - entity.posz) > z_radius:
                continue
            mask = np.logical_or(
                mask,
                intersects_with_point(other_entity, points).reshape(resolution)
            )
        mask = mask.T # Go back to Ny, Nx
        return np.logical_not(mask)

    def __init__(self, entities_or_binfpath=None, pbar=True):
        self.entities_or_binfpath = entities_or_binfpath
        self.pbar = pbar
        if isinstance(self.entities_or_binfpath, str):
            self.entities_or_binfpath = read_data(self.entities_or_binfpath)

    @property
    def trackgroups(self):
        if not hasattr(self, '_trackgroups'):
            self._trackgroups = {}
            assert self.entities_or_binfpath is not None
            for entity in self.entities_or_binfpath if not self.pbar else tqdm(
                self.entities_or_binfpath, unit='entities', desc='Associating entities with tracks'
            ):
                self.associate(entity)
        return self._trackgroups

    @trackgroups.setter
    def trackgroups(self, value):
        self._trackgroups = value

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
        return list(sorted(set(out), key=lambda x: x.tmin))

    def get_active_tracks(self, time):
        """Return a list of tracks that are active at the given time."""
        active_tracks = []
        for track in self.tracks:
            if track.tmin <= time < track.tmax:
                active_tracks.append(track)
        return active_tracks

    def tmin_from_tracks(self):
        return min([track.tmin for track in self.tracks])

    def tmax_from_tracks(self):
        return max([track.tmax for track in self.tracks])

    @property
    def tmin(self):
        return min([entity.wall_time for entity in self.entities_or_binfpath])

    @property
    def tmax(self):
        return max([entity.wall_time for entity in self.entities_or_binfpath])

    @property
    def player_ped(self):
        return self._get_first_track_by_prop(test=lambda track: track.is_player and not track.is_vehicle)

    @property
    def player_veh(self):
        return self._get_first_track_by_prop(test=lambda track: track.is_player and track.is_vehicle)
    
    def _get_first_track_by_prop(self, test):
        tested_tracks = [track for track in self.tracks if test(track)]
        if len(tested_tracks) == 0:
            return None
        elif len(tested_tracks) > 1:
            warn(
                "Multiple %s tracks found. Using the longest one. " % test
            )
            tested_tracks.sort(key=lambda track: track.duration)
        return tested_tracks[-1]

    def merge_player_tracks(self):
        tracks = list(sorted(self.tracks, key=lambda track: track.tmin))
        starting_num_tracks = len(tracks)

        # Identify player vehicles.
        player_vehicle = None
        player_ped = None

        new_tracks = []
        for track in tracks:
            if track.is_player:
                if track.is_vehicle:
                    if player_vehicle is None:
                        player_vehicle = track
                    else:
                        player_vehicle = player_vehicle + track
                else:
                    if player_ped is None:
                        player_ped = track
                    else:
                        player_ped = player_ped + track
                track = None    
            new_tracks.append(track)

        if player_vehicle is not None:
            new_tracks.append(player_vehicle)
        
        if player_ped is not None:
            new_tracks.append(player_ped)

        ending_num_tracks = self._delete_merged_tracks(new_tracks)

        assert starting_num_tracks >= ending_num_tracks
        return starting_num_tracks - ending_num_tracks

    def merge_tracks_by_id(self):
        tracks = list(sorted(self.tracks, key=lambda track: track.tmin))
        starting_num_tracks = len(tracks)

        all_ids = set()
        for track in tracks:
            all_ids.update(track.ids)

        track_merges = []
        #pbar = tqdm(total=len(tracks)*len(tracks), unit='pair', desc='Merging tracks by ID')
        for i, track in enumerate(tracks):
            this_track_merges = [i]
            for j, other_track in enumerate(tracks):
                #pbar.update()
                if track is other_track:
                    continue
                if (
                    (track.is_vehicle and not other_track.is_vehicle)
                    or
                    (not track.is_vehicle and other_track.is_vehicle)
                ):
                    continue
                if (
                    set(track.ids).intersection(set(other_track.ids)) 
                    and track.can_merge(other_track)
                ):
                    this_track_merges.append(j)
            track_merges.append(this_track_merges)
        
        merge_sets = self._create_merge_sets(track_merges)
        new_tracks = self._merge_by_mergesets(tracks, merge_sets)
            
        ending_num_tracks = self._delete_merged_tracks(new_tracks)

        assert starting_num_tracks >= ending_num_tracks
        return starting_num_tracks - ending_num_tracks

    def merge_tracks_where_possible(self, method='dist'):
        tracks = list(sorted(self.tracks, key=lambda track: track.tmin))
        starting_num_tracks = len(tracks)

        if method == 'brute':
            pbar = tqdm(total=len(tracks)*len(tracks), unit='pair', desc='Merging tracks.')
            for i in range(len(tracks)):
                for j in range(len(tracks)):
                    pbar.update()
                    track_i = tracks[i]
                    track_j = tracks[j]
                    if i != j and track_i is not None and track_j is not None:
                        if track_i.can_merge(track_j):
                            merged = track_i + track_j
                            if merged is track_i:
                                tracks[j] = None
                            else:
                                tracks[i] = None

        else:
            import scipy.spatial
            assert method in ('dist', 'kdtree')
        
            # Make a big point cloud of all the points in all the tracks.
            points = np.vstack([
                np.hstack([
                    track._get_data(key).reshape((-1, 1))
                    for key in AFFINITY_KEYS
                ])
                for track in tracks
            ]) * AFFINITY_KEY_WEIGHTS

            # Make an array of ID indicators for which track each point belongs to.
            which_track = np.hstack([
                np.ones((len(track),), dtype=int) * i
                for i, track in enumerate(tracks)
            ])

            if method == 'kdtree':
                # Make a KD-tree of the points.
                kd_tree = scipy.spatial.cKDTree(points)

            # For each track, find the points that are within a certain distance of its points.
            track_merges = []
            for itrack, track in enumerate(tqdm(tracks, unit='track', desc='Merging tracks by %s' % method)):
                # if True in [itrack in mergees for mergees in track_merges]:
                #     continue
                
                points_for_this_track = points[which_track == itrack]

                if method == 'dist':

                    distances_to_tracks = []
                    for jtrack, other_track in enumerate(tracks):
                        points_for_other_track = points[which_track == jtrack]
                        dist_arr = scipy.spatial.distance.cdist(points_for_this_track, points_for_other_track)
                        distances_to_tracks.append(dist_arr.min())

                    tracks_near_this_track = list(np.argwhere(np.array(distances_to_tracks) <= track.merge_threshold).ravel())
                    track_merges.append(tracks_near_this_track)

                else:
                    assert method == 'kdtree'
                    #dist, idx = kd_tree.query(query, k=4, distance_upper_bound=track.unaffinity_threshold)
                    neighbors_for_each_point = kd_tree.query_ball_point(points_for_this_track, track.unaffinity_threshold)
                    all_neighbors = np.hstack([a for a in neighbors_for_each_point])
                    neighbor_set = np.unique(all_neighbors)
                    # nearby_point_indices = np.unique(idx[dist != np.inf])
                    tracks_nearby = list(np.unique(which_track[neighbor_set]))
                    assert itrack in tracks_nearby
                    #if itrack in mergees:
                    #    mergees.remove(itrack)
                    track_merges.append(tracks_nearby)

            
            merge_sets = self._create_merge_sets(track_merges)
            tracks = self._merge_by_mergesets(tracks, merge_sets)


        # Remove any tracks that were merged away.
        ending_num_tracks = self._delete_merged_tracks(tracks)

        assert starting_num_tracks >= ending_num_tracks
        return starting_num_tracks - ending_num_tracks

    @staticmethod
    def _create_merge_sets(track_merges):
        # Find overlapping sets.
        merge_sets = []
        for mergees in track_merges:
            mergees = set(mergees)
            has_overlap = False
            for existing_set in merge_sets:
                if existing_set.intersection(mergees):
                    # There's an overlap; replace the existing set with the union one.
                    new_set = set(existing_set).union(mergees)
                    merge_sets.remove(existing_set)
                    merge_sets.append(new_set)
                    has_overlap = True
            if not has_overlap:
                merge_sets.append(mergees)
        return merge_sets

    @staticmethod
    def _merge_by_mergesets(tracks, merge_sets):
        # Do the merging.
        for i, mergees in enumerate(merge_sets):
            if len(mergees) > 0:
                for j in mergees:
                    if j != i:
                        ti = tracks[i]
                        tj = tracks[j]
                        if ti is not None and tj is not None:
                            merged = tracks[i] + tracks[j]
                            if merged is ti:
                                tracks[j] = None
                            else:
                                tracks[i] = None
        return tracks

    def _delete_merged_tracks(self, tracks_or_Nones):
        # Remove any tracks that were merged away.
        new_trackgroups = {}
        ending_num_tracks = 0
        for track in tracks_or_Nones:
            if track is None:
                continue
            ending_num_tracks += 1
            for i in track.ids:
                if i not in new_trackgroups:
                    new_trackgroups[i] = []
                new_trackgroups[i].append(track)
        self.trackgroups = new_trackgroups
        return ending_num_tracks

    def show_tracks(self, ax=None, plot_3d=False, **kw_plot):
        kw_plot.setdefault('alpha', 0.75)

        if ax is None:
            if not plot_3d:
                fig, ax = plt.subplots()
                ax.set_aspect('equal')
            else:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
        
        for track in self.tracks:
            x = track._get_data('posx')
            y = track._get_data('posy')
            if plot_3d:
                z = track._get_data('posz')
                data = x, y, z
            else:
                data = x, y
            ax.plot(*data, **kw_plot)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            if plot_3d:
                ax.set_zlabel('z')

        ax.get_figure().tight_layout()

        return ax.get_figure(), ax

    @property
    def protocol_definition(self):
        return self.tracks[0].protocol_definition


def read_data_main(plot_3d=False, fname=join(HOME, 'data', 'gta', 'velocity_prediction', 'Protocol V2'), search_for_truncated=False):
    if fname.endswith('\\') or fname.endswith('/') or not fname.endswith('.bin'):
        from glob import glob
        if search_for_truncated:
            search_path = join(fname, 'GTA_recording*-truncated.bin')
        else:
            search_path = join(fname, "GTA_recording*bin")
        binfiles = list(sorted(glob(search_path)))
        fname = binfiles[-1]
        print('Found most recent binfile:', fname)

    #track_manager = TrackManager(fname)
    track_manager = cached_TrackManager_fetch(fname)

    #track_manager.show_tracks(plot_3d=False)

    # n_merged = track_manager.merge_tracks_where_possible()
    n_merged = track_manager.merge_player_tracks()
    if n_merged:
        print('Merged away', n_merged, 'player tracks.')

    # n_merged = track_manager.merge_tracks_by_id()
    # # if n_merged:
    # print('Merged away', n_merged, 'tracks by ID.')

    t_mean = (track_manager.tmin + track_manager.tmax) / 2.
    if track_manager.protocol_definition.version >= version.parse('2'):
        track_manager.show_occupancy(t_mean)
    #active_tracks = track_manager.get_active_tracks(t_mean)
 
    # Make some pretty plots.
    fig_dir = join(HERE, '..', '..', 'doc')

    for minimum_length in 10,:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_aspect("equal")
        for track in track_manager.tracks:

            if len(track) < minimum_length or track.has_constant_position:
                continue

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
            ax.scatter(X, Y, color=line._color, s=1, alpha=.2)
            
        ax.set_xlabel('$x$ $[m]$')
        ax.set_ylabel('$y$ $[m]$')
        vehicle_tracks = [track for track in track_manager.tracks if len(track) >= minimum_length and track.is_vehicle and not track.has_constant_position]
        pla_veh_tracks = [track for track in vehicle_tracks if len(track) >= minimum_length and track.is_player]
        ped_tracks = [track for track in track_manager.tracks if not track.is_vehicle and not track.has_constant_position]
        pla_ped_tracks = [track for track in ped_tracks if track.is_player]
        print('Lengths of player ped tracks:', [
            '%d ents, %.1f sec' % (len(track._entities), track.tmax - track.tmin) for track in pla_ped_tracks
        ])
        pla_veh_tracks.sort(key=lambda track: track.tmin)
        print('Lengths of player veh tracks:', [
            '%d ents, %.1f sec' % (len(track._entities), track.tmax - track.tmin) for track in pla_veh_tracks
        ])

        if len(pla_veh_tracks) > 1:
            print('Affinity distances between subsequent (by t0) tracks')
            track = pla_veh_tracks[0]
            for i, track2 in enumerate(pla_veh_tracks[1:]):
                s1 = track._entities[-1]
                x1 = np.array([getattr(s1, attr) for attr in AFFINITY_KEYS])
                s2 = track2._entities[0]
                x2 = np.array([getattr(s2, attr) for attr in AFFINITY_KEYS])
                dist = np.linalg.norm(x1 - x2)
                print('%d to %d: %.2f' % (i, i+1, dist))
                track = track2

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
        # ax.set_xlim(new_lowx, new_highx)
        # ax.set_ylim(new_lowy, new_highy)
        # ax.scatter(
        #     [track._get_data('posx')[0] for track in track_manager.tracks if len(track) < minimum_length],
        #     [track._get_data('posy')[0] for track in track_manager.tracks if len(track) < minimum_length],
        #     color='black', s=42, alpha=.8, marker='+',
        # )
        for ext in ['png', 'pdf']:
            fig.savefig(join(fig_dir, 'associated_tracks-minlen_%d.%s' % (minimum_length, ext)))


    player_veh = track_manager.player_veh
    if player_veh is not None:
        fig, (ax, bx) = plt.subplots(ncols=2, figsize=(12, 6))
        T = np.linspace(player_veh.tmin, player_veh.tmax, 1200)
        fig.colorbar(
            bx.scatter(player_veh.get('posx', T), player_veh.get('posy', T), c=player_veh.get('posz', T), s=2, alpha=.9),
            ax=bx,
            label='Z'
        )
        bx.scatter(player_veh.get('posx', T[0]),  player_veh.get('posy', T[0]),  c='green', label='Start', marker='s', s=42)
        bx.scatter(player_veh.get('posx', T[-1]), player_veh.get('posy', T[-1]), c='red',   label='End',   marker='s', s=42)
        bx.set_aspect('equal')
        bx.legend()
        bx.set_xlabel('$x$')
        bx.set_ylabel('$y$')
        bx.set_title('Position $[m]$')

        velx = player_veh.get('velx', T)
        vely = player_veh.get('vely', T)
        velz = player_veh.get('velz', T)
        ax.plot(T, velx, label='$v_x$')
        ax.plot(T, vely, label='$v_y$')
        ax.plot(T, velz, label='$v_z$')
        spd = np.sqrt(velx**2 + vely**2 + velz**2)
        ax.plot(T, spd, label='overall speed ($\sqrt{v_x^2 + v_y^2 + v_z^2}$)', linestyle='--')
        ax.set_title('Velocity or Speed $[m/s]$')
        ax.set_xlabel('Time $[s]$')
        ax.legend()
        fig.tight_layout()
        # ax.set_xlim(new_lowx, new_highx)
        # ax.set_ylim(new_lowy, new_highy)
        for ext in ['png', 'pdf']:
            fig.savefig(join(fig_dir, 'player_vehicle.%s' % ext))

        x_ego = [d.posx for d in player_veh._entities]
        y_ego = [d.posy for d in player_veh._entities]
        z_ego = [d.posz for d in player_veh._entities]


   
    plt.show()


if __name__ == "__main__":
    read_data_main()
