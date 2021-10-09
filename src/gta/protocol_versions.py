from ctypes import c_uint, c_int, c_double, c_float, c_bool, Structure, c_char, sizeof

class EntityStateBase(Structure):
    pass

class EntityStateV1(EntityStateBase):
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

        # Absolute position in the world [meters].
        ("posx", c_float),
        ("posy", c_float),
        ("posz", c_float),

        # Euler angles in the world frame (right)?
        # TODO: check frames.
        ("roll",  c_float),
        ("pitch", c_float),
        ("yaw",   c_float),

        # Velocity [meters per second].
        ("velx", c_float),
        ("vely", c_float),
        ("velz", c_float),

        # Rotational velocity
        ("rvelx", c_float),
        ("rvely", c_float),
        ("rvelz", c_float),

        # Corresponds to Python's time.monotonic
        ("wall_time", c_double),

        # If the entity is on-screen, its coordinates in screen space [fractional].
        ("screenx", c_float),
        ("screeny", c_float),

        # Is the entity on-screen? This tends to err on the side of reporting False,
        # even if something like a tree is almost completely blocking it.
        # Or, maybe it just doesn't even count trees as occluding the entity.
        ("is_occluded", c_bool),

        # Whether the entity is a vehicle (vs a pedestrian).
        ("is_vehicle", c_bool),

        # Whether the entity is the player or player's vehicle.
        ("is_player", c_bool),

    ]


class EntityStateV2(EntityStateBase):
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

        # The entity's model.
        ("entity_type", c_int),

        # Absolute position in the world [meters].
        ("posx", c_float),
        ("posy", c_float),
        ("posz", c_float),

        # Euler angles in the world frame (right)?
        # TODO: check frames.
        ("roll",  c_float),
        ("pitch", c_float),
        ("yaw",   c_float),

        # Velocity [meters per second].
        ("velx", c_float),
        ("vely", c_float),
        ("velz", c_float),

        # Rotational velocity
        ("rvelx", c_float),
        ("rvely", c_float),
        ("rvelz", c_float),

        # The entity's model's bounding box [meters].
        ("bboxdx", c_float),
        ("bboxdy", c_float),
        
        # Corresponds to Python's time.monotonic
        ("wall_time", c_double),

        # If the entity is on-screen, its coordinates in screen space [fractional].
        ("screenx", c_float),
        ("screeny", c_float),

        # Is the entity on-screen? This tends to err on the side of reporting False,
        # even if something like a tree is almost completely blocking it.
        # Or, maybe it just doesn't even count trees as occluding the entity.
        ("is_occluded", c_bool),

        # Whether the entity is a vehicle (vs a pedestrian).
        ("is_vehicle", c_bool),

        # Whether the entity is the player or player's vehicle.
        ("is_player", c_bool),

        # Whether the entity is stopped at a light, if it's a vehicle.
        ("is_stopped_at_light", c_bool),

        # Whether the entity is damaged, if it's a vehicle.
        ("is_damaged", c_bool),
    ]

