from os.path import join, expanduser

HOME = expanduser('~')
VELOCITY_DATA_DIR = join(HOME, 'data', 'gta', 'velocity_prediction')
PROTOCOL_V2_DIR = join(VELOCITY_DATA_DIR, 'Protocol V2')
# SCREEN_CONTROLLER_REC_DIR = join(HOME, 'data', 'gta', 'recordings')
SCREEN_CONTROLLER_REC_DIR = PROTOCOL_V2_DIR