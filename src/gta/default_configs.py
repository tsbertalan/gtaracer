from os.path import join, expanduser

HOME = expanduser('~')
CODE_DIR = join(HOME, 'Dropbox', 'Projects', 'GTARacer', 'src')
VELOCITY_DATA_DIR = join(HOME, 'data', 'gta', 'velocity_prediction')
PROTOCOL_V2_DIR = join(VELOCITY_DATA_DIR, 'Protocol V2')
SCREEN_CONTROLLER_REC_DIR = PROTOCOL_V2_DIR
OFLOW_VEL_MODEL_SAVE_PATH = join(CODE_DIR, 'gta', 'nn', 'saved_models', 'oflow_vel_model.ckpt')