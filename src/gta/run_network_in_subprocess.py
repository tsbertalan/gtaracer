import time, multiprocessing, numpy as np
import gta.train_velocity_predictor


def execute_network_subprocess(network_path, pipe):

    # Load the network.
    print('Loading from', network_path)
    oflow_velocity_model = gta.train_velocity_predictor.\
        VelocityPredictorFromOpticalFlow.load_from_checkpoint(network_path)
    
    # Run until the subprocess is killed.
    _last_small_image = None
    while True:
        # Get the next image.
        t, full_frame = pipe.recv()
        if full_frame is None:
            break
       
        full_frame = np.array(full_frame)
        current_small_image = gta.train_velocity_predictor.shrink_img_for_oflow(full_frame)
        if _last_small_image is None:
            vel = None
        else:
            oflow = gta.train_velocity_predictor.convert_image_pair_to_optical_flow(
                _last_small_image, current_small_image
            )
            oflow = gta.train_velocity_predictor.reshape_oflow_for_net(oflow)
            vel = oflow_velocity_model.predict_from_numpy(oflow)
            vel = float(vel)
        _last_small_image = current_small_image
        pipe.send((t, vel))


class SubprocessNetworkExecutor:

    def __init__(self, network_load_path):
        self.network_load_path = network_load_path
        
        # Use a pipe to communicate with the child process.
        self.parent_conn, child_conn = multiprocessing.Pipe()
      
        # Start a worker.
        self.process = multiprocessing.Process(
            target=execute_network_subprocess,
            args=(self.network_load_path, child_conn)
        )
        self.process.start()

        self._is_ready = True
    
    @property
    def is_ready(self):
        if not hasattr(self, '_is_ready'):
            return False
        else:
            return self._is_ready

    def __call__(self, full_frame):
        t = time.time()
        self.parent_conn.send((t, full_frame))

        # If there are any results, return them.
        # Otherwise, return None.
        pipe_nonempty = self.parent_conn.poll(0)
        if pipe_nonempty:
            res = self.parent_conn.recv()
            return res

    def shutdown(self):
        # First, send a None to the child process to tell it to stop.
        self.parent_conn.send(None)

        # Then, wait for it to finish.
        self.process.join()
        