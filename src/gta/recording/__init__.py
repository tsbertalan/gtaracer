import time, os
from multiprocessing import Process, Manager

import numpy as np

import gta.utils

class BaseTask(object):

    def __call__(self):
        # Do something and return a result.
        raise NotImplementedError


def work(Task, workSignal, resultsQueueWorker, period, waitPeriod=1, **taskKwargs):
    task = Task(**taskKwargs)
    while True:
        if workSignal.is_set():
            result = task()
            if result is not None:
                now = time.monotonic()
                resultsQueueWorker.put((now, result))
            time.sleep(period)
        else:
            time.sleep(waitPeriod)


class BaseRecorder(object):

    def __init__(self, 
        period, 
        Task=BaseTask, waitPeriod=1, giveQueueDirectly=False,
        **taskKwargs
        ):
        """
        Parameters
        ----------
        period : float
            Time in seconds between calls to the Task.
        Task : BaseTask
            The task to be instantiated by the worker and run repeatedly.
            Must implement __call__ method with no arguments.
            If None, no worker is created.
        waitPeriod : float
            Minimum time in seconds between stopping work and resuming.
            (Allows for longer sleep time between checks for whether to begin work.)
        giveQueueDirectly : bool
            Whether the resultsQueue should be passed to the Task constructor directly.
        **taskKwargs
            Passed on to constructor of Task.
        """

        self.Task = Task

        self.taskKwargs = taskKwargs
        self.period = period
        self.waitPeriod = waitPeriod
        self.giveQueueDirectly = giveQueueDirectly

    def kill_subprocesses(self):
        print('Killing worker for', self.__class__.__name__)
        if hasattr(self, 'worker'):
            self.worker.terminate()
            self.worker.join()

    def create_subprocesses(self):
        self.manager = Manager()
        self.workSignal = self.manager.Event()
        self.resultsQueue = self.manager.Queue()
        self._resultsList = []
        
        if self.Task is not None:
            passed = {'waitPeriod': self.waitPeriod}
            passed.update(self.taskKwargs)
            if self.giveQueueDirectly:
                passed['resultsQueue'] = self.resultsQueue
            self.workerArgs = (self.Task, self.workSignal, self.resultsQueue, self.period)
            self.worker = Process(
                target=work, 
                args=self.workerArgs,
                kwargs=passed,
            )
            self.worker.daemon = True
            self.worker.start()

    def start(self):
        self.workSignal.set()

    def stop(self):
        self.workSignal.clear()

    @property
    def resultsList(self):
        while not self.resultsQueue.empty():
            self._resultsList.append(self.resultsQueue.get())
        return self._resultsList

    @property
    def results(self):
        if len(self.resultsList) > 0:
            return np.stack([r for (t, r) in self.resultsList])
        else:
            return np.array([])

    @property
    def times(self):
        return np.array([t for (t, r) in self.resultsList])

    def toSave(self):
        return dict(
            results=self.results,
            times=self.times,
        )

    def save(self, fpath=None, compressed=False, **kwargs):
        start = time.time()
        if compressed:
            saver = np.savez_compressed
        else:
            saver = np.savez
        if fpath is None:
            fpath = os.path.join(os.path.expanduser('~'), 'data', '%s-%s.npz' % (
                type(self).__name__, start,
            ))
        dirname = os.path.dirname(fpath)
        gta.utils.mkdir_p(dirname)
        print('Saving to %s ... ' % fpath, end='')
        toSave = self.toSave(**kwargs)
        toSave.setdefault('dtype', 'object')
        saver(fpath, **toSave)
        print('done (%.3g s).' % (time.time() - start,))
        return toSave


class DemoTask(BaseTask):

        def __init__(self, multiplier=1):
            self.multiplier = multiplier

        def __call__(self):
            return self.multiplier * time.time()


class DemoRecorder(BaseRecorder):

    def __init__(self, period=1, **k):
        super(self.__class__, self).__init__(period, Task=DemoTask, **k)

    
# from. import vision, keyboard, gamepad, unified