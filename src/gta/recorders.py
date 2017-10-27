import time
from multiprocessing import Process, Manager

import numpy as np

class BaseTask(object):

    def __call__(self):
        # Do something and return a result.
        raise NotImplementedError


def work(Task, workSignal, resultQueue, period, waitPeriod=1, **taskKwargs):
    task = Task(**taskKwargs)
    while True:
        if workSignal.is_set():
            resultQueue.put(task())
            time.sleep(period)
        else:
            time.sleep(waitPeriod)


class BaseRecorder(object):

    def __init__(self, period, Task=BaseTask, waitPeriod=1, **taskKwargs):
        """
        Parameters
        ----------
        period : float
            Time in seconds between calls to the Task.
        Task : BaseTask
            The task to be instantiated by the worker and run repeatedly.
            Must implement __call__ method with no arguments.
        waitPeriod : float
            Minimum time in seconds between stopping work and resuming.
            (Allows for longer sleep time between checks for whether to begin work.)
        **taskKwargs
            Passed on to constructor of Task.
        """

        self.Task = Task
        self.taskKwargs = taskKwargs
        self.manager = Manager()
        self.workSignal = self.manager.Event()
        self.resultQueue = self.manager.Queue()
        self._resultsList = []

        passed = {'waitPeriod': waitPeriod}
        passed.update(taskKwargs)
        
        self.worker = Process(
            target=work, 
            args=(Task, self.workSignal, self.resultQueue, period),
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
        while not self.resultQueue.empty():
            self._resultsList.append(self.resultQueue.get())
        return self._resultsList

    @property
    def results(self):
        return np.stack(self.resultsList)


class DemoTask(BaseTask):

        def __init__(self, multiplier=1):
            self.multiplier = multiplier

        def __call__(self):
            return self.multiplier * time.time()


class DemoRecorder(BaseRecorder):

    def __init__(self, period=1, **k):
        super(self.__class__, self).__init__(period, Task=DemoTask, **k)

    

