import time
from multiprocessing import Process, Manager

import numpy as np

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
                now = time.time()
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
        self.manager = Manager()
        self.workSignal = self.manager.Event()
        self.resultsQueue = self.manager.Queue()
        self._resultsList = []
        
        if Task is not None:
            passed = {'waitPeriod': waitPeriod}
            passed.update(taskKwargs)
            if giveQueueDirectly:
                passed['resultsQueue'] = self.resultsQueue
            self.workerArgs = (Task, self.workSignal, self.resultsQueue, period)
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
        return np.stack([r for (t, r) in self.resultsList])

    @property
    def times(self):
        return [t for (t, r) in self.resultsList]


class DemoTask(BaseTask):

        def __init__(self, multiplier=1):
            self.multiplier = multiplier

        def __call__(self):
            return self.multiplier * time.time()


class DemoRecorder(BaseRecorder):

    def __init__(self, period=1, **k):
        super(self.__class__, self).__init__(period, Task=DemoTask, **k)

    

