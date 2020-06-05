from threading import Thread, Semaphore

import six
from logging import getLogger

from tfsnippet.utils import AutoInitAndCloseable
from .base import DataFlow

if six.PY2:
    from Queue import Queue
else:
    from queue import Queue

__all__ = ['ThreadingFlow']


class ThreadingFlow(DataFlow, AutoInitAndCloseable):
    """
    Data flow to prefetch from the source data flow in a background thread.

    Usage::

        array_flow = DataFlow.arrays([x, y], batch_size=256)
        with array_flow.threaded(prefetch=5) as df:
            for epoch in epochs:
                for batch_x, batch_y in df:
                    ...
    """

    EPOCH_END = object()
    """Object to mark an ending position of an epoch."""

    def __init__(self, source, prefetch):
        """
        Construct a :class:`ThreadingFlow`.

        Args:
            source (DataFlow): The source data flow.
            prefetch (int): Number of mini-batches to prefetch ahead.
                It should be at least 1.
        """
        # check the parameters
        if prefetch < 1:
            raise ValueError('`prefetch_num` must be at least 1')

        # memorize the parameters
        self._source = source
        self._prefetch_num = prefetch

        # internal states for background worker
        self._worker = None  # type: Thread
        self._batch_queue = None  # type: Queue
        self._epoch_counter = None  # counter for tracking the active epoch
        self._stopping = None
        self._worker_alive = None
        self._worker_ready_sem = None

    @property
    def source(self):
        """Get the source data flow."""
        return self._source

    @property
    def prefetch_num(self):
        """Get the number of batches to prefetch."""
        return self._prefetch_num

    def _worker_func(self):
        active_epoch = self._epoch_counter
        self._worker_alive = True
        self._worker_ready_sem.release()

        try:
            while not self._stopping:
                # iterate through the mini-batches in the current epoch
                for batch in self.source:
                    if self._stopping or active_epoch < self._epoch_counter:
                        break
                    self._batch_queue.put((active_epoch, batch))

                # put the epoch ending mark into the queue
                if not self._stopping:
                    self._batch_queue.put((active_epoch, self.EPOCH_END))

                # move to the next epoch
                active_epoch += 1
        except Exception:  # pragma: no cover
            getLogger(__name__).warning(
                '{} exited because of error.'.format(self.__class__.__name__),
                exc_info=True
            )
            raise
        finally:
            self._worker_alive = False

    def _init(self):
        # prepare for the worker states
        self._batch_queue = Queue(self.prefetch_num)
        self._epoch_counter = 0
        self._stopping = False
        self._worker_ready_sem = Semaphore(value=0)

        # create and start the worker
        self._worker = Thread(target=self._worker_func)
        self._worker.daemon = True
        self._worker.start()

        # wait for the thread to show up
        self._worker_ready_sem.acquire()

    def _close(self):
        try:
            # prevent the worker thread from further work
            self._stopping = True
            # exhaust all remaining queue items to notify the background worker
            while not self._batch_queue.empty():
                self._batch_queue.get()
            # wait until the worker exit
            self._worker.join()
        finally:
            self._worker = None
            self._batch_queue = None
            self._worker_ready_sem = None
            self._initialized = False

    def _minibatch_iterator(self):
        self.init()

        try:
            # iterate through one epoch
            while self._worker_alive:
                epoch, payload = self._batch_queue.get()
                if epoch < self._epoch_counter:
                    # we've got a remaining item from the last epoch, skip it
                    pass
                elif epoch > self._epoch_counter:  # pragma: no cover
                    # we've accidentally got an item from the future epoch
                    # it should be a bug, and we shall report it
                    raise RuntimeError('Unexpected entry from future epoch.')
                elif payload is self.EPOCH_END:
                    # we've got the epoch ending mark for the current epoch,
                    # so we should break the loop
                    break
                else:
                    # we've got a normal batch for the current epoch,
                    # so yield it
                    yield payload
        finally:
            self._epoch_counter += 1
