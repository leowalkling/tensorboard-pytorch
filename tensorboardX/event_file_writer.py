# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Writes events to disk in a logdir."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os.path
import socket
import threading
import time
from time import perf_counter
from collections import deque

import six
from six.moves.queue import Queue, Empty

from .proto import event_pb2
from .record_writer import RecordWriter, directory_check


class EventsWriter(object):
    '''Writes `Event` protocol buffers to an event file.'''

    def __init__(self, file_prefix, filename_suffix=''):
        '''
        Events files have a name of the form
        '/some/file/path/events.out.tfevents.[timestamp].[hostname]'
        '''
        self._file_name = file_prefix + ".out.tfevents" + filename_suffix

        # Open(Create) the log file with the particular form of name.
        logging.basicConfig(filename=self._file_name)

        self._num_outstanding_events = 0

        self._py_recordio_writer = RecordWriter(self._file_name)

        # Initialize an event instance.
        self._event = event_pb2.Event()

        self._event.wall_time = time.time()

        self.write_event(self._event)

    def write_event(self, event):
        '''Append "event" to the file.'''

        # Check if event is of type event_pb2.Event proto.
        if not isinstance(event, event_pb2.Event):
            raise TypeError("Expected an event_pb2.Event proto, "
                            " but got %s" % type(event))
        return self._write_serialized_event(event.SerializeToString())

    def _write_serialized_event(self, event_str):
        self._num_outstanding_events += 1
        self._py_recordio_writer.write(event_str)

    def flush(self):
        '''Flushes the event file to disk.'''
        self._num_outstanding_events = 0
        return True

    def close(self):
        '''Call self.flush().'''
        return_value = self.flush()
        self._py_recordio_writer.close()
        return return_value


class RequestStop(object):
    pass


class EventFileWriter(object):
    """Writes `Event` protocol buffers to an event file.
    The `EventFileWriter` class creates an event file in the specified directory,
    and asynchronously writes Event protocol buffers to the file. The Event file
    is encoded using the tfrecord format, which is similar to RecordIO.
    @@__init__
    @@add_event
    @@flush
    @@close
    """

    def __init__(self, logdir, max_queue=None, flush_secs=None, filename_suffix='', worker=None):
        """Creates a `EventFileWriter` and an event file to write to.
        On construction the summary writer creates a new event file in `logdir`.
        This event file will contain `Event` protocol buffers, which are written to
        disk via the add_event method.
        The other arguments to the constructor control the asynchronous writes to
        the event file:
        *  `flush_secs`: How often, in seconds, to flush the added summaries
           and events to disk.
        *  `max_queue`: Maximum number of summaries or events pending to be
           written to disk before one of the 'add' calls block.
        Args:
          logdir: A string. Directory where event file will be written.
          max_queue: Integer. Size of the queue for pending events and summaries.
          flush_secs: Number. How often, in seconds, to flush the
            pending events and summaries to disk.
        """

        self._logdir = logdir
        directory_check(self._logdir)
        self._ev_writer = EventsWriter(os.path.join(
            self._logdir, "events"), filename_suffix)
        self._closed = False

        if worker is not None:
            if max_queue is not None:
                raise ValueError("cannot specify max_queue when giving an existing event logger thread")
            if flush_secs is not None:
                raise ValueError("cannot specify flush_secs when giving an existing event logger thread")
            self._worker_owned = False
            self._worker = worker
        else:
            self._worker_owned = True
            self._worker = EventLoggerThread(max_queue or 30,
                                             flush_secs or 120)
            self._worker.start()

    def get_logdir(self):
        """Returns the directory where event file will be written."""
        return self._logdir

    def reopen(self):
        """Reopens the EventFileWriter.
        Can be called after `close()` to add more events in the same directory.
        The events will go into a new events file.
        Does nothing if the EventFileWriter was not closed.
        """
        if self._closed:
            if self._worker_owned:
                self._worker.start()
            self._closed = False

    def add_event(self, event):
        """Adds an event to the event file.
        Args:
          event: An `Event` protocol buffer.
        """
        if not self._closed:
            self._worker.add_event(self._ev_writer, event)

    def flush(self):
        """Flushes the event file to disk.
        Call this method to make sure that all pending events have been written to
        disk.
        """
        if self._worker_owned:
            self._worker.request_stop()
            self._worker.join()
            worker = EventLoggerThread(self._worker.queue.maxsize, self._worker._flush_secs)
            worker.queue = self._worker.queue
            self._worker = worker
            self._worker.start()

    def close(self):
        """Flushes the event file to disk and close the file.
        Call this method when you do not need the summary writer anymore.
        """
        self._worker.close_event_writer(self._ev_writer)
        if self._worker_owned:
            self._worker.request_stop()
            self._worker.join()
            self._worker = None
        self._closed = True


class CloseEventWriter(object):
    __slots__ = "writer"
    def __init__(self, writer):
        self.writer = writer


class Flush(object):
    __slots__ = "writer", "event"

    def __init__(self, writer, event):
        self.writer = writer
        self.event = event


class EventLoggerThread(threading.Thread):
    """Thread that logs events."""
    _next_free_thread_number = 0

    def __init__(self, max_queue=30, flush_secs=120, name=None):
        """Creates an EventLoggerThread.
        Args:
          queue: A Queue from which to dequeue events.
          flush_secs: How often, in seconds, to flush the
            pending file to disk.
        """
        if name is None:
            thread_num = EventLoggerThread._next_free_thread_number =\
                EventLoggerThread._next_free_thread_number + 1
            name = "tensorboardX_thread_{:d}".format(thread_num)
        threading.Thread.__init__(self, daemon=True, name=name)
        self.daemon = True
        self._queue = deque()
        self._queue_lock = threading.Lock()
        self._queue_cond = threading.Condition(self._queue_lock)
        self._queue_nonempty = threading.Event()
        # self._queue_cond_lock = threading.Lock()
        # self._queue_cond_lock.acquire(False)
        self._queue_finish = threading.Event()
        self._max_queue = max_queue
        # self.queue = Queue(max_queue)
        self._flush_secs = flush_secs
        # The first event will be flushed immediately.
        self._next_event_flush_time = perf_counter()
        self._cancel_event = threading.Event()
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def close(self):
        return self.__exit__(None, None, None)

    def stop(self):
        self.request_stop()
        self.join()

    def request_stop(self):
        self._put(RequestStop())

    def _put(self, obj):
        q = self._queue
        with self._queue_lock:
            q.append(obj)
            if len(q) > 1:
                self._queue_nonempty.set()
                self._queue_finish.clear()

    def _get(self, timeout=None):
        q = self._queue
        l = self._queue_lock
        with self._queue_lock:
            if len(q) == 0:
                try:
                    l.release()
                    if not self._queue_nonempty.wait(timeout):
                        raise Empty()
                finally:
                    l.acquire()
            item = q.popleft()
            if len(q) == 0:
                self._queue_nonempty.clear()
                self._queue_finish.set()
            return item

    def close_event_writer(self, event_writer):
        self._put(CloseEventWriter(event_writer))

    def cancel_loop(self):
        self._cancel_event.set()

    def wait_for_queued_events(self):
        if len(self._queue) > 0:
            self._queue_finish.wait()

    def add_event(self, ev_writer, event):
        self._put((ev_writer, event))

    def run(self):
        cancel_event = self._cancel_event

        try:
            while not cancel_event.is_set():
                try:
                    msg = self._get(timeout=0.3)

                    if isinstance(msg, RequestStop):
                        cancel_event.set()
                        self._closed = True
                    elif self._closed:
                        pass  # If the writer is closed, the operations below cannot be performed
                    elif isinstance(msg, tuple):
                        ev_writer, event = msg
                        ev_writer.write_event(event)
                        # Flush the event writer every so often.
                        now = perf_counter()
                        if now > self._next_event_flush_time:
                            ev_writer.flush()
                            self._next_event_flush_time = now + self._flush_secs
                    elif isinstance(msg, CloseEventWriter):
                        ev_writer = msg.writer
                        ev_writer.close()
                    elif isinstance(msg, Flush):
                        ev_writer = msg.writer
                        ev_writer.flush()
                        now = perf_counter()
                        self._next_event_flush_time = now + self._flush_secs
                except Empty:
                    pass
        except Exception as e:
            import sys
            exc_info = sys.exc_info()
            import traceback
            traceback.print_exception(e, *exc_info[1:])
