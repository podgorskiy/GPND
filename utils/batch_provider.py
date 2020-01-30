# Copyright 2018 Stanislav Pidhorskyi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

try:
    from Queue import Queue, Empty
except ImportError:
    from queue import Queue, Empty
from threading import Thread, Lock, Event


def batch_provider(data, batch_size, processor, worker_count=1, queue_size=16, report_progress=False):
    class State:
        def __init__(self):
            self.current_batch = 0
            self.lock = Lock()
            self.batches_count = len(data) // batch_size + (1 if len(data) % batch_size != 0 else 0)
            self.quit_event = Event()
            self.queue = Queue(queue_size)
            self.batches_done_count = 0
            self.progress_bar = None
            if report_progress:
                self.progress_bar = ProgressBar(self.batches_count)

        def get_next_batch_it(self):
            try:
                self.lock.acquire()
                if self.quit_event.is_set() or self.current_batch == self.batches_count:
                    raise StopIteration
                cb = self.current_batch
                self.current_batch += 1
                return cb
            finally:
                self.lock.release()

        def push_done_batch(self, batch):
            try:
                self.lock.acquire()
                state.queue.put(batch)
                self.batches_done_count += 1
            finally:
                self.lock.release()

        def all_done(self):
            return self.batches_done_count == self.batches_count and state.queue.empty()

    state = State()

    def _worker():
        while not state.quit_event.is_set():
            try:
                cb = state.get_next_batch_it()
                data_slice = data[cb * batch_size:min((cb + 1) * batch_size, len(data))]
                b = processor(data_slice)
                state.push_done_batch(b)
            except StopIteration:
                break

    workers = []
    for i in range(worker_count):
        worker = Thread(target=_worker)
        worker.start()
        workers.append(worker)
    try:
        while not state.quit_event.is_set() and not state.all_done():
            item = state.queue.get()
            state.queue.task_done()
            yield item
            if state.progress_bar is not None:
                state.progress_bar.increment()

    except GeneratorExit:
        state.quit_event.set()
        while not state.queue.empty():
            try:
                state.queue.get(False)
            except Empty:
                continue
            state.queue.task_done()


class ProgressBar:
    def __init__(self, total_iterations, prefix='Progress:', suffix='', decimals=1, length=70, fill='#'):
        self.format_string = "\r%s |%%s| %%.%df%%%% [%%d/%d] %s" % (prefix, decimals, total_iterations, suffix)
        self.total_iterations = total_iterations
        self.length = length
        self.fill = fill
        self.current_iteration = 0

    def increment(self, val=1):
        self.current_iteration += val
        percent = 100 * (self.current_iteration / float(self.total_iterations))
        filled_length = int(self.length * self.current_iteration // self.total_iterations)
        bar = self.fill * filled_length + '-' * (self.length - filled_length)
        print(self.format_string % (bar, percent, self.current_iteration, ), end='\r')
        if self.current_iteration == self.total_iterations:
            print()

# Example
if __name__ == '__main__':
    from time import sleep

    def process(x):
        sleep(0.3)
        return x + x

    data = [x for x in range(200)]

    # Without printing progress
    batches = batch_provider(data, 5, process, 4, report_progress=False)

    for batch in batches:
        print(batch)

    # With printing progress
    batches = batch_provider(data, 5, process, 4, report_progress=True)

    for batch in batches:
        pass
