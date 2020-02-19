# Copyright 2020 Stanislav Pidhorskyi
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


import multiprocessing
from multiprocessing import Process, Pipe
import os
import functools


def _get_gpu_count(conn):
    import torch
    gpu_count = torch.cuda.device_count()
    conn.send(gpu_count)
    conn.close()


def get_gpu_count():
    parent_conn, child_conn = Pipe()
    p = Process(target=_get_gpu_count, args=(child_conn,))
    p.start()
    gpu_count = parent_conn.recv()
    p.join()
    return gpu_count


def set_numpy_treads(world_size):
    cpu_count = multiprocessing.cpu_count()
    os.environ["OMP_NUM_THREADS"] = str(max(1, int(cpu_count / world_size)))


def _init(queue):
    global idx
    idx = queue.get()


def _f(func, arg):
    global idx
    import torch
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(idx)
    device = torch.cuda.current_device()
    print("Running on GPU: %d, %s" % (idx, torch.cuda.get_device_name(device)))
    return func(arg)


def map(func, world_size, iteratable):
    manager = multiprocessing.Manager()
    idQueue = manager.Queue()

    for i in range(world_size):
        idQueue.put(i)

    p = multiprocessing.Pool(world_size, _init, (idQueue,))

    return p.map(functools.partial(_f, func), iteratable)
