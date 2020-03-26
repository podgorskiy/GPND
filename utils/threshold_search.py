# Copyright 2018-2020 Stanislav Pidhorskyi
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

import numpy as np
import itertools


def find_maximum(f, min_x, max_x, epsilon=1e-5):
    def binary_search(l, r, fl, fr, epsilon):
        mid = l + (r - l) / 2
        fm = f(mid)
        binary_search.eval_count += 1
        if (fm == fl and fm == fr) or r - l < epsilon:
            return mid, fm
        if fl > fm >= fr:
            return binary_search(l, mid, fl, fm, epsilon)
        if fl <= fm < fr:
            return binary_search(mid, r, fm, fr, epsilon)
        p1, f1 = binary_search(l, mid, fl, fm, epsilon)
        p2, f2 = binary_search(mid, r, fm, fr, epsilon)
        if f1 > f2:
            return p1, f1
        else:
            return p2, f2

    binary_search.eval_count = 0

    best_th, best_value = binary_search(min_x, max_x, f(min_x), f(max_x), epsilon)
    # print("Found maximum %f at x = %f in %d evaluations" % (best_value, best_th, binary_search.eval_count))
    return best_th, best_value


def find_maximum_mv(f, min_x, max_x, ftoll=1e-6, xtoll=1e-6, n=4, verbose=False, max_iter=5):
    vf = np.vectorize(f, signature='(i)->()')

    if not n >= 4:
        raise ValueError("n must be >= 3, got %d instead" % n)

    min_x = np.asarray(min_x)
    max_x = np.asarray(max_x)

    if min_x.shape != max_x.shape:
        raise ValueError("min_x and max_x must be of the same shape, but got {} and {}".format(min_x.shape, max_x.shape))

    d = min_x.shape[0]

    cube = np.asarray(list(itertools.product(*zip([0] * d, [1] * d))))
    cube = np.reshape(cube, [2] * d + [d])
    linspace = np.linspace(0.0, 1.0, n)
    grid = np.stack(np.meshgrid(* [linspace] * d, indexing='ij'), axis=-1)

    min_x = np.reshape(min_x, [1] * d + [d])
    max_x = np.reshape(max_x, [1] * d + [d])

    eval_count = 0
    for _ in range(max_iter):
        grid_this = min_x + grid * (max_x - min_x)
        values = vf(grid_this)
        eval_count += n ** d
        rargmax = np.argmax(values)
        iargmax = np.asarray(np.unravel_index(rargmax, values.shape))
        i_min = iargmax - 5
        i_max = iargmax + 5
        i_min = np.ravel_multi_index(i_min, grid_this.shape[:-1], mode='clip')
        i_max = np.ravel_multi_index(i_max, grid_this.shape[:-1], mode='clip')

        min_x, max_x = grid_this.reshape(-1, d)[i_min], grid_this.reshape(-1, d)[i_max]

        vmax = values.flatten()[rargmax]
        cmax = grid_this.reshape(-1, d)[rargmax]

        if all(abs(x - vmax) < ftoll for x in values.flatten()) or np.all(max_x - min_x < xtoll):
            break

    if verbose:
        print("Found maximum {} at x = {} in {} evaluations".format(vmax, cmax, eval_count))
    return cmax, vmax


def find_maximum_mv_it(f, min_x, max_x, ftoll=1e-6, xtoll=1e-6, n=4, verbose=False):
    min_x = np.asarray(min_x)
    max_x = np.asarray(max_x)

    if min_x.shape != max_x.shape:
        raise ValueError("min_x and max_x must be of the same shape, but got {} and {}".format(min_x.shape, max_x.shape))

    d = min_x.shape[0]

    x = (min_x + max_x) / 2.0

    eval = 0
    for _i in range(100):
        max_delta = 0
        for i in range(d):
            old = x[i]

            def _f(a):
                x[i] = a
                return f(x)

            _x, _y = find_maximum(_f, min_x[i], max_x[i], ftoll)
            eval += 1
            max_delta = max(max_delta, abs(old - _x))
            x[i] = _x
        if max_delta < xtoll:
            break

    if verbose:
        print("Found maximum {} at x = {} in {} evaluations".format(_y, x, eval))
    return x, _y
