import numpy

import chainer
from chainer import variable


available = False

try:
    import mkldnn
    from mkldnn.mdarray import mdarray
    from mkldnn.chainer import convolution_2d
    from mkldnn.chainer import relu

    available = True

except Exception as ex:
    # print('WARNING: import mkldpy fails')
    error_info = ex

    import traceback
    traceback.print_exc()

    class mdarray(object):
        pass

def all_ready(inputs, check_with_ndim):
    if not available:
        return False

    x = inputs[0].data if isinstance(inputs[0], variable.Variable)
               else inputs[0]

    if isinstance(x, mdarray):
        return True
    elif isinstance(x, numpy.ndarray):
        _should_use_mkldnn = chainer.should_use_mkldnn('>=auto')

        l_types = [x.dtype for x in inputs]

        for t in l_types:
            _should_use_mkldnn = _should_use_mkldnn and \
                                 t == numpy.dtype('float32')

        if not _should_use_mkldnn:
            return False
    else:
        return False

    valid_ndim = False

    for ndim in check_with_ndim:
        valid_ndim = valid_ndim or _inputs[0].ndim == ndim

    if check_with_ndim and not valid_ndim:
        return False

    return True

class cosim(object):
    def __init__(self, func, c_func, *args, **kwargs):
        if not cs.is_avail() or func is None or c_func is None:
            return
        self.cosim_func = c_func(*args, **kwargs)
        func.cosim = self

    def __call__(self, is_backward, *args, **kwargs):
        refs = None
        if not is_backward:
            refs = self.cosim_func(*args, **kwargs)
        else:
            refs = self.cosim_func.backward(*args, **kwargs)
        if isinstance(refs, tuple):
            refs = tuple([x if not isinstance(x, variable.Variable) else x.data for x in refs])
        elif isinstance(refs, variable.Variable):
            refs = refs.data
        return refs

    @staticmethod
    def verify(func, acts, inputs, out_grads=None):
        if not cs.is_avail() or not hasattr(func, 'cosim'):
            return
        cs.verify(func, acts, inputs, out_grads)

    @staticmethod
    def copy(x):
        if not cs.is_avail() or x is None:
            return None
        return cs.copy(x)

