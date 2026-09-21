try:
    from . import graph, data, deep, env, metrics, util, tfm, limix, graphpfn

    from .util import *  # noqa: F403
except ImportError:
    # Heavy optional deps (e.g. dgl) may be unavailable in lightweight envs.
    pass
