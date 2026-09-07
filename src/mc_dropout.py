try:
    from .ensemble import EnsembleTCN
except ImportError:
    from ensemble import EnsembleTCN


class MCDropoutTCN(EnsembleTCN):
    """EnsembleTCN evaluated by enabling its dropout layers at inference."""
    pass
