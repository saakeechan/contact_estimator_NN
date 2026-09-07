try:
    from .ucb import UCBTCN
except ImportError:
    from ucb import UCBTCN


class VCLTCN(UCBTCN):
    def __init__(self, *args, vcl_prior_sigma=1., **kwargs):
        if vcl_prior_sigma <= 0: raise ValueError('vcl_prior_sigma must be positive.')
        super().__init__(*args, initial_prior_sigma=vcl_prior_sigma, **kwargs)
