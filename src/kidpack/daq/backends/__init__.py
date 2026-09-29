"""Backend factories. Driver modules are imported only when actually used."""


def create_scope(cfg):
    if cfg.backend == 'niscope':
        from kidpack.daq.backends.niscope_backend import NiScopeBackend
        return NiScopeBackend(cfg)
    from kidpack.daq.backends.simulator import SimulatedScope
    return SimulatedScope(cfg)


def create_sg(cfg):
    """Return the SG backend, or None when the DAQ is not to control the SG."""
    if cfg.sg is None:
        return None
    if cfg.backend == 'niscope':
        from kidpack.daq.backends.nirfsg_backend import NiRfsgBackend
        return NiRfsgBackend(cfg.sg.resource)
    from kidpack.daq.backends.simulator import SimulatedSg
    return SimulatedSg()
