import numpy as np
from scipy.optimize import minimize

class MPC:
    def __init__(self, ocp, opt_object, opt_options, n_horizon, use_opt_method=True):
        self.n_horizon = n_horizon
        self.ocp = ocp
        self.z_init = None
        self.opt_object = opt_object
        self.opt_method = self.opt_object.optimize if use_opt_method else None
        self.opt_options = opt_options

    def optimize_horizon(self):
        res_opt = minimize(self.ocp.horizon_cost, self.z_init,
                           method=self.opt_method,
                           constraints=self.ocp.constraints,  # bounds=self.ocp.bounds,
                           options=self.opt_options,
                           tol=self.opt_options['xtol'])
        return res_opt
