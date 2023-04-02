"""
Adapted from Pylot source by Fangyu Wu, Edward Fang.
https://github.com/erdos-project/pylot/tree/a71ae927328388dc44acc784662bf32a99f273f0/pylot/control/mpc

Reference: "Model Predictive Control for Autonomous and Semiautonomous Vehicles"
by Gao (page 33)
"""
import copy
import numpy as np
import cvxpy
from cvxpy.expressions import constants
from dataclasses import dataclass
from .Models import Model, KBModel


@dataclass
class ControllerConfig:
    Q: np.array = np.diag([1.0, 1.0, 0.01, 0.01])  # Weight on reference deviations.
    R: np.array = np.diag([0.01, 0.10])            # Weight on control input.
    S: np.array = np.diag([0.01, 1.0])             # Weight on change in control input.
    prediction_horizon: int = 20
    tracking_horizon: int = 5                     # tracking_horizon < prediction_horizon


@dataclass
class Reference:
    states: np.array  # [X, Y, vel, psi] x prediction_horizon
    curvature: np.array  # [k] x prediction_horizon

    def __copy__(self):
        return Reference(copy.copy(self.states), copy.copy(self.curvature))


class MPC:
    def __init__(self,
                 reference: Reference,
                 model: KBModel,
                 config: ControllerConfig):
        self.reference = reference
        self.model = model
        self.config = config

        self.predicted_steer = np.zeros((self.config.tracking_horizon, 1))
        self.predicted_accel = np.zeros((self.config.tracking_horizon, 1))

        self.num_state = 4
        self.predicted_state = np.zeros((self.num_state, self.config.tracking_horizon + 1))

        self.set_imminent_ref()

    def set_imminent_ref(self):
        rel_x = [X - self.model.car.x for X in self.reference.states[:, 0]]
        rel_y = [Y - self.model.car.y for Y in self.reference.states[:, 1]]
        rel_coords = zip(rel_x, rel_y)
        dist = [np.linalg.norm(pair) for pair in rel_coords]

        start_idx = np.argmin(dist)
        end_idx = min(len(self.reference.states)-2, start_idx+self.config.tracking_horizon+1)

        self.reference.states = self.reference.states[start_idx:end_idx, :]
        self.reference.curvature = self.reference.curvature[start_idx:end_idx]

    def get_predicted_states(self):
        predicted_state = np.zeros((self.num_state, self.config.tracking_horizon + 1))
        predicted_state[:, 0] = self.model.get_state()
        self.predicted_steer = self.reference.curvature

        state = predicted_state[:, 0]
        for accel, steer, t in zip(self.predicted_accel,
                                   self.predicted_steer,
                                   range(1, self.config.tracking_horizon + 1)):
            state = self.model.step([accel, steer])
            predicted_state[:, t] = state
        return predicted_state

    def control(self):
        self.predicted_state = self.get_predicted_states()

        x = cvxpy.Variable((self.num_state, self.config.tracking_horizon+1))  # [X, Y, v, psi]
        u = cvxpy.Variable((2, self.config.tracking_horizon))
        cost = constants.Constant(0.0)
        constraints = []

        for t in range(self.config.tracking_horizon):
            cost += cvxpy.quad_form(u[:, t], self.config.R)

            if t != 0:
                cost += cvxpy.quad_form(self.reference.states[t, :] - x[:, t],
                                        self.config.Q)

            A, B, C = self.model.lin_step(self.predicted_state[2, t], self.predicted_state[3, t],
                                          self.predicted_steer[t])
            constraints += [x[:, t+1] == A @ x[:, t] + B @ u[:, t] + C]

            if t < self.config.tracking_horizon-1:
                cost += cvxpy.quad_form(u[:, t+1] - u[:, t],
                                        self.config.S)

        constraints += [x[:, 0] == self.model.get_state()]
        constraints += [x[2, :] <= self.model.car.params.vel_max]
        constraints += [x[2, :] >= self.model.car.params.vel_min]
        constraints += [u[0, :] <= self.model.car.params.accel_max]
        constraints += [u[0, :] >= self.model.car.params.accel_min]
        constraints += [u[1, :] <= self.model.car.params.steer_max]
        constraints += [u[1, :] >= self.model.car.params.steer_min]

        prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        res = prob.solve(solver=cvxpy.OSQP, verbose=False, warm_start=True)

        return u.value[0, 0], u.value[1, 0]
