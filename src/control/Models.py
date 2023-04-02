from dataclasses import dataclass
import numpy as np


@dataclass
class CarParams:
    m: int = 550  # Mass [kg]
    Iz: int = 960  # Yaw moment of inertia [m*N*s^2]
    lf: int = 1  # Longitudinal distance from c.g. to front tires [m]
    lr: int = 1  # Longitudinal distance from c.g. to reartires [m]
    Cf: int = 19000  # Front tire cornering stiffness [N/rad]
    Cr: int = 33000  # Rear tire cornering stiffness [N/rad]
    accel_max: float = 2  # [m/s^s]
    accel_min: float = -1  # [m/s^s]
    steer_max: float = 0.26  # [rad]
    steer_min: float = -0.26  # [rad]
    vel_max: float = 30  # [m/s]
    vel_min: float = 0  # [m/s]


class Car:
    def __init__(self, params: CarParams):
        self.x = 0
        self.y = 0
        self.xdot = 0
        self.ydot = 0
        self.psi = 0
        self.psidot = 0
        self.params = params

    def set_state(self, state):
        """
        :param state: [x, y, xdot, ydot, psi, psidot]
        :return:
        """
        self.x = state[0]
        self.y = state[1]
        self.xdot = state[2]
        self.ydot = state[3]
        self.psi = state[4]
        self.psidot = state[5]

    def get_v(self):
        return np.linalg.norm((self.xdot, self.ydot), ord=2)

    def get_state(self):
        return np.array([self.x, self.y, self.xdot, self.ydot, self.psi, self.psidot])


class Model:
    def __init__(self, car, dt):
        self.car = car
        self.dt = dt

    def step(self, control):
        raise NotImplementedError


class KBModel(Model):
    def __init__(self, car, dt):
        super().__init__(car, dt)

    def get_state(self):
        return np.array([self.car.x, self.car.y, self.car.get_v(), self.car.psi])

    def step(self, control):
        """
        Kinematic bicycle model from A. Carvalho et. al. (2015), section 3.1.1.
        :control: [a, delta]
        :return: np.array [X, Y, v, psi]
        """
        # Assign convenience names for legibility.
        accel = control[0]
        steer = control[1]
        p = self.car.params
        c = self.car
        v_old = self.car.get_v()
        state = self.car.get_state()

        new_state = np.zeros(6)

        steer = np.clip(steer, self.car.params.steer_min, self.car.params.steer_max)
        v_old = np.clip(v_old, self.car.params.vel_min, self.car.params.vel_max)

        beta = np.arctan(np.tan(steer) * p.lr / (p.lf + p.lr))
        v = v_old + self.dt * accel

        new_state[0] = state[0] + self.dt * v * np.cos(beta + c.psi)
        new_state[1] = state[1] + self.dt * v * np.sin(beta + c.psi)
        new_state[2] = state[2] + self.dt * accel * np.cos(beta)
        new_state[3] = state[3] + self.dt * accel * np.sin(beta)
        new_state[4] = state[4] + self.dt * v * np.sin(beta) / p.lr
        new_state[5] = (new_state[4] - c.psi) / self.dt

        self.car.set_state(new_state)

        # X, Y, vel, yaw.
        return np.array([new_state[0], new_state[1], self.car.get_v(), new_state[4]])

    def lin_step(self, vel, yaw, steer):
        # state matrix
        matrix_a = np.zeros((4, 4))
        matrix_a[0, 0] = 1.0
        matrix_a[1, 1] = 1.0
        matrix_a[2, 2] = 1.0
        matrix_a[3, 3] = 1.0
        matrix_a[0, 2] = self.dt * np.cos(yaw)
        matrix_a[0, 3] = -self.dt * vel * np.sin(yaw)
        matrix_a[1, 2] = self.dt * np.sin(yaw)
        matrix_a[1, 3] = self.dt * vel * np.cos(yaw)
        matrix_a[3, 2] = \
            self.dt * np.tan(steer) / (self.car.params.lf + self.car.params.lr)

        # input matrix
        matrix_b = np.zeros((4, 2))
        matrix_b[2, 0] = self.dt
        matrix_b[3, 1] = self.dt * vel / \
                         ((self.car.params.lf + self.car.params.lr) * np.cos(steer) ** 2)

        # constant matrix
        matrix_c = np.zeros(4)
        matrix_c[0] = self.dt * vel * np.sin(yaw) * yaw
        matrix_c[1] = -self.dt * vel * np.cos(yaw) * yaw
        matrix_c[3] = - self.dt * vel * steer / \
                      ((self.car.params.lf + self.car.params.lr) * np.cos(steer) ** 2)

        return matrix_a, matrix_b, matrix_c
