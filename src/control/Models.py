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
    v_max: float = 30  # [m/s]
    v_min: float = 0  # [m/s]


class Car:
    def __init__(self,
                 x: float,
                 y: float,
                 xdot: float,
                 ydot: float,
                 psi: float,
                 psidot: float,
                 accel: float,
                 steer: float,
                 params: CarParams):
        self.x = x
        self.y = y
        self.xdot = xdot
        self.ydot = ydot
        self.psi = psi
        self.psidot = psidot
        self.accel = accel
        self.steer = steer
        self.params = params

    def set_state(self, state):
        """
        :param state: nparray [x, y, xdot, ydot, psi, psidot]
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
        return np.array(self.x, self.y, self.xdot, self.ydot, self.psi, self.psidot)

    def set_control(self, control):
        """
        :param control: nparray [accel, steer]
        :return:
        """
        self.accel = control[0]
        self.steer = control[1]

    def get_control(self):
        return np.array((self.accel, self.steer))


class Model:
    def __init__(self, car, dt):
        self.car = car
        self.dt = dt

    def step(self):
        raise NotImplementedError


class KBModel(Model):
    def __init__(self, car, dt):
        super().__init__(car, dt)

    def step(self):
        """
        Kinematic bicycle model from A. Carvalho et. al. (2015), section 3.1.1.
        :return:
        """
        # Assign convenience names for legibility.
        p = self.car.params
        c = self.car
        v_old = self.car.get_v()
        state = self.car.get_state()

        new_state = np.zeros(6)

        beta = np.arctan(np.tan(c.steer) * p.lr / (p.lf + p.lr))
        v = v_old + self.dt * c.accel

        new_state[0] = state[0] + self.dt * v * np.cos(beta + c.psi)
        new_state[1] = state[1] + self.dt * v * np.sin(beta + c.psi)
        new_state[2] = state[2] + self.dt * v * np.cos(beta)
        new_state[3] = state[3] + self.dt * v * np.sin(beta)
        new_state[4] = state[4] + self.dt * v * np.sin(beta) / p.lr
        new_state[5] = (new_state[4] - c.psi) / self.dt

        self.car.set_state(new_state)
