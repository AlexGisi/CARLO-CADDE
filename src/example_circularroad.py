import numpy as np
from world import World
from agents import Car, RingBuilding, CircleBuilding, Painting, Pedestrian
from geometry import Point
from sensors import GPS
import time
from tkinter import *
import control.simple

human_controller = False

dt = 0.1  # time steps in terms of seconds. In other words, 1/dt is the FPS.
world_width = 120  # in meters
world_height = 120
inner_building_radius = 30
num_lanes = 2
lane_marker_width = 0.5
num_of_lane_markers = 50
lane_width = 3.5

# The world is 120 meters by 120 meters. ppm is the pixels per meter.
w = World(dt, width=world_width, height=world_height, ppm=6)

# Let's add some sidewalks and RectangleBuildings.
# A Painting object is a rectangle that the vehicles cannot collide with. So we use them for the sidewalks /
# zebra crossings / or creating lanes.
# A CircleBuilding or RingBuilding object is also static -- they do not move. But as opposed to Painting, they can be
# collided with.

# To create a circular road, we will add a CircleBuilding and then a RingBuilding around it
cb = CircleBuilding(Point(world_width / 2, world_height / 2), inner_building_radius, 'gray80')
w.add(cb)
rb = RingBuilding(Point(world_width / 2, world_height / 2),
                  inner_building_radius + num_lanes * lane_width + (num_lanes - 1) * lane_marker_width,
                  1 + np.sqrt((world_width / 2) ** 2 + (world_height / 2) ** 2), 'gray80')
w.add(rb)

"""
# Let's also add some lane markers on the ground. This is just decorative. Because, why not.
for lane_no in range(num_lanes - 1):
    lane_markers_radius = inner_building_radius + (lane_no + 1) * lane_width + (lane_no + 0.5) * lane_marker_width
    lane_marker_height = np.sqrt(2 * (lane_markers_radius ** 2) * (1 - np.cos(
        (2 * np.pi) / (2 * num_of_lane_markers))))  # approximate the circle with a polygon and then use cosine theorem
    for theta in np.arange(0, 2 * np.pi, 2 * np.pi / num_of_lane_markers):
        dx = lane_markers_radius * np.cos(theta)
        dy = lane_markers_radius * np.sin(theta)
        w.add(Painting(Point(world_width / 2 + dx, world_height / 2 + dy), Point(lane_marker_width, lane_marker_height),
                       'white', heading=theta))
"""
num_traj_points = 100
traj_radius = inner_building_radius + 1 * lane_width + 0.5 * lane_marker_width
traj = list()  # X [m], Y [m], psi [rad], v [m/s]
for theta in np.arange(0, 2 * np.pi, 2 * np.pi / num_traj_points):
    dx = traj_radius * np.cos(theta)
    dy = traj_radius * np.sin(theta)
    traj.append((world_width / 2 + dx, world_height / 2 + dy, theta, 30))
    w.add(Painting(Point(world_width / 2 + dx, world_height / 2 + dy), Point(0.4, 0.4),
                   'white', heading=0))

# A Car object is a dynamic object -- it can move. We construct it using its center location and heading angle.
c1 = Car(Point(91.75, 60), np.pi / 2, is_ego=True, gps=GPS(w))
c1.max_speed = 30.0  # let's say the maximum is 30 m/s (108 km/h)
c1.velocity = Point(0, 3.0)

w.add(c1)
w.set_ego(c1)

w.render()  # This visualizes the world we just constructed.

if not human_controller:
    # Let's implement some simple policy for the car c1
    for k in range(600):
        control.simple(w, lane_width, lane_marker_width,
                       num_lanes, rb, cb)

        w.advance(sleep=dt / 4)

        if w.collision_exists():  # We can check if there is any collision at all.
            print('Collision exists somewhere...')
    w.close()

else:  # Let's use the keyboard input for human control
    from interactive_controllers import KeyboardController

    c1.set_control(0., 0.)  # Initially, the car will have 0 steering and 0 throttle.
    controller = KeyboardController(w)
    for k in range(60000):
        c1.set_control(controller.steering, controller.throttle)
        w.tick()  # This ticks the world for one time step (dt second)
        w.render()
        time.sleep(dt / 4)  # Let's watch it 4x
        if w.collision_exists():
            import sys

            sys.exit(0)
    w.close()
