import numpy as np


def simple(world, lane_width, lane_marker_width, num_lanes, rb, cb):
    desired_lane = 1
    lp = 0.

    if world.ego.distanceTo(cb) < desired_lane * (lane_width + lane_marker_width) + 0.2:
        lp += 0.
    elif world.ego.distanceTo(rb) < (num_lanes - desired_lane - 1) * (lane_width + lane_marker_width) + 0.3:
        lp += 1.

    v = world.ego.center - cb.center
    v = np.mod(np.arctan2(v.y, v.x) + np.pi / 2, 2 * np.pi)
    if world.ego.heading < v:
        lp += 0.7
    else:
        lp += 0.

    if np.random.rand() < lp:
        world.ego.set_control(0.2, 0.1)
    else:
        world.ego.set_control(-0.1, 0.1)
