class Sensor:
    def __init__(self, world):
        self.world = world

    def poll(self):
        raise NotImplementedError


class GPS(Sensor):
    def __init__(self, world):
        super().__init__(world)

    def poll(self):
        return self.world.ego.center.x, self.world.ego.center.y
