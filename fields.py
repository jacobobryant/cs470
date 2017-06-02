import asyncio
import json
from time import sleep
import numpy as np
import math
from functools import reduce

debug = False

# robot is pointed towards the target
example1 = {"time": 2189617.79221862, "21": {"corners": [[991.0, 478.0], [1009.0, 573.0], [912.0, 591.0], [894.0, 497.0]],
                                             "orientation": [0.9822942018508911, -0.1873447746038437], "center": [951.5, 534.75]},
        "robot": {"corners": [[973.0, 263.0], [867.0, 288.0], [842.0, 180.0], [949.0, 156.0]], "orientation": [0.22220909595489502, 0.9749990701675415], "center": [907.75, 221.75]}}

# robot is pointed away from the target
example2 = {"time": 2189691.664889238, "21": {"corners": [[990.0, 478.0], [1009.0, 573.0], [913.0, 592.0], [894.0, 496.0]],
                                              "orientation": [0.9819334149360657, -0.18922674655914307], "center": [951.5, 534.75]},
        "robot": {"corners": [[902.0, 143.0], [1005.0, 182.0], [966.0, 285.0], [863.0, 245.0]], "orientation": [0.35561609268188477, -0.9346320629119873], "center": [934.0, 213.75]}}

# target is on left side of robot
example3 = {"robot": {"corners": [[787.0, 208.0], [756.0, 107.0], [859.0, 75.0], [891.0, 177.0]], "center": [823.25, 141.75], "orientation": [-0.9566738605499268, 0.29116159677505493]}, "21": {"corners": [[932.0, 390.0], [967.0, 478.0], [876.0, 514.0], [842.0, 425.0]], "center": [904.25, 451.75], "orientation": [0.9309388995170593, -0.36517491936683655]}, "time": 2248780.235014935}

# target is on right side of robot
example4 = {"robot": {"corners": [[898.0, 68.0], [882.0, 173.0], [775.0, 157.0], [792.0, 52.0]], "center": [836.75, 112.5], "orientation": [0.98890221118927, 0.14856746792793274]}, "21": {"corners": [[932.0, 390.0], [967.0, 478.0], [877.0, 514.0], [841.0, 426.0]], "center": [904.25, 452.0], "orientation": [0.9291830658912659, -0.36961978673934937]}, "time": 2248804.190855203}

example5 = {"time": 17880.044430377, "robot": {"corners": [[1251.0, 314.0], [1174.0, 250.0], [1239.0, 177.0], [1317.0, 244.0]], "orientation": [-0.6754910945892334, 0.737368106842041], "center": [1245.25, 246.25]}, "29": {"corners": [[830.0, 417.0], [735.0, 410.0], [743.0, 318.0], [837.0, 323.0]], "orientation": [-0.08038419485092163, 0.9967640042304993], "center": [786.25, 367.0]}, "25": {"corners": [[400.0, 377.0], [377.0, 291.0], [466.0, 262.0], [490.0, 348.0]], "orientation": [-0.9513070583343506, 0.3082447350025177], "center": [433.25, 319.5]}}


#   0 1 2 3 4 5 6 7 8 9
# 0
# 1
# 2       * *
# 3         * *
# 4 S         *       E
# 5           *
# 6         * *
# 7       * *
# 8
# 9
#
# EXAMPLE OUTPUT
# path: 3,1 2,2 1,3 1,4 1,5 1,6 2,7 3,8 4,9
grid_example1 = {"grid": set.difference({(x,y) for x in range(10)
                                               for y in range(10)},
                                        {(2,3), (2,4), (3,4), (3,5), (4,5),
                                         (5,5), (6,4), (6,5), (7,3), (7,4)}),
        "start": (4,0), "end": (4,9)}

# The minimum speed for an individual wheel.
min_speed = 3

## The maximum angle at which the robot will move both wheels forward
## instead of turning in place.
#max_angle = math.pi*(1/2)

# The maximum proportion of the outer wheel speed to the inner wheel
# speed when turning.
max_proportion = 2.5

def grid_coordinates(cam_coordinates, cell_length):
    return tuple(int(x / cell_length) for x in cam_coordinates)

def cam_coordinates(grid_coordinates, cell_length):
    return tuple(int(x * cell_length + cell_length/2)
                 for x in grid_coordinates)

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    # always returns a nonnegative angle.
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def normalize(vector, length):
    magnitude = np.linalg.norm(vector)
    return (vector * length) / magnitude

def radius(square):
    return np.linalg.norm(np.subtract(square["center"],
        square["corners"][0])) * 2

def attraction_field(robot, target):
    vector = np.subtract(target["center"], robot["center"])
    r = radius(target)
    spread = r * 3
    distance = np.linalg.norm(vector)
    if distance > spread:
        return normalize(vector, 1)
    elif distance < r:
        return (0, 0)
    return normalize(vector, 100)

def closer_side(robot, target, front_side="front"):
    offset = 0 if front_side == "front" else 2
    distance = lambda corner: np.linalg.norm(np.subtract(
        target["center"], corner))
    return "left" if distance(robot["corners"][0+offset]) < distance(
            robot["corners"][1+offset]) else "right"

def get_command(robot, vector):
    angle = angle_between(robot["orientation"], vector)
    front_side = "front"
    if angle > math.pi/2:
        front_side = "back"
        angle = math.pi - angle
    turn_direction = closer_side(robot, {"center":
            np.add(robot["center"], vector * 50)},
            front_side)

    if debug:
        print("vector:", vector)
        print("orientation:", robot["orientation"])
        print("angle:", angle)

    #if angle >= math.pi/2:
    #    # robot is facing away from the target
    #    if turn_direction == "right":
    #        return min_speed, -1 * min_speed
    #    return -1 * min_speed, min_speed

    # robot is facing roughly towards the target
    sign = 1 if front_side == "front" else -1
    proportion = 1 + (max_proportion - 1) * angle / (math.pi/2)

    inner_wheel_speed = min_speed * sign
    outer_wheel_speed = min_speed * proportion * sign
    command = [inner_wheel_speed, outer_wheel_speed]

    if turn_direction == "right":
        command.reverse()
    if front_side == "back":
        command.reverse()

    return command

def positions(data, target_num):
    robot = data.get("robot", None)
    target = data.get(target_num, None)
    obstacles = tuple(data[key] for key in data
                      if key not in ('time', 'robot', target_num))
    return robot, target, obstacles

# This function handles all the steps in the project description up until the
# a*/rrt part.
def get_grid(obstacles):
    cell_length = max(np.linalg.norm(np.subtract(*o["corners"][:2]))
                      for o in obstacles)
    corners = tuple(corner for o in obstacles for corner in o["corners"])
    occupied = {grid_coordinates(corner, cell_length) for corner in corners}
    return occupied, cell_length

def get_path(robot, target, obstacles, algorithm="astar"):
    # Get the path to follow.
    # grid will be a set of coordinates for cells that don't have an obstacle.
    # cell_length is the pixel width of each cell in the grid.
    # path will be a list of grid coordinates.
    grid, cell_length = get_grid(obstacles)
    path = (astar if algorithm == "astar" else rrt)(grid,
            grid_coordinates(robot["center"], cell_length),
            grid_coordinates(target["center"], cell_length))
    return path

def astar(grid, start, end):
    # TODO implement
    # See grid_example1 for example input, output
    path = []
    return path

def rrt(grid, start, end):
    # TODO implement
    # See grid_example1 for example input, output
    path = []
    return path

def on_target(robot, target):
    vector = np.subtract(target, robot["center"])
    r = radius(robot)
    distance = np.linalg.norm(vector)
    return distance < r

def main(host, port, target_num, algorithm="astar"):
    loop = asyncio.get_event_loop()
    reader, writer = loop.run_until_complete(
        asyncio.open_connection(host, port))
    print(reader.readline())

    def do(command):
        print('>>>', command)
        writer.write(command.strip().encode())
        res = loop.run_until_complete(reader.readline()).decode().strip()
        print('<<<', res)
        print()
        return res

    def get_positions():
        while True:
            try:
                data = json.loads(do('where'))
                if "robot" in data:
                    return positions(data, target_num)
            except json.decoder.JSONDecodeError:
                pass
            print("server returned bad response")
            sleep(0.1)

    robot, target, obstacles = get_positions()
    path = get_path(robot, target, obstacles, algorithm)
    while path:
        target = cam_coordinates(path[0], cell_length)
        vector = attraction_field(robot, target)
        arg_list = map(lambda x: int(round(x)), get_command(robot, vector))

        if debug:
            print("command:", list(arg_list))
            print()
            input("press Enter")
        else:
            do("speed " + " ".join(str(arg) for arg in arg_list))
            sleep(0.1)

        robot, _, _ = get_positions()
        if on_target(robot, target):
            path.pop(0)
    
    do("power 0 0")
    writer.close()

def run_tests():
    #robot, target, obstacles = positions(example3, "21")
    #assert closer_side(robot, target) == "left"

    #robot, target, obstacles = positions(example4, "21")
    #assert closer_side(robot, target) == "right"

    #for i, ex in enumerate([example1, example2, example3, example4]):
    #    print("get_command for example{}: {}".format(i + 1,
    #        get_command(*positions(ex, "21"))))
    #print("^^Make sure this looks good^^")
    args = tuple(grid_example1[k] for k in ["grid", "start", "end"])
    print("a* with grid_example1:", astar(*args))
    print("rrt with grid_example1:", rrt(*args))
    print("(should be something like "
            "[(3,1), (2,2), (1,3), (1,4), (1,5), (1,6), (2,7), (3,8), (4,9)]")
    print()
    robot, target, obstacles = positions(example5, "25")
    print("grid with example5:", get_grid(obstacles))
    print("a* with example5:", get_path(robot, target, obstacles, "astar"))
    print("rrt with example5:", get_path(robot, target, obstacles, "rrt"))
    
if __name__ == '__main__':
    from sys import argv
    run_tests()
    #main(*argv[1:])
