import asyncio
import json
from time import sleep
import numpy as np
import math
from functools import reduce

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

max_speed = 8
min_speed = 4

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    # always returns a nonnegative angle.
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def normalize(vector, action=None):
    length = np.linalg.norm(vector)
    if length > max_speed or action == "maximize":
        return [element * max_speed / length for element in vector]
    elif length < min_speed or action == "minimize":
        return [element * min_speed / length for element in vector]
    return vector

def attraction_field(robot_center, target_center, obstacle_center_list):
    return normalize(np.subtract(target_center, robot_center), "maximize")

def repulsion_field(robot_center, target_center, obstacle_center_list):
    return (0, 0)

def creative_field(robot_center, target_center, obstacle_center_list):
    return (0, 0)

def get_vector(robot, target, obstacles):
    field_list = [attraction_field, repulsion_field, creative_field]
    return normalize(reduce(np.add, [field(robot["center"], target["center"], obstacles)
                                     for field in field_list]))

def closer_side(robot, target):
    distance = lambda corner: np.linalg.norm(np.subtract(target["center"], corner))
    return "left" if distance(robot["corners"][0]) < distance(robot["corners"][1]) else "right"

def get_command(robot, target, obstacles):
    if None in (robot, target):
        return (0, 0)

    vector = get_vector(robot, target, obstacles)
    angle = angle_between(robot["orientation"], vector)
    turn_direction = closer_side(robot, target)
    magnitude = np.linalg.norm(vector)

    if angle >= math.pi/3:
        # robot is facing away from the target
        if turn_direction == "right":
            return max_speed, -1 * max_speed
        return -1 * max_speed, max_speed

    # robot is facing roughly towards the target
    cos = math.cos(angle)
    inner_wheel_speed = max(cos * magnitude, min_speed)
    outer_wheel_speed = magnitude * 2 - inner_wheel_speed
    if turn_direction == "right":
        return outer_wheel_speed, inner_wheel_speed
    return inner_wheel_speed, outer_wheel_speed

def positions(data, target_num):
    robot = data.get("robot", None)
    target = data.get(target_num, None)
    obstacles = [data[key]["center"] for key in data if key not in ('time', 'robot', target_num)]
    return robot, target, obstacles

def main(host, port, target_num):
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
            data = json.loads(do('where'))
            if "robot" in data:
                break
            print("server returned bad response")
            sleep(0.1)

        return positions(data, target_num)

    while True:
        try:
            arg_list = map(int, get_command(*get_positions()))
            print("vector:", list(arg_list))
            #do("speed " + " ".join(str(arg) for arg in arg_list))
            #sleep(0.4)

            ## reset the motors
            #inverted = map(lambda x: -1 * x, arg_list)
            #do("speed " + " ".join(str(arg) for arg in inverted))
            #sleep(0.20)
        except json.decoder.JSONDecodeError:
            print("ERROR: asyncio concatenated the command")

        #do("power 0 0")
        sleep(1)
    
    writer.close()

def run_tests():
    #robot, target, obstacles = positions(example1, "21")
    ##print("vector:", get_vector(robot, target, obstacles))
    #assert need_to_turn(robot, get_vector(robot, target, obstacles)) == False
    ##print("passed test1")

    #robot, target, obstacles = positions(example2, "21")
    ##print("vector:", get_vector(robot, target, obstacles))
    #assert need_to_turn(robot, get_vector(robot, target, obstacles)) == True
    ##print("passed test2")

    robot, target, obstacles = positions(example3, "21")
    assert closer_side(robot, target) == "left"

    robot, target, obstacles = positions(example4, "21")
    assert closer_side(robot, target) == "right"

    for i, ex in enumerate([example1, example2, example3, example4]):
        print("get_command for example{}: {}".format(i + 1,
            get_command(*positions(ex, "21"))))
    print("^^Make sure this looks good^^")

    print("All tests pass")

if __name__ == '__main__':
    from sys import argv
    run_tests()
    main(*argv[1:])
