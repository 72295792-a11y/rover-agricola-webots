import math
from controller import Robot

robot = Robot()
timestep = int(robot.getBasicTimeStep())
motor = robot.getDevice("slide_motor")
motor.setVelocity(0.55)

while robot.step(timestep) != -1:
    t = robot.getTime()
    motor.setPosition(1.0 * math.sin(0.5 * t))
