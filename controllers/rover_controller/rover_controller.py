# -*- coding: utf-8 -*-
"""Punto de entrada del controlador modular del Rover Agricola 4x4."""
from controller import Robot
from mision import MisionAgricola


def main():
    robot = Robot()
    mision = MisionAgricola(robot)
    mision.ejecutar()


if __name__ == "__main__":
    main()
