from controller import Robot

# Inicializar el robot
robot = Robot()

# Obtener el paso de tiempo base de la simulación
timestep = int(robot.getBasicTimeStep())

# --- Ruedas ---
left_wheel = robot.getDevice('left wheel motor')
right_wheel = robot.getDevice('right wheel motor')

# Poner las ruedas en modo velocidad (posición infinita)
left_wheel.setPosition(float('inf'))
right_wheel.setPosition(float('inf'))

# --- Sensores de proximidad del E-puck (ps0 a ps7) ---
proximity_sensors = []
for i in range(8):
    sensor = robot.getDevice('ps{}'.format(i))
    sensor.enable(timestep)
    proximity_sensors.append(sensor)

# --- Parámetros de comportamiento ---
max_speed = 3.0            # velocidad de avance (m/s)
turn_speed = 2.0           # velocidad de giro sobre su eje (m/s)
obstacle_threshold = 80.0  # valor de proximidad que indica obstáculo

# Distribución de los sensores frontales
LEFT_FRONT = [0, 1]              # ps0, ps1  (frente-izquierda)
RIGHT_FRONT = [6, 7]             # ps6, ps7  (frente-derecha)
FRONT = LEFT_FRONT + RIGHT_FRONT # ps0, ps1, ps6, ps7 (frente)

# --- Bucle principal de la simulación ---
while robot.step(timestep) != -1:
    # Leer todos los sensores de proximidad (valor 0 = libre, 100 = tocado)
    values = [sensor.getValue() for sensor in proximity_sensors]

    # ¿Hay una pared u obstáculo al frente? (ps0, ps1, ps6 o ps7 > 80.0)
    if max(values[i] for i in FRONT) > obstacle_threshold:
        # Girar sobre su eje hacia el lado más libre (el de menor lectura)
        left_avg = sum(values[i] for i in LEFT_FRONT) / len(LEFT_FRONT)
        right_avg = sum(values[i] for i in RIGHT_FRONT) / len(RIGHT_FRONT)

        if left_avg <= right_avg:
            # Lado izquierdo más libre -> girar a la izquierda
            left_wheel.setVelocity(-turn_speed)
            right_wheel.setVelocity(turn_speed)
        else:
            # Lado derecho más libre -> girar a la derecha
            left_wheel.setVelocity(turn_speed)
            right_wheel.setVelocity(-turn_speed)
    else:
        # Camino despejado -> avanzar
        left_wheel.setVelocity(max_speed)
        right_wheel.setVelocity(max_speed)
