# workoutdetector-

El programa implementa un detector de posturas utilizando la biblioteca Mediapipe. Permite detectar y contar el número de repeticiones realizadas en diferentes ejercicios para los brazos y las piernas.

El programa utiliza las siguientes bibliotecas:
- cv2: Para capturar y procesar el video de la cámara.
- pandas: Para manejar los datos en formato de tabla.
- mediapipe: Para realizar la detección de las posturas.
- numpy: Para realizar cálculos numéricos.

El programa consta de las siguientes partes principales:

1. Definición de variables y configuraciones iniciales:
   - `counter_left`, `counter_right`, `counter_leg`: Contadores para el número de repeticiones realizadas para el brazo izquierdo, brazo derecho y pierna, respectivamente.
   - `stage_left`, `stage_right`, `stage_leg`: Variables para controlar el estado actual de la postura (levantando o bajando el brazo o la pierna).
   - `session_data`: Lista para almacenar los datos de la sesión de entrenamiento.
   - `calculate_angle()`: Función para calcular el ángulo entre tres puntos.

2. Configuración de la cámara:
   - Se inicializa la captura de video desde la cámara.

3. Bucle principal:
   - Dentro del bucle principal, se lee cada fotograma del video capturado por la cámara.
   - Se realiza la detección de posturas utilizando Mediapipe en cada fotograma.
   - Se extraen los landmarks (puntos clave) de las posturas detectadas.
   - Se calculan los ángulos para los brazos y las piernas utilizando la función `calculate_angle()`.
   - Se actualizan los contadores de repeticiones y los estados de postura según los ángulos calculados.
   - Se muestra el video en una ventana y se superponen los landmarks y la información de repeticiones y clase en el video.

4. Finalización del programa:
   - Cuando se presiona la tecla 'q', se cierran todas las ventanas y se libera la captura de video.

