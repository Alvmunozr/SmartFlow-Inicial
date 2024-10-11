from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Cargar el modelo YOLO preentrenado con las clases adecuadas
model = YOLO('yolov8n.pt')

# Cargar el video desde la ruta especificada
video = cv2.VideoCapture('C:\\Users\\alvar\\Desktop\\SEMANA_I_RECONOCIENDOIMAGENES\\IMG_3513.MP4')

# Obtener las propiedades del video para el guardado
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))

# Crear el objeto VideoWriter para guardar el video de salida
output_path = 'C:\\Users\\alvar\\Desktop\\SEMANA_I_RECONOCIENDOIMAGENES\\output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para guardar el video en formato .mp4
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Inicializar variables de conteo y listas para gráficos
person_crossings = 0  # Contador de personas que cruzan la línea roja
car_crossings = 0     # Contador de autos que cruzan la línea roja
detected_person_ids = set()  # Almacenar los IDs de las personas que cruzaron la línea
detected_car_ids = set()     # Almacenar los IDs de los autos que cruzaron la línea

# Listas para almacenar los datos de los cruces por tiempo
time_intervals = []   # Guardar los intervalos de tiempo para el eje X
person_crossings_data = []  # Cantidad acumulada de personas que cruzaron la línea roja
car_crossings_data = []     # Cantidad acumulada de autos que cruzaron la línea roja

# Inicializar contador de fotogramas y tiempo
frame_count = 0
current_time = 0

# Procesar el video fotograma a fotograma
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    frame_count += 1

    # Procesar solo 1 de cada 32 fotogramas (ajusta según la carga computacional)
    if frame_count % 32 != 0:
        continue

    # Calcular el tiempo actual en segundos
    current_time = frame_count // fps

    # Obtener dimensiones del fotograma
    height, width, _ = frame.shape

    # Calcular el punto de división para la línea roja (en la mitad del video)
    division_line = height // 2  # Línea en el centro del video

    # Realizar la detección en el fotograma actual
    results = model(frame)

    # Revisar las detecciones y verificar si cruzan la línea
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])  # ID de la clase detectada

            # Convertir las coordenadas del tensor a un array de numpy
            xyxy = box.xyxy.cpu().numpy()[0]

            # Extraer coordenadas individuales del array de numpy
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

            # Verificar el cruce con la línea roja usando la coordenada inferior del cuadro delimitador
            mid_y = y2

            # Obtener un identificador único para cada objeto
            obj_id = box.id.item() if box.id is not None else None

            # Si es una persona y cruza la línea roja
            if class_id == 0 and mid_y >= division_line and obj_id not in detected_person_ids:
                person_crossings += 1
                if obj_id is not None:
                    detected_person_ids.add(obj_id)  # Agregar la persona detectada a la lista

            # Si es un auto y cruza la línea roja
            elif class_id == 2 and mid_y >= division_line and obj_id not in detected_car_ids:
                car_crossings += 1
                if obj_id is not None:
                    detected_car_ids.add(obj_id)  # Agregar el auto detectado a la lista

    # Almacenar los datos de cruces para el gráfico
    time_intervals.append(current_time)
    person_crossings_data.append(person_crossings)
    car_crossings_data.append(car_crossings)

    # Dibujar la línea roja en el centro del fotograma para indicar la división
    cv2.line(frame, (0, division_line), (width, division_line), (0, 0, 255), 4)  # Línea roja en el centro

    # Escribir el fotograma con las detecciones en el video de salida
    out.write(frame)

    # Mostrar el video en tiempo real con la línea roja
    cv2.imshow("Detección en Estacionamiento (Mitad del Video)", frame)

    # Salir si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar el video y el escritor de video
video.release()
out.release()
cv2.destroyAllWindows()

# Imprimir resultados totales

print(f"Total de Personas que cruzaron la Línea Roja: {person_crossings}")
print(f"Total de Autos que cruzaron la Línea Roja: {car_crossings}")
print(f"Video guardado en: {output_path}")

# Crear gráficos basados en los cruces de la línea roja
plt.figure(figsize=(12, 6))
plt.plot(time_intervals, person_crossings_data, label="Personas que cruzaron", color='green', marker='o')
plt.plot(time_intervals, car_crossings_data, label="Autos que cruzaron", color='red', marker='o')
plt.title("Cruces de Personas y Autos que Cruzaron la Línea Roja a lo Largo del Tiempo")
plt.xlabel("Tiempo (segundos)")
plt.ylabel("Cantidad de Cruces")
plt.legend()
plt.grid()
plt.show()

# Gráfico de barras del total de cruces
plt.figure(figsize=(8, 5))
plt.bar(["Personas", "Autos"], [person_crossings, car_crossings], color=['green', 'red'])
plt.title("Total de Personas y Autos que Cruzaron la Línea Roja")
plt.ylabel("Cantidad Total de Cruces")
plt.show()
