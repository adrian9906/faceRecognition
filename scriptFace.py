import cv2
import face_recognition
import sqlite3
import numpy as np
import pickle
import os
from datetime import datetime
import time
import sys
from deepface import DeepFace
# --- CONFIGURACIÓN ---
DB_NAME = "rostros.db"
FOTO_DIR = "fotos_conocidas"
UMBRAL_SIMILITUD = 0.5
FOTOS_POR_PERSONA = 50  # Número de fotos a capturar
TIEMPO_ENTRE_FOTOS = 0.2  # Segundos entre cada captura

# Crear directorio para fotos si no existe
if not os.path.exists(FOTO_DIR):
    os.makedirs(FOTO_DIR)

# --- BASE DE DATOS ---
def iniciar_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS usuarios
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  nombre TEXT, 
                  encoding BLOB, 
                  fecha_registro TEXT,
                  foto_path TEXT)''')  # Añadimos ruta de la foto
    conn.commit()
    conn.close()

def guardar_persona(nombre, encodings_list, frames_list):
    """Guarda múltiples fotos y crea un encoding promedio"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    if not encodings_list:
        print("[!] No hay encodings para guardar")
        return None
    
    # 1. Calcular encoding promedio (más robusto)
    encoding_promedio = np.mean(encodings_list, axis=0)
    
    # 2. Guardar cada foto en disco
    foto_paths = []
    for i, frame in enumerate(frames_list):
        nombre_archivo = f"{nombre}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i+1}.jpg"
        ruta_foto = os.path.join(FOTO_DIR, nombre_archivo)
        cv2.imwrite(ruta_foto, frame)
        foto_paths.append(ruta_foto)
    
    # 3. Guardar en DB (usamos pickle para serializar el encoding promedio)
    encoding_blob = pickle.dumps(encoding_promedio)
    rutas_texto = ";".join(foto_paths)  # Guardamos todas las rutas separadas por ;
    
    c.execute("INSERT INTO usuarios (nombre, encoding, fecha_registro, foto_path) VALUES (?, ?, ?, ?)",
              (nombre, encoding_blob, datetime.now(), rutas_texto))
    
    conn.commit()
    conn.close()
    print(f"[+] Persona '{nombre}' registrada con {len(frames_list)} fotos")
    return encoding_promedio

def obtener_personas_conocidas():
    """Devuelve una lista de encodings y nombres desde la DB"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT nombre, encoding FROM usuarios")
    data = c.fetchall()
    conn.close()
    
    known_names = []
    known_encodings = []
    
    for row in data:
        known_names.append(row[0])
        known_encodings.append(pickle.loads(row[1]))
        
    return known_names, known_encodings
def calcular_ear(puntos_ojo):
    """Calcula el Eye Aspect Ratio para detectar parpadeo (Prueba de vida)"""
    A = np.linalg.norm(np.array(puntos_ojo[1]) - np.array(puntos_ojo[5]))
    B = np.linalg.norm(np.array(puntos_ojo[2]) - np.array(puntos_ojo[4]))
    C = np.linalg.norm(np.array(puntos_ojo[0]) - np.array(puntos_ojo[3]))
    return (A + B) / (2.0 * C)

def modo_analisis_demografico():
    """OPCIÓN 4: Analiza Edad, Etnia y Emoción (Usa DeepFace)"""
    cap = cv2.VideoCapture(0)
    print("\n[INFO] Cargando modelos de IA... (esto puede tardar unos segundos)")
    print("[INFO] Presiona 'q' para salir del análisis demográfico.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        try:
            # Analizamos cada frame (enforce_detection=False evita que el script se detenga si no hay nadie)
            # silent=True evita llenar la consola de logs
            resultados = DeepFace.analyze(frame, actions=['age', 'race', 'emotion'], enforce_detection=False, silent=True)
            res = resultados[0]
            
            # Extraer info
            edad = int(res['age'])
            etnia = res['dominant_race']
            emocion = res['dominant_emotion']
            region = res['region']

            # Dibujar recuadro y etiquetas
            cv2.rectangle(frame, (region['x'], region['y']), 
                          (region['x']+region['w'], region['y']+region['h']), (255, 0, 0), 2)
            
            texto = f"Etnia: {etnia} | Edad: {edad} | Emocion: {emocion}"
            cv2.putText(frame, texto, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        except Exception as e:
            cv2.putText(frame, "Buscando rostro...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Analisis Demografico', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    cap.release()
    cv2.destroyAllWindows()

def main_avanzado_1N_liveness():
    """OPCIÓN 5: 1:N + Landmarks + Prueba de vida (Blinking)"""
    nombres_conocidos, encodings_conocidos = obtener_personas_conocidas()
    cap = cv2.VideoCapture(0)
    
    print(f"\n[INFO] Iniciando Reconocimiento Avanzado con {len(nombres_conocidos)} usuarios.")
    print("[INFO] Presiona 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Reducimos para mejorar FPS
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detectar ubicaciones, landmarks y encodings
        face_locations = face_recognition.face_locations(rgb_small)
        face_landmarks_list = face_recognition.face_landmarks(rgb_small, face_locations)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        for encoding, location, landmarks in zip(face_encodings, face_locations, face_landmarks_list):
            # 1. BÚSQUEDA 1:N (Identidad)
            name = "Desconocido"
            color = (0, 0, 255)
            if encodings_conocidos:
                distances = face_recognition.face_distance(encodings_conocidos, encoding)
                best_match = np.argmin(distances)
                if distances[best_match] < UMBRAL_SIMILITUD:
                    name = nombres_conocidos[best_match]
                    conf = (1 - distances[best_match]) * 100
                    name = f"{name} ({conf:.0f}%)"
                    color = (0, 255, 0)

            # 2. LIVENESS (Parpadeo)
            ear_izq = calcular_ear(landmarks['left_eye'])
            ear_der = calcular_ear(landmarks['right_eye'])
            ear_promedio = (ear_izq + ear_der) / 2.0
            status_vida = "VIVO" if ear_promedio > 0.20 else "PARPADEO"
            
            # 3. DIBUJAR LANDMARKS (Puntos faciales)
            top, right, bottom, left = [coord * 2 for coord in location] # Escalamos de vuelta
            for feature in landmarks.keys():
                for punto in landmarks[feature]:
                    # Escalamos los puntos x2 porque procesamos en small_frame
                    cv2.circle(frame, (punto[0]*2, punto[1]*2), 1, (255, 255, 255), -1)

            # Dibujar etiquetas
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, f"{name} | {status_vida}", (left, top - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow('Reconocimiento Avanzado (1:N + Liveness)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

# --- BLOQUE PRINCIPAL DEL PROGRAMA ---


def registrar_persona_con_multifotos():
    """Registra una persona capturando múltiples fotos en video"""
    cap = cv2.VideoCapture(0)
    
    print("\n" + "="*50)
    print("REGISTRO CON MÚLTIPLES FOTOS")
    print("="*50)
    iniciar_db()
    nombre = input("Ingresa el nombre de la persona (ej: Adrian): ").strip()
    
    if not nombre:
        print("[!] Nombre no válido")
        cap.release()
        return None
    
    print(f"\n[INFO] Preparándose para registrar a: {nombre}")
    print("[INFO] Muévete ligeramente y cambia expresión para mejores resultados")
    print("[INFO] Presiona 'ESPACIO' para comenzar la captura")
    print("[INFO] Presiona 'q' para cancelar")
    
    capturando = False
    fotos_capturadas = 0
    encodings_list = []
    frames_list = []
    ultima_captura = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: 
            break
        
        frame_copy = frame.copy()
        
        # Detectar cara en tiempo real
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        
        if len(face_locations) > 0:
            # Dibujar rectángulo alrededor de la cara
            top, right, bottom, left = face_locations[0]
            cv2.rectangle(frame_copy, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Si estamos capturando, mostrar contador
            if capturando:
                tiempo_actual = time.time()
                
                # Capturar foto cada TIEMPO_ENTRE_FOTOS segundos
                if tiempo_actual - ultima_captura > TIEMPO_ENTRE_FOTOS and fotos_capturadas < FOTOS_POR_PERSONA:
                    # Extraer encoding de la cara
                    face_encoding = face_recognition.face_encodings(rgb_frame, [face_locations[0]])[0]
                    
                    # Guardar frame y encoding
                    frames_list.append(frame.copy())
                    encodings_list.append(face_encoding)
                    fotos_capturadas += 1
                    ultima_captura = tiempo_actual
                    
                    print(f"[CAPTURA] Foto {fotos_capturadas}/{FOTOS_POR_PERSONA} tomada")
                    
                    # Destacar la foto capturada con flash visual
                    flash_frame = frame.copy()
                    flash_frame[:, :, :] = 255  # Blanco
                    cv2.imshow('Registro Multi-Foto', flash_frame)
                    cv2.waitKey(50)
                
                # Mostrar progreso en pantalla
                cv2.putText(frame_copy, f"CAPTURANDO: {fotos_capturadas}/{FOTOS_POR_PERSONA}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame_copy, "Muevete ligeramente...", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                
                # Dibujar barra de progreso
                progreso = int((fotos_capturadas / FOTOS_POR_PERSONA) * 100)
                cv2.rectangle(frame_copy, (50, 100), (50 + progreso * 3, 130), (0, progreso * 2.55, 255 - progreso * 2.55), -1)
                cv2.putText(frame_copy, f"{progreso}%", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Detener cuando tengamos todas las fotos
                if fotos_capturadas >= FOTOS_POR_PERSONA:
                    capturando = False
                    print(f"[✓] ¡Captura completada! {FOTOS_POR_PERSONA} fotos tomadas")
            
            else:
                cv2.putText(frame_copy, f"Registrar: {nombre}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame_copy, "Presiona ESPACIO para comenzar", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            cv2.putText(frame_copy, "NO SE DETECTA CARA", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame_copy, "Ajusta tu posicion", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Registro Multi-Foto', frame_copy)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' ') and not capturando and len(face_locations) > 0:
            # Iniciar captura
            capturando = True
            fotos_capturadas = 0
            frames_list = []
            encodings_list = []
            ultima_captura = time.time()
            print(f"[INICIANDO] Capturando {FOTOS_POR_PERSONA} fotos...")
        
        elif key == ord('q'):
            print("[!] Registro cancelado")
            break
        
        # Si completamos la captura, esperar confirmación
        if fotos_capturadas >= FOTOS_POR_PERSONA:
            cv2.putText(frame_copy, "¡CAPTURA COMPLETADA!", (10, 160), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            cv2.putText(frame_copy, "Presiona 's' para guardar, 'r' para repetir", (10, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            key_final = cv2.waitKey(0) & 0xFF
            if key_final == ord('s'):
                # Guardar persona
                encoding_promedio = guardar_persona(nombre, encodings_list, frames_list)
                break
            elif key_final == ord('r'):
                # Repetir captura
                capturando = False
                print("[↻] Repitiendo captura...")
            elif key_final == ord('q'):
                print("[!] Registro cancelado")
                break
    
    cap.release()
    cv2.destroyAllWindows()
    return encodings_list[0] if encodings_list else None

def identificar_persona_1N_foto_unica():
    """
    Toma una sola captura de la cámara y busca quién es en toda la base de datos.
    """
    # 1. Cargar datos de la DB
    nombres_conocidos, encodings_conocidos = obtener_personas_conocidas()
    
    if not encodings_conocidos:
        print("[!] La base de datos está vacía. Registra a alguien primero.")
        return

    cap = cv2.VideoCapture(0)
    print("\n[INFO] Buscando rostro frente a la cámara...")
    print("[INFO] Presiona 'ESPACIO' para capturar y buscar, o 'q' para cancelar.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Mostrar guía en pantalla
        cv2.putText(frame, "Posa para la busqueda 1:N", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, "Presiona ESPACIO para identificar", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow('Busqueda 1:N - Captura Única', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '): # PRESIONAR ESPACIO PARA IDENTIFICAR
            # Convertir a RGB para face_recognition
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locations = face_recognition.face_locations(rgb_frame)
            
            if len(locations) == 0:
                print("[!] No se detectó ningún rostro. Intenta de nuevo.")
                continue
            
            # Obtener encoding de la foto capturada
            encoding_sonda = face_recognition.face_encodings(rgb_frame, locations)[0]

            # --- ALGORITMO 1:N ---
            # Comparamos la distancia de esta foto contra TODA la lista de la DB
            distancias = face_recognition.face_distance(encodings_conocidos, encoding_sonda)
            indice_mejor_match = np.argmin(distancias) # El que tenga la distancia más pequeña
            distancia_minima = distancias[indice_mejor_match]

            if distancia_minima < UMBRAL_SIMILITUD:
                nombre_encontrado = nombres_conocidos[indice_mejor_match]
                confianza = (1 - distancia_minima) * 100
                print(f"\n[✓] PERSONA IDENTIFICADA: {nombre_encontrado}")
                print(f"[i] Nivel de confianza: {confianza:.2f}%")
                
                # Mostrar resultado en imagen estática
                top, right, bottom, left = locations[0]
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, f"IDENTIFICADO: {nombre_encontrado}", (left, top - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow('Busqueda 1:N - Captura Única', frame)
                cv2.waitKey(3000) # Mostrar resultado 3 segundos
                break
            else:
                print("\n[X] PERSONA NO ENCONTRADA en la base de datos.")
                cv2.putText(frame, "DESCONOCIDO", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Busqueda 1:N - Captura Única', frame)
                cv2.waitKey(2000)
                break

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
def main():
    iniciar_db()
    
    # Cargar base de datos
    nombres_conocidos, encodings_conocidos = obtener_personas_conocidas()
    
    cap = cv2.VideoCapture(0)
    
    print("\n" + "="*50)
    print("SISTEMA DE RECONOCIMIENTO FACIAL")
    print("="*50)
    print("Comandos:")
    print("  'r' - Registrar nueva persona")
    print("  'q' - Salir")
    print(f"  Personas registradas: {len(nombres_conocidos)}")
    print("="*50)

    while True:
        ret, frame = cap.read()
        if not ret: break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detectar caras
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            name = "Desconocido"
            
            if encodings_conocidos:
                face_distances = face_recognition.face_distance(encodings_conocidos, face_encoding)
                best_match_index = np.argmin(face_distances)
                
                if face_distances[best_match_index] < UMBRAL_SIMILITUD:
                    name = nombres_conocidos[best_match_index]
                    confidence = (1 - face_distances[best_match_index]) * 100
                    name = f"{name} ({confidence:.1f}%)"
                else:
                    name = "Desconocido"
            
            # Dibujar resultado
            top, right, bottom, left = face_location
            top *= 4; right *= 4; bottom *= 4; left *= 4
            
            color = (0, 255, 0) if "Desconocido" not in name else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Mostrar información en pantalla
        cv2.putText(frame, f"Personas registradas: {len(nombres_conocidos)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Presiona 'r' para registrar nueva persona", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow('Reconocimiento Facial', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('r'):
            # Registrar nueva persona con múltiples fotos
            nuevo_encoding = registrar_persona_con_multifotos()
            if nuevo_encoding:
                # Actualizar listas en memoria
                nuevo_nombre = input("¿Nombre para esta persona?: ").strip()
                if nuevo_nombre:
                    nombres_conocidos.append(nuevo_nombre)
                    encodings_conocidos.append(nuevo_encoding)
                    print(f"[✓] {nuevo_nombre} añadido a memoria")
        
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
def analizar_foto_por_ruta():
    """
    Carga una imagen desde una ruta, identifica si está en la DB
    y realiza el análisis demográfico.
    """
    path = input("\n[RUTA] Ingresa la ruta de la imagen (ej: fotos/persona.jpg): ").strip()
    
    # 1. Verificar si el archivo existe
    if not os.path.exists(path):
        print("[!] Error: El archivo no existe en esa ruta.")
        return

    # 2. Cargar la imagen
    frame = cv2.imread(path)
    if frame is None:
        print("[!] Error: No se pudo leer la imagen.")
        return

    # 3. Preparar datos de búsqueda 1:N
    nombres_conocidos, encodings_conocidos = obtener_personas_conocidas()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Localizar caras
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    if not face_encodings:
        print("[!] No se detectaron rostros en la imagen proporcionada.")
        return

    print("\n[PROCESANDO] Analizando imagen y buscando en la base de datos...")

    # 4. Procesar cada cara encontrada en la foto
    for i, (encoding, location) in enumerate(zip(face_encodings, face_locations)):
        # --- Búsqueda 1:N ---
        nombre_identificado = "Desconocido"
        confianza = 0
        
        if encodings_conocidos:
            distancias = face_recognition.face_distance(encodings_conocidos, encoding)
            indice_mejor = np.argmin(distancias)
            if distancias[indice_mejor] < UMBRAL_SIMILITUD:
                nombre_identificado = nombres_conocidos[indice_mejor]
                confianza = (1 - distancias[indice_mejor]) * 100

        # --- Análisis Demográfico (DeepFace) ---
        try:
            # Recortar la cara para DeepFace (mejora precisión)
            top, right, bottom, left = location
            face_crop = frame[top:bottom, left:right]
            
            demog = DeepFace.analyze(face_crop, actions=['age', 'race', 'emotion'], silent=True)[0]
            
            edad = demog['age']
            etnia = demog['dominant_race']
            emocion = demog['dominant_emotion']
            
            # Imprimir resultados en consola
            print(f"\n--- RESULTADOS CARA {i+1} ---")
            print(f"ID: {nombre_identificado} ({confianza:.1f}% match)")
            print(f"Edad estimada: {int(edad)} años")
            print(f"Etnia dominante: {etnia}")
            print(f"Estado de ánimo: {emocion}")

            # Dibujar en la imagen
            color = (0, 255, 0) if nombre_identificado != "Desconocido" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, f"{nombre_identificado}", (left, top - 40), 1, 1.5, color, 2)
            cv2.putText(frame, f"{int(edad)}a | {etnia} | {emocion}", (left, top - 10), 1, 1, (255, 255, 255), 1)

        except Exception as e:
            print(f"[!] No se pudo realizar el análisis demográfico detallado: {e}")

    # 5. Mostrar la imagen analizada
    cv2.imshow('Analisis de Archivo', frame)
    print("\n[INFO] Cerrando ventana de imagen en 5 segundos...")
    cv2.waitKey(5000) 
    cv2.destroyAllWindows()
# --- MENÚ PRINCIPAL ---
if __name__ == "__main__":
    while True:
        print("\n" + "═"*45)
        print("      SISTEMA DE IDENTIDAD BIOMETRICA")
        print("═"*45)
        print("1. Iniciar reconocimiento en tiempo real (Base)")
        print("2. Registrar nueva persona (Multifoto)")
        print("3. Ver lista de personas registradas")
        print("4. Analisis Demografico (Edad, Etnia, Emocion)")
        print("5. Reconocimiento Avanzado (1:N + Liveness + Puntos)")
        print("6. Búsqueda 1:N")
        print("7. Búsqueda por foto")
        print("8. Salir")
        print("═"*45)
        
        opcion = input("\nSelecciona una opción (1-6): ")
        
        if opcion == "1":
            main() # Tu función original
            
        elif opcion == "2":
            registrar_persona_con_multifotos() # Tu función original
            print("\n[INFO] Reiniciando sistema para cargar nuevos datos...")
            os.execv(sys.executable, ['python'] + sys.argv)
            
        elif opcion == "3":
            nombres, _ = obtener_personas_conocidas()
            print(f"\n--- Usuarios Registrados ({len(nombres)}) ---")
            for i, n in enumerate(nombres, 1): print(f"  {i}. {n}")
            input("\nPresiona Enter para continuar...")
            
        elif opcion == "4":
            modo_analisis_demografico()
            
        elif opcion == "5":
            main_avanzado_1N_liveness()
            
        elif opcion == "6":
            identificar_persona_1N_foto_unica()
        
        elif opcion== '7':
            analizar_foto_por_ruta()
        elif opcion== '8':
            print("Saliendo del sistema..."); break
        
        else:
            print("[!] Opción no válida, intenta de nuevo.")