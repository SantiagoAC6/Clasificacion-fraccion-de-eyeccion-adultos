import streamlit as st
import tensorflow as tf
import numpy as np
import cv2 # Para procesar videos o imágenes

# Cargar el modelo entrenado
@st.cache_resource # Almacena en caché el modelo para que no se recargue en cada interacción
def load_my_model():
    # ¡Asegúrate de que esta ruta sea correcta para tu entorno de Codespaces!
    model_path = '/workspaces/Clasificacion-fraccion-de-eyeccion-adultos/best_echonet_model (1).keras'
    model = tf.keras.models.load_model(model_path)
    return model

model = load_my_model()

st.title("Aplicación de Predicción de Fracción de Eyección con IA")

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2 # Para procesar videos o imágenes
import os # Para manejar archivos temporales

# Parámetros Globales
# Estos valores deben ser los mismos que usaste en tu Google Colab
NUM_FRAMES = 30
IMG_SIZE = (128, 128) # Tu modelo se entrenó con frames de 128x128
FE_THRESHOLD = 50.0 # Umbral para la clasificación binaria (Normal/Anormal)

# Cargar el Modelo Entrenado
# Usamos st.cache_resource para cargar el modelo solo una vez cuando la app inicia,
# lo que mejora el rendimiento.
@st.cache_resource
def load_my_model():
    # Asegúrate de que esta ruta sea correcta para tu entorno de Codespaces
    # Basado en tu información, esta es la ruta:
    model_path = '/workspaces/Clasificacion-fraccion-de-eyeccion-adultos/best_echonet_model (1).keras'
    
    if not os.path.exists(model_path):
        st.error(f"Error: El archivo del modelo no se encontró en la ruta: {model_path}")
        st.stop() # Detiene la ejecución de la app si el modelo no está

    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        st.stop() # Detiene la ejecución de la app si hay un problema al cargar

model = load_my_model()

# Función de Preprocesamiento del Video ---
# Esta función debe replicar EXACTAMENTE la lógica de load_video de tu EchoNetVideoGenerator
# Se ajusta para recibir la ruta del archivo temporal y devolver el tensor listo para la predicción.
def preprocess_video_for_prediction(video_path, num_frames=NUM_FRAMES, img_size=IMG_SIZE):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames_in_video == 0:
        st.error(f"Error: El video '{video_path}' está vacío o corrupto.")
        cap.release()
        return None

    # Calcular los índices de los frames a extraer (distribución uniforme)
    indices = np.linspace(0, total_frames_in_video - 1, num_frames, dtype=int)
    current_frame_idx = 0
    frame_idx_to_extract_counter = 0 # Contador para los índices a extraer

    while frame_idx_to_extract_counter < num_frames:
        ret, frame = cap.read()
        if not ret:
            # Si no hay más frames y aún no hemos extraído suficientes, rellenar con ceros.
            # Esto maneja videos más cortos que NUM_FRAMES
            while frame_idx_to_extract_counter < num_frames:
                frames.append(np.zeros((img_size[0], img_size[1], 1), dtype=np.float32))
                frame_idx_to_extract_counter += 1
            break

        # Si el frame actual coincide con uno de los índices que queremos extraer
        if current_frame_idx == indices[frame_idx_to_extract_counter]:
            # Procesar el frame: redimensionar, convertir a gris, normalizar
            frame = cv2.resize(frame, img_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = frame / 255.0  # Normalización a [0, 1]
            frames.append(frame[..., np.newaxis]) # Añadir la dimensión del canal (para Conv2D)

            frame_idx_to_extract_counter += 1
            # Asegura que no nos salgamos del rango de `indices`
            if frame_idx_to_extract_counter >= len(indices):
                frame_idx_to_extract_counter = len(indices) - 1

        current_frame_idx += 1

    cap.release()
    
    # Asegurarse de que tenemos exactamente NUM_FRAMES
    if len(frames) != num_frames:
        st.warning(f"Advertencia: Se extrajeron {len(frames)} frames, se esperaban {num_frames}. Rellenando/Truncando.")
        if len(frames) < num_frames:
            # Rellenar con ceros si faltan frames
            while len(frames) < num_frames:
                frames.append(np.zeros((img_size[0], img_size[1], 1), dtype=np.float32))
        else:
            # Truncar si hay demasiados frames (no debería pasar con la lógica de linspace)
            frames = frames[:num_frames]

    # Convertir a numpy array y añadir la dimensión de batch
    # Forma esperada: (1, NUM_FRAMES, IMG_SIZE[0], IMG_SIZE[1], 1)
    return np.array(frames, dtype=np.float32).reshape(1, num_frames, img_size[0], img_size[1], 1)

# --- 4. Interfaz de Usuario de Streamlit ---
st.set_page_config(
    page_title="Predicción de Fracción de Eyección",
    page_icon="❤️",
    layout="centered"
)

st.title("🫀 Aplicación de Predicción de Fracción de Eyección (FE) con IA")
st.markdown("Sube un video de ecocardiograma (formato AVI) para obtener una predicción de la Fracción de Eyección y una clasificación de la función cardíaca (Normal/Anormal).")

st.info(f"El modelo espera {NUM_FRAMES} frames de {IMG_SIZE[0]}x{IMG_SIZE[1]} píxeles en escala de grises.")
st.write(f"Umbral de clasificación: Fracción de Eyección menor a **{FE_THRESHOLD}%** se considera **Anormal**.")

uploaded_file = st.file_uploader("Sube tu video de ecocardiograma (solo .avi)", type=["avi"])

if uploaded_file is not None:
    st.write("---")
    st.subheader("Video Subido")
    st.video(uploaded_file) # Muestra el video subido al usuario

    # Crear una barra de progreso
    progress_text = "Procesando video y haciendo predicción. Por favor, espera..."
    my_bar = st.progress(0, text=progress_text)

    # Guardar el archivo temporalmente para que OpenCV pueda leerlo
    temp_video_path = "uploaded_video_temp.avi"
    try:
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        my_bar.progress(25, text="Video guardado temporalmente.")

        # Preprocesar el video
        processed_video_tensor = preprocess_video_for_prediction(temp_video_path, NUM_FRAMES, IMG_SIZE)
        
        my_bar.progress(75, text="Video preprocesado. Realizando predicción...")

        if processed_video_tensor is not None:
            # Realizar la predicción
            prediction = model.predict(processed_video_tensor)[0][0] # Extraer el valor escalar
            
            my_bar.progress(100, text="Predicción completa!")
            st.write("---")
            st.subheader("Resultados de la Predicción")
            st.success(f"La Fracción de Eyección (FE) predicha es: **{prediction:.2f}%**")

            # Clasificación binaria basada en el umbral
            if prediction < FE_THRESHOLD:
                st.error(f"**Diagnóstico: Función Cardíaca Anormal** (FE predicha < {FE_THRESHOLD}%)")
                st.markdown("⚠️ _Se recomienda consultar a un especialista para una evaluación detallada._")
            else:
                st.success(f"**Diagnóstico: Función Cardíaca Normal** (FE predicha >= {FE_THRESHOLD}%)")
                st.markdown("✅ _El modelo sugiere una función cardíaca normal._")
        else:
            st.error("No se pudo procesar el video para la predicción. Por favor, verifica el archivo.")
            my_bar.empty() # Borra la barra de progreso en caso de error

    except Exception as e:
        st.error(f"Ocurrió un error inesperado durante el procesamiento o predicción: {e}")
        my_bar.empty() # Borra la barra de progreso en caso de error
    finally:
        # Limpiar el archivo temporal, sin importar si hubo error o no
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        my_bar.empty() # Asegurarse de que la barra de progreso desaparezca

st.markdown("---")
st.markdown("Creado con ❤️ y IA para la evaluación de ecocardiogramas.")

