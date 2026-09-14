from io import BytesIO

import pandas as pd
import streamlit as st


# =========================================================
# CONFIGURACIÓN
# =========================================================
st.set_page_config(
    page_title="Telco Customer Churn",
    page_icon="📡",
    layout="wide"
)

AUTOR = "Fidel Napoleón Bringas Salazar"
CURSO = "Especialización en Python for Analytics"
ANIO = 2026


# =========================================================
# CLASE PARA TRABAJAR CON EL DATASET
# Ampliaremos sus métodos al desarrollar los ítems del EDA.
# =========================================================
class AnalizadorTelco:

    def __init__(self, dataframe):
        self.df = dataframe.copy()

    def obtener_dimensiones(self):
        return self.df.shape

    def obtener_vista_previa(self, cantidad=5):
        return self.df.head(cantidad)


# =========================================================
# LECTURA Y VALIDACIÓN INICIAL
# =========================================================
def cargar_dataset(contenido):
    dataframe = None

    for codificacion in ("utf-8-sig", "cp1252"):
        try:
            dataframe = pd.read_csv(
                BytesIO(contenido),
                sep=None,
                engine="python",
                encoding=codificacion
            )
            break

        except UnicodeDecodeError:
            continue

    if dataframe is None:
        raise ValueError(
            "No se pudo interpretar la codificación del archivo."
        )

    if dataframe.empty:
        raise ValueError(
            "El archivo no contiene registros."
        )

    if dataframe.shape[1] < 2:
        raise ValueError(
            "Solo se reconoció una columna. "
            "Comprueba el separador y el contenido del CSV."
        )

    # Validación inicial de estructura.
    # Revisaremos las columnas específicas al cargar el archivo.
    return dataframe


# =========================================================
# MENÚ Y CARGADOR PERSISTENTE
# =========================================================
st.sidebar.title("📡 Menú principal")

opcion = st.sidebar.selectbox(
    "Seleccione un módulo:",
    [
        "Home",
        "Carga del dataset",
        "Análisis EDA"
    ]
)

# El cargador permanece visible al cambiar de módulo.
archivo = st.sidebar.file_uploader(
    "Cargar TelcoCustomerChurn.csv",
    type=["csv"],
    key="archivo_telco"
)

if archivo is None:
    st.session_state.pop("dataset_telco", None)
    st.session_state.pop("contenido_telco", None)
    st.session_state.pop("error_telco", None)

else:
    contenido = archivo.getvalue()

    # Solo volvemos a leer cuando cambia el archivo.
    if contenido != st.session_state.get("contenido_telco"):
        st.session_state.pop("dataset_telco", None)
        st.session_state.pop("error_telco", None)

        st.session_state["contenido_telco"] = contenido

        try:
            st.session_state["dataset_telco"] = cargar_dataset(
                contenido
            )

        except Exception as error:
            st.session_state["error_telco"] = str(error)

if "dataset_telco" in st.session_state:
    st.sidebar.success("CSV leído correctamente.")

elif "error_telco" in st.session_state:
    st.sidebar.error("No se pudo cargar el CSV.")

st.sidebar.divider()
st.sidebar.write(f"**Autor:** {AUTOR}")
st.sidebar.write(f"**Año:** {ANIO}")


# =========================================================
# MÓDULO 1: HOME
# =========================================================
if opcion == "Home":

    st.title("📡 Telco Customer Churn")
    st.subheader("Análisis Exploratorio de Datos")

    st.write(
        """
        Aplicación interactiva desarrollada con Python y Streamlit
        para explorar el dataset TelcoCustomerChurn.csv.
        """
    )

    st.info(
        """
        **Objetivo:** analizar, limpiar, transformar y visualizar
        los datos para identificar patrones asociados a la fuga
        de clientes y apoyar la comprensión de los factores
        relacionados con su permanencia.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Datos del autor")
        st.write(f"**Nombre:** {AUTOR}")
        st.write(f"**Curso / Especialización:** {CURSO}")
        st.write(f"**Año:** {ANIO}")

    with col2:
        st.subheader("Tecnologías del proyecto")
        st.write(
            """
            - Python y Streamlit
            - Pandas y NumPy
            - Matplotlib y Seaborn
            - Programación Orientada a Objetos
            - Estadística descriptiva
            """
        )

    st.subheader("Descripción del dataset")

    st.write(
        """
        El dataset Telco Customer Churn reúne información
        sobre clientes de una empresa de telecomunicaciones
        y su condición de permanencia o abandono del servicio.

        A partir del archivo se explorarán sus características,
        los servicios contratados y las variables disponibles
        para comprender patrones asociados a la fuga de clientes.
        """
    )

    st.subheader("Alcance del proyecto")

    st.write(
        """
        El proyecto comprende la revisión de calidad de datos,
        limpieza, transformación, estadísticas descriptivas
        y visualizaciones interactivas.
        """
    )

    st.warning(
        "Este proyecto no construye modelos predictivos de churn."
    )

    st.write(
        "Para comenzar, carga el CSV en la barra lateral "
        "y selecciona 'Carga del dataset'."
    )


# =========================================================
# MÓDULO 2: CARGA DEL DATASET
# =========================================================
elif opcion == "Carga del dataset":

    st.title("📂 Carga del dataset")

    st.write(
        "Selecciona TelcoCustomerChurn.csv mediante "
        "el cargador de la barra lateral."
    )

    if "error_telco" in st.session_state:
        st.error(st.session_state["error_telco"])

    if "dataset_telco" not in st.session_state:
        st.warning(
            "Debes cargar un CSV válido para continuar."
        )
        st.stop()

    df = st.session_state["dataset_telco"]
    analizador = AnalizadorTelco(df)

    filas, columnas = analizador.obtener_dimensiones()

    st.success(
        "El CSV fue leído correctamente y contiene "
        "filas y columnas para explorar."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Número de filas", f"{filas:,}")

    with col2:
        st.metric("Número de columnas", columnas)

    st.subheader("Vista previa: primeras cinco filas")

    st.dataframe(
        analizador.obtener_vista_previa(),
        use_container_width=True
    )

    st.subheader("Nombres de las columnas")
    st.write(df.columns.tolist())

    st.caption(
        "Esta vista muestra el resultado de la lectura inicial. "
        "Todavía no se han aplicado limpieza, imputación "
        "ni eliminación de registros."
    )


# =========================================================
# MÓDULO 3: ANÁLISIS EDA
# =========================================================
elif opcion == "Análisis EDA":

    st.title("📊 Análisis Exploratorio de Datos")

    if "dataset_telco" not in st.session_state:
        st.warning(
            "Primero carga un CSV válido en la barra lateral."
        )

        if "error_telco" in st.session_state:
            st.error(st.session_state["error_telco"])

        st.stop()

    df = st.session_state["dataset_telco"]

    st.success(
        f"Dataset disponible: {len(df):,} filas "
        f"y {df.shape[1]} columnas."
    )

    st.info(
        "Aquí desarrollaremos los diez ítems del caso, "
        "uno por uno, siguiendo las instrucciones."
    )
