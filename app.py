from io import BytesIO, StringIO

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

    def obtener_info(self):
        buffer = StringIO()
        self.df.info(buf=buffer, verbose=True, show_counts=True)
        return buffer.getvalue()

    def resumen_estructura(self):
        nulos = self.df.isna().sum()

        return pd.DataFrame({
            "Variable": self.df.columns,
            "Tipo de dato": self.df.dtypes.astype(str).values,
            "Valores no nulos": self.df.notna().sum().values,
            "Valores nulos": nulos.values,
            "Nulos (%)": (
                nulos.values / len(self.df) * 100
            ).round(2)
        })

    def clasificar_variables(self):
        """Clasifica las variables según su significado en Telco."""
        numericas = {"tenure", "MonthlyCharges", "TotalCharges"}
        registros = []

        for columna in self.df.columns:
            tipo_actual = str(self.df[columna].dtype)

            if columna == "customerID":
                clasificacion = "Identificador"
                observacion = "Identifica al cliente; no es una medida."

            elif columna in numericas:
                clasificacion = "Numérica"

                if pd.api.types.is_numeric_dtype(self.df[columna]):
                    observacion = "Almacenada como número."
                else:
                    observacion = (
                        "Representa una cantidad, pero está almacenada "
                        "como texto. Requiere revisión y conversión."
                    )

            elif columna == "SeniorCitizen":
                clasificacion = "Categórica"
                observacion = (
                    "Indicador binario: 0 y 1 representan categorías."
                )

            elif pd.api.types.is_numeric_dtype(self.df[columna]):
                clasificacion = "Numérica"
                observacion = "Clasificación basada en el tipo de dato."

            else:
                clasificacion = "Categórica"
                observacion = "Representa categorías o etiquetas."

            registros.append({
                "Variable": columna,
                "Tipo actual en Pandas": tipo_actual,
                "Clasificación": clasificacion,
                "Observación": observacion
            })

        return pd.DataFrame(registros)

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
    analizador_eda = AnalizadorTelco(df)

    st.success(
        f"Dataset disponible: {len(df):,} filas "
        f"y {df.shape[1]} columnas."
    )

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "1. Información general",
    "2. Clasificación de variables",
    "3. Estadísticas descriptivas",
    "4. Valores faltantes",
    "5. Variables numéricas",
    "6. Variables categóricas",
    "7. Numérico vs categórico",
    "8. Categórico vs categórico",
    "9. Análisis dinámico",
    "10. Hallazgos"
])

    # -----------------------------------------------------
    # ÍTEM 1: INFORMACIÓN GENERAL DEL DATASET
    # -----------------------------------------------------
    with tab1:

        st.header("Ítem 1: Información general del dataset")

        st.write(
            """
            Revisamos la estructura inicial del archivo, los tipos
            de datos reconocidos por Pandas y la cantidad de valores
            nulos. Esta revisión permite identificar qué aspectos
            requieren limpieza antes de continuar con el análisis.
            """
        )

        filas, columnas = analizador_eda.obtener_dimensiones()
        total_nulos = int(df.isna().sum().sum())

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Número de filas", f"{filas:,}")

        with col2:
            st.metric("Número de columnas", columnas)

        with col3:
            st.metric("Celdas nulas", f"{total_nulos:,}")

        st.subheader("Información obtenida con .info()")

        st.code(
            analizador_eda.obtener_info(),
            language="text"
        )

        st.subheader("Tipos de datos y conteo de nulos")

        st.dataframe(
            analizador_eda.resumen_estructura(),
            hide_index=True,
            use_container_width=True
        )

        st.subheader("Interpretación inicial")

        st.write(
            f"El archivo contiene {filas:,} registros y "
            f"{columnas} variables. Pandas reconoce "
            f"{total_nulos:,} celdas como valores nulos."
        )

        if total_nulos == 0:
            st.info(
                "No se detectaron nulos mediante isna(). "
                "Sin embargo, las cadenas vacías o formadas "
                "por espacios pueden requerir una revisión adicional."
            )

        else:
            variables_con_nulos = int(
                (df.isna().sum() > 0).sum()
            )

            st.warning(
                f"Hay {variables_con_nulos} variables con nulos. "
                "Revisaremos su tratamiento en el ítem de "
                "valores faltantes."
            )

        if "TotalCharges" in df.columns:
            st.write(
                f"**Tipo actual de TotalCharges:** "
                f"`{df['TotalCharges'].dtype}`."
            )

            st.caption(
                "Revisaremos si sus valores pueden convertirse "
                "a números cuando desarrollemos la limpieza. "
                "En este ítem todavía no se modifica esa columna."
            )

    with tab2:
        st.header("Ítem 2: Clasificación de variables")

        st.write(
            "Clasificamos las variables según su significado en el "
            "dataset. El tipo almacenado en Pandas puede diferir "
            "de su función en el análisis."
        )

        clasificacion = analizador_eda.clasificar_variables()

        conteos = (
            clasificacion["Clasificación"]
            .value_counts()
            .reindex(
                ["Numérica", "Categórica", "Identificador"],
                fill_value=0
            )
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Variables numéricas", int(conteos["Numérica"]))

        with col2:
            st.metric("Variables categóricas", int(conteos["Categórica"]))

        with col3:
            st.metric("Identificadores", int(conteos["Identificador"]))

        st.subheader("Clasificación detallada")

        st.dataframe(
            clasificacion,
            hide_index=True,
            use_container_width=True
        )

        st.subheader("Conteo por clasificación")
        st.bar_chart(conteos.rename("Cantidad"))

        st.subheader("Interpretación")

        st.write(
            "**Numéricas:** tenure representa la antigüedad del "
            "cliente; MonthlyCharges, el cargo mensual; y "
            "TotalCharges, los cargos acumulados."
        )

        st.write(
            "**Categóricas:** describen características del cliente, "
            "servicios contratados, condiciones de contratación "
            "y abandono del servicio. SeniorCitizen pertenece "
            "a este grupo aunque esté codificada con números."
        )

        st.write(
            "**Identificador:** customerID se utiliza para identificar "
            "clientes y revisar posibles duplicados."
        )

        st.info(
            "TotalCharges se clasifica como numérica por su significado. "
            "Su conversión desde texto se realizará durante la limpieza, "
            "comprobando los valores que no puedan convertirse."
        )

        st.caption(
            "Esta clasificación no modifica los datos originales."
        )
