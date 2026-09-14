from io import BytesIO, StringIO

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


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

# -----------------------------------------------------
# ÍTEM 3: ESTADÍSTICAS DESCRIPTIVAS
# -----------------------------------------------------
with tab3:

    st.header("Ítem 3: Estadísticas descriptivas")

    st.write(
        """
        En este apartado se analizan las principales medidas
        estadísticas de las variables numéricas del dataset.
        Se revisan medidas de tendencia central y dispersión
        para comprender el comportamiento de los datos.
        """
    )

    # Intentamos convertir TotalCharges si viene como texto
    df_est = df.copy()

    if "TotalCharges" in df_est.columns:
        df_est["TotalCharges"] = pd.to_numeric(
            df_est["TotalCharges"],
            errors="coerce"
        )

    variables_numericas = [
        col for col in [
            "tenure",
            "MonthlyCharges",
            "TotalCharges"
        ]
        if col in df_est.columns
    ]

    if len(variables_numericas) == 0:

        st.warning(
            "No se identificaron variables numéricas para analizar."
        )

    else:

        st.subheader("Resumen estadístico (.describe())")

        st.dataframe(
            df_est[variables_numericas].describe(),
            use_container_width=True
        )

        st.subheader(
            "Medidas de tendencia central y dispersión"
        )

        resumen = []

        for columna in variables_numericas:

            resumen.append({
                "Variable": columna,
                "Media": round(
                    df_est[columna].mean(), 2
                ),
                "Mediana": round(
                    df_est[columna].median(), 2
                ),
                "Desv. estándar": round(
                    df_est[columna].std(), 2
                ),
                "Mínimo": round(
                    df_est[columna].min(), 2
                ),
                "Máximo": round(
                    df_est[columna].max(), 2
                )
            })

        st.dataframe(
            pd.DataFrame(resumen),
            hide_index=True,
            use_container_width=True
        )

        st.subheader("Interpretación básica")

        if "MonthlyCharges" in variables_numericas:

            media_monthly = (
                df_est["MonthlyCharges"].mean()
            )

            mediana_monthly = (
                df_est["MonthlyCharges"].median()
            )

            st.write(
                f"""
                • El cargo mensual promedio es
                **{media_monthly:.2f}**.

                • La mediana de cargos mensuales es
                **{mediana_monthly:.2f}**.

                • La diferencia entre media y mediana
                permite identificar posibles asimetrías
                en la distribución.
                """
            )

        if "tenure" in variables_numericas:

            tenure_promedio = (
                df_est["tenure"].mean()
            )

            st.write(
                f"""
                • La permanencia promedio de los clientes es
                de **{tenure_promedio:.2f} meses**.
                """
            )

        st.info(
            """
            La desviación estándar permite evaluar la
            variabilidad de los datos. Valores altos
            indican mayor dispersión entre clientes.
            """
        )

# -----------------------------------------------------
# ÍTEM 4: ANÁLISIS DE VALORES FALTANTES
# -----------------------------------------------------
with tab4:

    st.header("Ítem 4: Análisis de valores faltantes")

    st.write(
        """
        En este apartado se revisa la existencia de valores
        faltantes en el dataset. La identificación de datos
        incompletos es importante porque puede afectar los
        análisis posteriores.
        """
    )

    # Conteo de nulos por columna
    nulos = df.isnull().sum()

    tabla_nulos = pd.DataFrame({
        "Variable": nulos.index,
        "Valores faltantes": nulos.values,
        "Porcentaje (%)":
            (nulos.values / len(df) * 100).round(2)
    })

    st.subheader("Conteo de valores faltantes")

    st.dataframe(
        tabla_nulos,
        hide_index=True,
        use_container_width=True
    )

    total_nulos = int(nulos.sum())

    st.metric(
        "Total de valores faltantes",
        total_nulos
    )

    st.subheader(
        "Variables con valores faltantes"
    )

    nulos_filtrados = (
        tabla_nulos[
            tabla_nulos["Valores faltantes"] > 0
        ]
        .sort_values(
            "Valores faltantes",
            ascending=False
        )
    )

    if len(nulos_filtrados) > 0:

        st.dataframe(
            nulos_filtrados,
            hide_index=True,
            use_container_width=True
        )

        st.bar_chart(
            nulos_filtrados.set_index(
                "Variable"
            )["Valores faltantes"]
        )

    else:

        st.success(
            "No se detectaron valores nulos mediante isnull()."
        )

    # Caso especial de TotalCharges
    if "TotalCharges" in df.columns:

        totalcharges_nulos = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce"
        ).isnull().sum()

        st.subheader(
            "Revisión adicional de TotalCharges"
        )

        st.write(
            f"""
            Al intentar convertir la columna
            TotalCharges a formato numérico,
            se detectaron **{totalcharges_nulos}**
            registros no válidos.
            """
        )

    st.subheader("Interpretación")

    if total_nulos == 0:

        st.info(
            """
            El dataset no presenta valores faltantes
            identificados por Pandas. Sin embargo,
            pueden existir cadenas vacías o espacios
            que requieran una revisión adicional.
            """
        )

    else:

        st.warning(
            f"""
            Se identificaron {total_nulos} valores
            faltantes en el dataset. Antes de realizar
            análisis más avanzados sería recomendable
            evaluar estrategias de limpieza o imputación.
            """
        )

# -----------------------------------------------------
# ÍTEM 5: DISTRIBUCIÓN DE VARIABLES NUMÉRICAS
# -----------------------------------------------------
with tab5:

    st.header("Ítem 5: Distribución de variables numéricas")

    st.write(
        """
        Se analiza la distribución de las variables
        numéricas mediante histogramas para identificar
        concentración de valores, dispersión y posibles
        asimetrías.
        """
    )

    df_num = df.copy()

    if "TotalCharges" in df_num.columns:
        df_num["TotalCharges"] = pd.to_numeric(
            df_num["TotalCharges"],
            errors="coerce"
        )

    variables_numericas = [
        col for col in [
            "tenure",
            "MonthlyCharges",
            "TotalCharges"
        ]
        if col in df_num.columns
    ]

    for variable in variables_numericas:

        st.subheader(variable)

        fig, ax = plt.subplots(figsize=(8,4))

        sns.histplot(
            data=df_num,
            x=variable,
            bins=30,
            kde=True,
            color="steelblue",
            ax=ax
        )

        ax.set_title(
            f"Distribución de {variable}"
        )

        st.pyplot(fig)

    st.subheader("Interpretación")

    st.write(
        """
        Los histogramas permiten identificar la forma
        de distribución de cada variable, observar
        concentraciones de clientes y detectar posibles
        asimetrías o valores extremos.
        """
    )
