# mi_streamlit_1.py Este incorpora el modelo predictivo de pickle.1
# -----------------------------------------------
# App interactiva con Streamlit para predecir especies de pingüinos
# Usa el modelo entrenado guardado en archivos .pickle
# -----------------------------------------------

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# ---------------- Configuración inicial ----------------
st.set_page_config(layout='centered', page_title='Talento Tech Innovador', page_icon=':penguin:')

t1, t2 = st.columns([0.3, 0.7])
t1.image('./index.jpg', width=180)
t2.title('Mi primer tablero')
t2.markdown('**Tel:** 123 **| Email:** talentotech@gmail.com')

# ---------------- Secciones (tabs) ----------------
steps = st.tabs(['Introducción', 'Visualización de datos', 'Modelo ML', '$\int_{-\infty}^\infty e^{\sigma\mu}dt$'])

# ---------------- TAB 1: Introducción ----------------
with steps[0]:
    st.title('📘 Metadata')
    st.write('Bienvenido a mi proyecto de clasificación de pingüinos 🐧')

    # Cargar datos base
    df = pd.read_csv('penguins.csv')

    st.dataframe(df.head(), use_container_width=True)
    st.info(f"El dataset contiene **{df.shape[0]} filas** y **{df.shape[1]} columnas**.")

    # Mostrar columnas de forma más limpia
    st.markdown("### 🧩 Columnas disponibles:")
    for col in df.columns:
        st.markdown(f"- **{col}**")

# ---------------- TAB 2: Visualización ----------------
with steps[1]:
    st.markdown('### 📊 Gráfica de los tipos de Pingüinos')
    df = pd.read_csv('penguins.csv')

    species = st.selectbox('Escoja la especie a visualizar', df['species'].unique())
    x = st.selectbox('Selecciona la variable X', list(df.columns))
    y = st.selectbox('Selecciona la variable Y', list(df.columns))
    
    fig, ax = plt.subplots()
    sns.scatterplot(data=df[df['species'] == species], x=x, y=y, ax=ax)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    st.pyplot(fig)

# ---------------- TAB 3: Modelo ML ----------------
with steps[2]:
    st.markdown('### 🔮 Predicción de especie')

    # Cargar el modelo y los metadatos
    rfc = pickle.load(open('random_forest_penguin.pickle', 'rb'))
    unique_penguin_mapping = pickle.load(open('output_penguin.pickle', 'rb'))
    model_columns = pickle.load(open('model_columns_penguin.pickle', 'rb'))

    # Entradas del usuario
    island = st.selectbox('Isla', ['Biscoe', 'Dream', 'Torgersen'])
    sex = st.selectbox('Sexo', ['MALE', 'FEMALE'])
    bill_length = st.number_input('Longitud del pico (mm)', min_value=0.0)
    bill_depth = st.number_input('Profundidad del pico (mm)', min_value=0.0)
    flipper_length = st.number_input('Longitud de aleta (mm)', min_value=0.0)
    body_mass = st.number_input('Masa corporal (g)', min_value=0.0)

    # Mostrar datos ingresados en formato legible
    st.markdown("#### 🧾 Datos ingresados:")
    st.markdown(f"""
    - **Isla:** {island}  
    - **Sexo:** {sex}  
    - **Longitud del pico:** {bill_length} mm  
    - **Profundidad del pico:** {bill_depth} mm  
    - **Longitud de aleta:** {flipper_length} mm  
    - **Masa corporal:** {body_mass} g
    """)

    # Crear DataFrame de entrada y codificar igual que el modelo
    input_df = pd.DataFrame({
        'bill_length_mm': [bill_length],
        'bill_depth_mm': [bill_depth],
        'flipper_length_mm': [flipper_length],
        'body_mass_g': [body_mass],
        'island': [island],
        'sex': [sex]
    })

    # One-hot encoding
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    # Predicción
    if st.button('Predecir especie 🧠'):
        prediction = rfc.predict(input_encoded)
        prediction_species = unique_penguin_mapping[prediction[0]]
        st.success(f'✅ La especie del pingüino es: **{prediction_species}**')

# ---------------- TAB 4: Ecuación decorativa ----------------
with steps[3]:
    st.latex(r"\int_{-\infty}^{\infty} e^{\sigma\mu} \, dt")
    st.info("Esta pestaña es decorativa para mostrar fórmulas en Streamlit 😄")
