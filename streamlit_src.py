import streamlit as st
import json
import leafmap.foliumap as leafmap

st.set_page_config(layout="wide")
st.title("Mapa de Barrios de Buenos Aires (compatible con Playground)")

# Sliders
a = st.slider("Valor A", 0, 100, 30)
b = st.slider("Valor B", 0, 50, 10)

# File uploader
uploaded = st.file_uploader("Cargar GeoJSON (.geojson o .txt)", type=["geojson", "txt"])

if not uploaded:
    st.warning("Subí el archivo con el GeoJSON.")
    st.stop()

# Leer archivo
content = uploaded.read().decode("utf-8")
geojson_data = json.loads(content)

# Crear mapa
m = leafmap.Map(center=[-34.61, -58.44], zoom=11)

# Cargar capa con estilos
m.add_geojson(
    geojson_data,
    layer_name="Barrios CABA",
    style_function=lambda feature: {
        "fillColor": "#3186cc",
        "color": "black",
        "weight": 1,
        "fillOpacity": 0.4,
    },
    highlight_function=lambda feature: {
        "fillColor": "#0d47a1",
        "color": "black",
        "weight": 2,
        "fillOpacity": 0.8,
    }
)

# Mostrar mapa
m.to_streamlit(height=600)
