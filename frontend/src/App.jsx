import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, GeoJSON } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import './App.css';

//Cargamos datos de Geolocalización
const GEOJSON_URL = "/barrios.geojson"; 

function App() {
  const [geoData, setGeoData] = useState(null);
  
  const [barrioSeleccionado, setBarrioSeleccionado] = useState(null);
  const [comunaSeleccionada, setComunaSeleccionada] = useState(null); 

  const [inputs, setInputs] = useState({ dia: 'Lunes', hora: '20:00' });
  const [prediccion, setPrediccion] = useState(null);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);

  // Dictionario para mapear días a números
  const diasMap = {
    'Lunes': 0, 'Martes': 1, 'Miercoles': 2, 'Jueves': 3, 
    'Viernes': 4, 'Sabado': 5, 'Domingo': 6
  };

  useEffect(() => {
    fetch(GEOJSON_URL)
      .then(res => res.json())
      .then(data => setGeoData(data))
      .catch(err => console.error("Error cargando mapa:", err));
  }, []);

  const getBarrioName = (feature) => {
    if (!feature?.properties) return "";
    return feature.properties.barrio || feature.properties.BARRIO || feature.properties.nombre || "";
  };

  const styleBarrio = (feature) => {
    const nombre = getBarrioName(feature);
    const esSeleccionado = barrioSeleccionado === nombre;

    let fillColor = '#83b7eaff'; 
    let fillOpacity = 0.3;
    let weight = 1;
    let color = '#ffffff'; 

    if (esSeleccionado) {
      fillColor = '#2196F3'; 
      fillOpacity = 0.6;
      weight = 3;
    }
    return { fillColor, fillOpacity, weight, color, opacity: 1 };
  };

  const onEachFeature = (feature, layer) => {
    const nombre = getBarrioName(feature);
    layer.bindTooltip(nombre);

    layer.on({
      click: (e) => {
        setBarrioSeleccionado(nombre);
        
        setComunaSeleccionada(feature.properties.COMUNA); 

        setPrediccion(null); 
        e.originalEvent.stopPropagation();
      },
      mouseover: (e) => {
        const layer = e.target;
        layer.setStyle({ fillOpacity: 0.7 });
      },
      mouseout: (e) => {
        const layer = e.target;
        layer.setStyle({ fillOpacity: barrioSeleccionado === nombre ? 0.6 : 0.3 });
      }
    });
  };

  const handlePredict = async () => {
    setLoading(true);
    setErrorMsg(null);
    const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8800';

    // Preparamos datos para la API
    const diaNumero = diasMap[inputs.dia];
    const horaSolo = parseInt(inputs.hora.split(':')[0], 10);

    try {
      const response = await fetch(`${apiUrl}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            barrio: parseInt(comunaSeleccionada, 10), 
            dia: parseInt(diaNumero, 10),             
            franja: parseInt(horaSolo, 10)            
        })
      });  
      if (!response.ok) throw new Error(response.detail || 'Error en la respuesta de la API');
      const data = await response.json();
      setPrediccion(data);
      setErrorMsg(null);

    } catch (error) {
      setErrorMsg("No pudimos conectar con el servicio de predicción.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div className="sidebar">
        <h2>Mapa del Delito CABA</h2>
        <p className="subtitle">Selecciona un barrio y consulta el riesgo.</p>

        <div className="form-group">
          <label>Barrio</label>
          {/* Mostramos el nombre del barrio al usuario, pero enviamos la Comuna */}
          <input 
            type="text" 
            value={barrioSeleccionado || "Selecciona en el mapa..."} 
            disabled 
            className="input-barrio"
          />
        </div>

        <div className="form-row">
            <div className="form-group half">
            <label>Día</label>
            <select value={inputs.dia} onChange={(e) => setInputs({...inputs, dia: e.target.value})}>
                {Object.keys(diasMap).map(d => 
                  <option key={d} value={d}>{d}</option>
                )}
            </select>
            </div>
            <div className="form-group half">
            <label>Hora</label>
            <input type="time" value={inputs.hora} onChange={(e) => setInputs({...inputs, hora: e.target.value})} />
            </div>
        </div>

        <button 
          className="predict-btn" 
          onClick={handlePredict}
          disabled={!barrioSeleccionado || loading}
        >
          {loading ? "Calculando..." : "Predecir Riesgo"}
        </button>

        {errorMsg && (
          <p className="error-message">{errorMsg}</p>
        )}

        {prediccion && (
          <div>
            <h4>Nivel de Riesgo: {prediccion.risk_score_0_1.toFixed(2).replace('.', ',')}</h4>
            <p>Probabilidad estimada: <strong>{(prediccion.risk_score_0_1 * 100).toFixed(1).replace('.', ',')}%</strong></p>
          </div>
        )}
      </div>

      <MapContainer center={[-34.6037, -58.3816]} zoom={12} className="map-container" zoomControl={false}>
        <TileLayer 
            attribution='&copy; OpenStreetMap'
            url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
        />
        {geoData && (
          <GeoJSON 
            key={barrioSeleccionado || "init"} 
            data={geoData} 
            style={styleBarrio}
            onEachFeature={onEachFeature}
          />
        )}
      </MapContainer>
    </div>
  );
}

export default App;
