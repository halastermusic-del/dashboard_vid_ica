# app.py

import streamlit as st
import pandas as pd
import requests
from datetime import datetime
import json
import numpy as np

# --- CONFIGURACIÓN Y VARIABLES CRÍTICAS ---
# Reemplaza 'TU_CLAVE_AQUI' con tu clave real de OpenWeatherMap
API_KEY = "0fb6a8e85137ba1421f4c286dd2f3bf0" 
LATITUD_ICA = -14.0678 # Coordenadas de Ica
LONGITUD_ICA = -75.7286
TEMP_BASE_VID = 10.0 # Temperatura base (Tb) en °C para la vid
# ------------------------------------------

@st.cache_data
def calcular_gdd(temp_max, temp_min, temp_base):
    """
    Calcula los Grados Día de Crecimiento (GDD) para un día.
    Fórmula: (Tmax + Tmin) / 2 - Tb. El resultado nunca es negativo.
    """
    temp_media = (temp_max + temp_min) / 2
    gdd = max(0.0, temp_media - temp_base)
    return gdd

@st.cache_data(ttl=3600) # El dashboard llamará a la API de nuevo después de 3600 segundos (1 hora)
def obtener_pronostico(lat, lon, api_key):
    """Obtiene el pronóstico de 5 días / 3 horas para el cálculo de GDD futuro."""
    # Usaremos el endpoint 'forecast'
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric&lang=es"
    pass

    try:
        # Añadir un timeout para seguridad
        response = requests.get(url, timeout=10) 
        response.raise_for_status() # Lanza una excepción para códigos de error HTTP (4xx o 5xx)
        
        pronostico_json = response.json()
        
        # Revisión de errores de la API (ej. clave inválida, que devuelve un cod='401')
        if 'cod' in pronostico_json and str(pronostico_json['cod']) != '200':
             st.error(f"Error de API (Código {pronostico_json['cod']}): {pronostico_json['message']}")
             return None

        return pronostico_json
        
    except requests.exceptions.RequestException as e:
        st.error(f"Error de conexión o HTTP: {e}")
        return None
    except Exception as e:
        st.error(f"Error al procesar la respuesta: {e}")
        return None

@st.cache_data
def generar_datos_gdd(pronostico_json):
    """Procesa el JSON del pronóstico para calcular el GDD diario y acumulado."""
    
    data = []
    for item in pronostico_json['list']:
        # Convertir timestamp a objeto datetime.date
        data.append({
            'fecha': datetime.fromtimestamp(item['dt']).date(),
            'temp_max': item['main']['temp_max'],
            'temp_min': item['main']['temp_min']
        })
    df = pd.DataFrame(data)
    
    # 2. Encontrar la Tmax y Tmin por día
    df_diario = df.groupby('fecha').agg(
        Tmax=('temp_max', 'max'),
        Tmin=('temp_min', 'min')
    ).reset_index()
    
    # 3. Aplicar la función GDD
    df_diario['GDD'] = df_diario.apply(
        lambda row: calcular_gdd(row['Tmax'], row['Tmin'], TEMP_BASE_VID), axis=1
    )
    
    # 4. Calcular el GDD acumulado (de la proyección)
    df_diario['GDD Acumulado'] = df_diario['GDD'].cumsum()
    
    return df_diario

@st.cache_data
def cargar_datos_historicos(ruta_archivo):
    """Carga y procesa datos climáticos históricos del CSV para validación."""
    try:
        df = pd.read_csv(ruta_archivo, parse_dates=['Fecha'])
    except FileNotFoundError:
        return pd.DataFrame()
        
    df['Tmax'] = pd.to_numeric(df['Tmax'])
    df['Tmin'] = pd.to_numeric(df['Tmin'])
    
    # Calcular GDD y Acumulado
    df['GDD_Calculado'] = df.apply(
        lambda row: calcular_gdd(row['Tmax'], row['Tmin'], TEMP_BASE_VID), axis=1
    )
    df['GDD_Acumulado'] = df['GDD_Calculado'].cumsum()
    
    return df

def evaluar_riesgo_mildiu(pronostico_json):
    """
    Evalúa el riesgo de infección primaria de Mildiu (Modelo simplificado 3 Dieces).
    """
    if 'list' not in pronostico_json:
        # Devuelve valores predeterminados seguros para evitar el KeyError.
        return "⚠️ ERROR DE DATOS", 0.0, False
    
    df_list = pd.json_normalize(pronostico_json['list'])
    df_24h = df_list.head(8) # 24 horas

    # 1. Condición de Temperatura (Tmin >= 10°C en las 24h)
    temp_critica_met = (df_24h['main.temp_min'] >= 10).any()
    
    # 2. Condición de Lluvia (Acumulación >= 10 mm en las 24h)
    # Lógica segura para acceder a 'rain.3h'
    lluvia_acumulada = 0
    if 'rain.3h' in df_24h.columns:
        lluvia_acumulada = df_24h['rain.3h'].fillna(0).sum() 
        
    lluvia_critica_met = (lluvia_acumulada >= 10.0)

    # 3. Evaluación Final
    if temp_critica_met and lluvia_critica_met:
        return "🔴 RIESGO ALTO", lluvia_acumulada, temp_critica_met 
    elif temp_critica_met or lluvia_critica_met:
        return "🟠 RIESGO MEDIO", lluvia_acumulada, temp_critica_met
    else:
        return "🟢 RIESGO BAJO", lluvia_acumulada, temp_critica_met
    
def evaluar_riesgo_oidio(pronostico_json):
    """
    Evalúa el riesgo de Oídio basado en las horas en el rango óptimo (21°C - 27°C).
    """

    if 'list' not in pronostico_json:
        # Devuelve valores predeterminados seguros para evitar el KeyError.
        return "⚠️ ERROR DE DATOS", 0

    df_list = pd.json_normalize(pronostico_json['list'])
    
    temp_min_optima = 21
    temp_max_optima = 27
    
    # Nos aseguramos de que 'main.temp' exista antes de usarla
    if 'main.temp' not in df_list.columns:
        # Si falta la temperatura, devolvemos riesgo bajo como valor por defecto
        return "🟢 RIESGO BAJO", 0

    df_list['en_riesgo'] = df_list['main.temp'].apply(
        lambda t: 1 if temp_min_optima <= t <= temp_max_optima else 0
    )
    
    horas_riesgo = df_list['en_riesgo'].sum() * 3
    
    if horas_riesgo >= 24: 
        return "🔴 RIESGO ALTO", horas_riesgo
    elif horas_riesgo >= 12:
        return "🟠 RIESGO MEDIO", horas_riesgo
    else:
        return "🟢 RIESGO BAJO", horas_riesgo

# --- FUNCIÓN PRINCIPAL DEL DASHBOARD ---

def main():
    st.set_page_config(layout="wide", page_title="Dashboard Vitivinícola Ica")

    if st.button("🔄 Cargar Pronóstico Climático Actual"):
        st.cache_data.clear()  # Limpia toda la caché de datos de Streamlit
        st.rerun()             # Fuerza a Streamlit a recargar la página 
    
    st.markdown("""
<style>
/* 1. Fondo general de la aplicación */
.stApp {
    /* Mantenemos la imagen de fondo y sus propiedades */
    background-image: url("https://i.imgur.com/cqACoBo.jpeg"); 
    background-size: cover; 
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-position: center;
}

/* 2. Color del texto y encabezados: FORZAR BLANCO y APLICAR SOMBRA */
h1, h2, h3, h4, 
.st-emotion-cache-10ohe8r, 
.st-emotion-cache-1y829r, 
.st-emotion-cache-1r6rzzc, 
p, label, span { 
    /* Selector universal para todo el texto */
    color: #f7f7f7 !important; /* Texto blanco brillante */
    
    /* Sombra de Texto: Horizontal (1px), Vertical (1px), Desenfoque (3px), Color (negro) */
    text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.9); /* <- ESTA ES LA CLAVE */
}

/* 3. Color de los contenedores de Streamlit (Columnas, Contenedores) - Contenido Semi-Opaco */
.st-emotion-cache-1r6rzzc, .st-emotion-cache-0 { 
    background-color: rgba(30, 30, 30, 0.85); /* Fondo gris oscuro con 85% opacidad */
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.4);
}

/* 4. Asegurar que los gráficos tengan un fondo transparente o muy oscuro */
.st-emotion-cache-lglj2c { 
    background-color: transparent !important;
}

</style>
""", unsafe_allow_html=True)
    
    st.title("🍇 Dashboard de Predicción Agronómica para Vid (Ica)")
    st.header("Inteligencia de Negocio y Decisión Predictiva")
    st.markdown("---")
    
    # --- 1. MÓDULO DE VALIDACIÓN HISTÓRICA ---
    st.subheader("1. Validación del Modelo GDD vs. Fenología Observada")
    
   # --- 1. MÓDULO DE VALIDACIÓN HISTÓRICA (PUNTO CRÍTICO) ---
    st.header("1. Validación del Modelo GDD vs. Fenología Observada")
    
    try:
        df_historico = cargar_datos_historicos('datos_historicos_ica.csv')
        
        # EL GRÁFICO CLAVE DE LA VALIDACIÓN
        st.subheader("Gráfico Clave: GDD Acumulado vs. Eventos Reales")
        
        # El gráfico es el GDD acumulado
        st.line_chart(df_historico.set_index('Fecha')['GDD_Acumulado'])

        # CÁLCULO Y DISPLAY DEL ERROR DE PREDICCIÓN
        # Filtramos los eventos de fenología, asegurando que sean cadenas de texto
        eventos_observados = df_historico[df_historico['Fenologia_Observada'].apply(lambda x: isinstance(x, str) and x.strip() != '')]
        
        error_total = 0
        st.markdown("### Análisis de Precisión del Modelo:")
        
        if eventos_observados.empty:
            st.info("No se encontraron eventos de fenología (Brotación/Floración) en el CSV para validar el modelo.")
        else:
            for index, row in eventos_observados.iterrows():
                fenologia = row['Fenologia_Observada'].strip()
                fecha_observada = row['Fecha']
                
                # --- Simulación de la Predicción y cálculo del error ---
                # Asumimos valores de GDD estándar para la validación:
                # Brotación: 100 GDD; Floración: 500 GDD
                
                gdd_umbral = 0
                evento = ""

                if 'Brotación' in fenologia:
                    gdd_umbral = 100
                    evento = "Brotación"
                elif 'Floración' in fenologia:
                    gdd_umbral = 500
                    evento = "Floración"
                
                if gdd_umbral > 0:
                    # Buscamos la fecha en que el modelo GDD superó el umbral
                    df_prediccion = df_historico[df_historico['GDD_Acumulado'] >= gdd_umbral]
                    
                    if not df_prediccion.empty:
                        # La fecha predicha es la primera fecha que supera el umbral
                        fecha_predicha = df_prediccion['Fecha'].iloc[0]
                        
                        # Cálculo del error en días
                        error_dias = (fecha_observada - fecha_predicha).days
                        
                        st.markdown(f"**Evento:** {evento} | **GDD Umbral:** {gdd_umbral} GDD")
                        st.markdown(f"* Fecha Predicha por el Modelo: **{fecha_predicha.strftime('%Y-%m-%d')}**")
                        st.markdown(f"* Fecha Observada en Campo: **{fecha_observada.strftime('%Y-%m-%d')}**")
                        
                        # Mostrar la precisión
                        if error_dias == 0:
                            st.success(f"  ✅ **PRECISIÓN PERFECTA:** Error de 0 días.")
                        elif error_dias > 0:
                            st.warning(f"  ⚠️ El modelo predijo **{abs(error_dias)} día(s) después** de la realidad. (Tardío)")
                        else:
                            st.info(f"  ➡️ El modelo predijo **{abs(error_dias)} día(s) antes** de la realidad. (Temprano)")
                            
                        error_total += abs(error_dias)
            
            # Métrica final del error promedio
            st.markdown("---")
            error_promedio = error_total / len(eventos_observados)
            st.metric(
                label="Error Promedio de Predicción (Días)", 
                value=f"{error_promedio:.1f} días", 
                delta="Objetivo: Menos de 3 días"
            )
            
    except FileNotFoundError:
        st.error("Archivo 'datos_historicos_ica.csv' no encontrado. Asegúrese de crearlo.")
        
    st.markdown("---")

    # --- 2. MÓDULO DE PREDICCIÓN EN TIEMPO REAL ---
    st.subheader("2. Predicción en Tiempo Real y Riesgo Fitosanitario")

    pronostico_json = obtener_pronostico(LATITUD_ICA, LONGITUD_ICA, API_KEY)
    
    if pronostico_json:
        # Procesamiento para GDD y Mildiu
        df_gdd = generar_datos_gdd(pronostico_json)
        pronostico_json = pd.json_normalize(pronostico_json['list'])
        estado_riesgo, lluvia_ac, temp_met_critica = evaluar_riesgo_mildiu(pronostico_json)

        # Procesamiento para oidio
        estado_oidio, horas_oidio = evaluar_riesgo_oidio(pronostico_json)
        
        # --- COLUMNAS PARA LA VISUALIZACIÓN ---
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Grados Día (GDD) Proyectado")
            gdd_final = df_gdd['GDD Acumulado'].iloc[-1]
            st.metric(
                label=f"GDD Acumulado Proyectado (Próx. {len(df_gdd)} días)", 
                value=f"{gdd_final:.1f} GDD", 
                delta=f"T-Base: {TEMP_BASE_VID}°C"
            )
            st.line_chart(df_gdd, x='fecha', y=['GDD', 'GDD Acumulado'])
            st.caption("_Este gráfico guía la planificación de labores de campo (ej. poda, manejo de canopia)_")
            pass

        with col2:
            st.subheader("Riesgo de Mildiu")
            
            # --- OUTPUT CLAVE: SEMÁFORO DE RIESGO y CASO DE USO ---
            if "ERROR DE DATOS" in estado_riesgo:
                st.error(estado_riesgo)
                st.subheader("⚠️ Revisar la conexión/clave API")
                st.caption("No se pudo obtener el pronóstico detallado para el cálculo de riesgo.")
            elif "🔴" in estado_riesgo:
                st.error(estado_riesgo)
                st.subheader("⚠️ ¡ACCIÓN INMEDIATA REQUERIDA!")
                st.info("Una de las condiciones críticas está cerca de cumplirse. Revise el pronóstico cada 12 horas.")
            else:
                st.success(estado_riesgo)
                st.subheader("✅ Riesgo Bajo (Sin Alerta)")
                st.caption("No es necesaria la aplicación preventiva inmediata. Ahorre fungicida.")
            
            # Datos de justificación
            st.markdown("---")
            st.caption("### Justificación de las Condiciones (Modelo 3 Dieces - 24h)")
            st.markdown(f"* Lluvia Acumulada: **{lluvia_ac:.1f} mm** (Umbral: 10 mm)")
            st.markdown(f"**Temperatura Mínima > 10°C:** **{'Sí' if temp_met_critica else 'No'}**")
            st.caption("_Esto demuestra la **Decisión Basada en Datos**_.")

        with col3:
            st.subheader("Riesgo de Oídio (Cenicilla)")
        
            # --- OUTPUT CLAVE: SEMÁFORO DE RIESGO ---
            if "ERROR DE DATOS" in estado_riesgo:
                st.error(estado_riesgo)
                st.subheader("⚠️ Revisar la conexión/clave API")
                st.caption("No se pudo obtener el pronóstico detallado para el cálculo de riesgo.")
            elif "🔴" in estado_riesgo:
                st.error(estado_riesgo)
                st.subheader("⚠️ ¡ACCIÓN INMEDIATA REQUERIDA!")
            else:
                st.success(estado_oidio)
        
            st.caption("### Justificación del Oídio")
            st.markdown(f"* Horas Acumuladas en Rango Óptimo (21°-27°C): **{horas_oidio} horas**")
            st.caption("Umbral crítico: 24 horas.")

    else:
        st.error("No se pudo cargar el pronóstico. Revise su clave API o conexión.")

        st.markdown("---") # Separador
    st.subheader("👨‍💻 Nuestro Equipo")

    try:
        st.image("https://i.imgur.com/a2SCCEb.jpeg", caption="El equipo detrás del proyecto", width=200)
    except FileNotFoundError:
        st.warning("No se encontró 'https://i.imgur.com/a2SCCEb.jpeg'. Asegúrate de que la imagen esté en la carpeta del proyecto.")


    col_info1, col_info2 = st.columns(2)

    with col_info1:
        st.markdown("""
        ### Integrantes:
        * **Soto Licla Brahyan**
        * **Pachas Bardales Adrian**
        """)

    with col_info2:
        st.markdown(f"""
        ### Información Adicional:
        * **Universidad:** Universidad Privada San Juan Bautista
        * **Carrera:** Ingenieria Agroindustrial
        * **Curso:** Informatica Aplicada a la Ingenieria
        * **Docente:** {st.session_state.get('docente', 'YSAC SAMUEL FLORES MENDOZA')}
        """)


    st.markdown("---")
    st.caption("© 2025 Dashboard Vitivinícola Ica. Todos los derechos reservados.")

if __name__ == "__main__":
    # La aplicación se ejecuta con 'streamlit run app.py'
    main()