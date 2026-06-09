import streamlit as st
import pandas as pd
import numpy as np
from motor import (
    SUELOS, descargar_datos_github, cargar_datos, calcular_inflacion,
    procesar_cultivos, armar_portafolio,
    grafica_regresion, grafica_margen, grafica_punto_equilibrio, grafica_distribucion,
)

st.set_page_config(
    page_title='D-Y-A-C',
)


@st.cache_data
def cargar_todo():
    descargar_datos_github()
    df = cargar_datos()
    tasa, _ = calcular_inflacion()
    return df, tasa


for key in ['analisis_hecho', 'resultados', 'descartados', 'decision',
            'portafolio', 'inputs']:
    if key not in st.session_state:
        st.session_state[key] = None
if st.session_state.analisis_hecho is None:
    st.session_state.analisis_hecho = False


st.title(' D-Y-A-C')
st.caption('Diversificación y Aprovechamiento de tus Cultivos')

try:
    df_historico, TASA_INFLACION = cargar_todo()
except Exception as e:
    st.error('Error al cargar los datos. Verifica tu conexión.')
    st.stop()


st.sidebar.header(' Datos del Productor')
nombre = st.sidebar.text_input('Nombre', value='')
hectareas = st.sidebar.number_input('Hectáreas disponibles',
                                     min_value=0.5, max_value=500.0, value=5.0, step=0.5)
presupuesto = st.sidebar.number_input('Presupuesto (MXN)',
                                       min_value=10_000, max_value=10_000_000,
                                       value=200_000, step=10_000, format='%d')
modalidad = st.sidebar.selectbox('Modalidad de agua', options=['Riego', 'Temporal'])

if modalidad == 'Temporal':
    ciclo = 'Primavera-Verano'
    st.sidebar.info('Ciclo: Primavera-Verano')
else:
    ciclo = st.sidebar.selectbox('Ciclo productivo',
                                   options=['Primavera-Verano', 'Otoño-Invierno', 'Ambas'],
                                   index=2)

opciones_suelo = [(k, v[0], v[1]) for k, v in SUELOS.items()]
tipo_suelo = st.sidebar.selectbox(
    'Tipo de suelo',
    options=[s[1] for s in opciones_suelo],
    format_func=lambda x: f'{x} — {next(s[2] for s in opciones_suelo if s[1] == x)}',
)
alpha = st.sidebar.slider('Coeficiente α de Hurwicz',
                            min_value=0.0, max_value=1.0, value=0.5, step=0.05)

st.sidebar.markdown('---')
if st.sidebar.button('ANALIZAR', type='primary', use_container_width=True):
    if not nombre.strip():
        st.sidebar.error('Ingresa tu nombre')
    else:
        with st.spinner('Analizando cultivos...'):
            resultados, descartados, decision = procesar_cultivos(
                df_historico, hectareas, presupuesto, modalidad, ciclo,
                tipo_suelo, alpha, TASA_INFLACION,
            )
            portafolio = armar_portafolio(resultados, hectareas, presupuesto) if resultados else None
            st.session_state.analisis_hecho = True
            st.session_state.resultados = resultados
            st.session_state.descartados = descartados
            st.session_state.decision = decision
            st.session_state.portafolio = portafolio
            st.session_state.inputs = {
                'nombre': nombre, 'hectareas': hectareas, 'presupuesto': presupuesto,
                'modalidad': modalidad, 'ciclo': ciclo, 'tipo_suelo': tipo_suelo,
                'alpha': alpha,
            }


if not st.session_state.analisis_hecho:
    st.info(' Llena los datos del productor y haz clic en **ANALIZAR**.')
    st.stop()

resultados = st.session_state.resultados
descartados = st.session_state.descartados
decision = st.session_state.decision
portafolio = st.session_state.portafolio
inputs = st.session_state.inputs

if not resultados:
    st.error('No se encontraron cultivos viables.')
    st.stop()

rentables = [r for r in resultados if r['utilidad_neta'] > 0]

st.success(f"¡Hola **{inputs['nombre']}**! Aquí está tu propuesta.")

col1, col2, col3, col4 = st.columns(4)
col1.metric('Hectáreas', f"{inputs['hectareas']:.1f}")
col2.metric('Presupuesto', f"${inputs['presupuesto']:,.0f}")
col3.metric('Modalidad', inputs['modalidad'])
col4.metric('Suelo', inputs['tipo_suelo'])

st.markdown('---')


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    ' Portafolio',
    ' Estado de Resultados',
    ' Cultivos Rentables',
    ' Análisis Detallado',
    ' Tabla de Decisión',
])


# TAB 1: PORTAFOLIO
with tab1:
    st.header(' Tu Portafolio Diversificado')

    if not portafolio or not portafolio['posiciones']:
        st.warning('No hay cultivos rentables suficientes.')
    else:
        for p in portafolio['posiciones']:
            with st.container():
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                col1.subheader(p['etiqueta'])
                col1.write(f"**{p['cultivo']}**")
                col2.metric('Hectáreas', f"{p['hectareas']:.2f} ha")
                col3.metric('Inversión', f"${p['inversion']:,.0f}")
                col4.metric('Utilidad/año', f"${p['utilidad_anual']:,.0f}")
                if p['es_estabilizador_perenne']:
                    st.warning(f"⚠ Cultivo perenne — primera cosecha en el año {p['años_maduracion']}.")
                st.markdown('---')

        col1, col2, col3 = st.columns(3)
        col1.metric('Hectáreas usadas',
                    f"{portafolio['hectareas_usadas']:.2f} / {portafolio['hectareas_totales']:.0f} ha")
        col2.metric('Presupuesto usado',
                    f"${portafolio['presupuesto_usado']:,.0f} / ${portafolio['presupuesto_total']:,.0f}")
        col3.metric('Ganancia anual', f"${portafolio['er_utilidad_anual']:,.0f}")


# TAB 2: ESTADO DE RESULTADOS
with tab2:
    st.header(' Estado de Resultados Pro-Forma')
    st.caption('Resultados consolidados del portafolio recomendado')

    if not portafolio or not portafolio['posiciones']:
        st.warning('No hay portafolio para generar el estado de resultados.')
    else:
        st.subheader('Composición del portafolio')
        df_comp = pd.DataFrame([{
            'Posición': p['etiqueta'],
            'Cultivo': p['cultivo'],
            'Hectáreas': f"{p['hectareas']:.2f}",
            'Costo de producción': f"${p['inversion']:,.2f}",
            'Ingresos estimados': f"${p['ingresos']:,.2f}",
            'Utilidad neta': f"${p['utilidad_neta']:,.2f}",
        } for p in portafolio['posiciones']])
        st.dataframe(df_comp, hide_index=True, use_container_width=True)

        st.markdown('---')
        st.subheader('Estado de Resultados (Formato Contable)')

        df_er = pd.DataFrame({
            'Concepto': [
                'INGRESOS POR VENTAS',
                '',
                '(-) COSTO DE PRODUCCIÓN',
                '',
                '= UTILIDAD BRUTA',
                '',
                '(-) GASTOS DE OPERACIÓN',
                '   Provisión por riesgos (15%)',
                '',
                '= UTILIDAD DE OPERACIÓN',
                '',
                'Utilidad anualizada (consolidada)',
            ],
            'Monto (MXN)': [
                f"${portafolio['er_ingresos']:,.2f}",
                '',
                f"(${portafolio['er_costos_produccion']:,.2f})",
                '',
                f"${portafolio['er_utilidad_bruta']:,.2f}",
                '',
                '',
                f"(${portafolio['er_provision_riesgos']:,.2f})",
                '',
                f"${portafolio['er_utilidad_neta']:,.2f}",
                '',
                f"${portafolio['er_utilidad_anual']:,.2f}/año",
            ]
        })
        st.dataframe(df_er, hide_index=True, use_container_width=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric('Ingresos totales', f"${portafolio['er_ingresos']:,.0f}")
        col2.metric('Costos producción', f"${portafolio['er_costos_produccion']:,.0f}")
        col3.metric('Utilidad bruta', f"${portafolio['er_utilidad_bruta']:,.0f}")
        col4.metric('Utilidad neta', f"${portafolio['er_utilidad_neta']:,.0f}")

        st.markdown('---')
        st.subheader('Análisis de Riesgos (15% provisión)')
        st.caption('Cada riesgo se cubre con un % del total de utilidad bruta positiva')
        riesgos = resultados[0]['riesgos']
        df_riesgos = pd.DataFrame({
            'Riesgo': list(riesgos.keys()),
            'Afectación': [f'{v*100:.0f}%' for v in riesgos.values()],
        })
        st.dataframe(df_riesgos, hide_index=True, use_container_width=True)


# TAB 3: CULTIVOS RENTABLES
with tab3:
    st.header(' Cultivos Rentables')

    if not rentables:
        st.warning('Ningún cultivo es rentable.')
    else:
        rentables_ord = sorted(rentables, key=lambda x: x['utilidad_neta_anualizada'], reverse=True)
        df_rentables = pd.DataFrame([{
            'Cultivo': r['cultivo'],
            'Tipo': 'Perenne' if r['es_perenne'] else 'Anual',
            'Años a cosecha': r['años_maduracion'],
            'Suelo': '✅' if r['suelo_ok'] else '❌',
            'Utilidad/año': f"${r['utilidad_neta_anualizada']:,.0f}",
            'Costo de producción': f"${r['inversion_inicial']:,.0f}",
        } for r in rentables_ord])
        st.dataframe(df_rentables, hide_index=True, use_container_width=True)


# TAB 4: ANÁLISIS DETALLADO
with tab4:
    st.header(' Análisis Detallado por Cultivo')
    st.info(f" Este análisis asume que dedicas las {inputs['hectareas']:.0f} hectáreas COMPLETAS a un cultivo. "
             "En tu portafolio recomendado, las hectáreas se distribuyen entre 3 cultivos según 60/30/10.")

    if not rentables:
        st.warning('No hay cultivos rentables.')
    else:
        nombres_rentables = [r['cultivo'] for r in sorted(rentables, key=lambda x: x['utilidad_neta_anualizada'], reverse=True)]
        cultivo_seleccionado = st.selectbox('Selecciona un cultivo', options=nombres_rentables)
        r = next(r for r in rentables if r['cultivo'] == cultivo_seleccionado)

        if r.get('proyeccion_sospechosa'):
            st.warning(f"⚠ {r['motivo_alerta']}")

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric('Tipo', 'Perenne' if r['es_perenne'] else 'Anual')
        col2.metric('Años a cosecha', r['años_maduracion'])
        col3.metric('Suelo compatible', '✅ Sí' if r['suelo_ok'] else '❌ No')
        col4.metric('Utilidad/año', f"${r['utilidad_neta_anualizada']:,.0f}")
        estabilidad = "Alta" if r['coef_variacion'] < 20 else "Media" if r['coef_variacion'] < 40 else "Baja"
        col5.metric('Estabilidad del Precio', estabilidad, delta=f"{r['coef_variacion']:.1f}% CV", delta_color="inverse")

        st.markdown('---')
        st.subheader('Gráficas')

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('**Regresión polinomial**')
            fig1 = grafica_regresion(r)
            st.pyplot(fig1)

        with col_b:
            st.markdown('**Margen sobre costo**')
            fig2 = grafica_margen(r, tasa_inflacion=TASA_INFLACION)
            st.pyplot(fig2)

        col_c, col_d = st.columns(2)
        with col_c:
            st.markdown('**Punto de equilibrio**')
            fig3 = grafica_punto_equilibrio(r)
            st.pyplot(fig3)

        with col_d:
            st.markdown('**Distribución del precio con IC 95%**')
            fig4 = grafica_distribucion(r)
            st.pyplot(fig4)

        st.markdown('---')
        st.subheader(f' Estado de Resultados ({inputs["hectareas"]:.0f} ha de {cultivo_seleccionado})')

        rendimiento_total = r['rendimiento_ha'] * inputs['hectareas']
        pe_hectareas = r['punto_equilibrio'] / r['rendimiento_ha'] if r['rendimiento_ha'] > 0 else 0

        df_er_ind = pd.DataFrame({
            'Concepto': [
                'INGRESOS POR VENTAS',
                '   Producción × Precio promedio',
                '',
                '(-) COSTO DE PRODUCCIÓN',
                '',
                '= UTILIDAD BRUTA',
                '',
                '(-) GASTOS DE OPERACIÓN',
                '   Provisión por riesgos (15%)',
                '',
                '= UTILIDAD DE OPERACIÓN',
                '',
                'Utilidad anualizada',
                'Punto de equilibrio (Ton)',
                'Punto de equilibrio (Hectáreas)',
            ],
            'Monto (MXN)': [
                f"${r['ingresos_estimados']:,.2f}",
                f"   {rendimiento_total:.2f} ton × ${r['precio_promedio_continuo']:,.2f}",
                '',
                f"(${r['inversion_inicial']:,.2f})",
                '',
                f"${r['utilidad_bruta']:,.2f}",
                '',
                '',
                f"(${r['provision_riesgos']:,.2f})",
                '',
                f"${r['utilidad_neta']:,.2f}",
                '',
                f"${r['utilidad_neta_anualizada']:,.2f}/año",
                f"{r['punto_equilibrio']:,.2f} Ton",
                f"{pe_hectareas:.2f} ha",
            ]
        })
        st.dataframe(df_er_ind, hide_index=True, use_container_width=True)


# TAB 5: TABLA DE DECISIÓN
with tab5:
    st.header(' Toma de Decisiones bajo Incertidumbre')

    st.markdown("""
    Aplicamos los **5 criterios de decisión bajo incertidumbre** sobre la utilidad
    anualizada de cada cultivo en tres escenarios: pesimista, probable y optimista.
    """)

    if decision:
        st.subheader('Recomendación por Criterio')
        df_ganadores = pd.DataFrame({
            'Criterio': list(decision['ganadores'].keys()),
            'Cultivo recomendado': list(decision['ganadores'].values()),
        })
        st.dataframe(df_ganadores, hide_index=True, use_container_width=True)

        st.subheader('Matriz de Pagos')
        opciones = decision['opciones']
        df_matriz = pd.DataFrame({
            'Cultivo': [op['nombre'] for op in opciones],
            'Pesimista': [f"${op['utilidades'][0]:,.0f}" for op in opciones],
            'Probable': [f"${op['utilidades'][1]:,.0f}" for op in opciones],
            'Optimista': [f"${op['utilidades'][2]:,.0f}" for op in opciones],
            'Maximax': [f"${op['maximax']:,.0f}" for op in opciones],
            'Maximin': [f"${op['maximin']:,.0f}" for op in opciones],
            f"Hurwicz (α={inputs['alpha']})": [f"${op['hurwicz']:,.0f}" for op in opciones],
            'Laplace': [f"${op['laplace']:,.0f}" for op in opciones],
            'Savage': [f"${op['savage']:,.0f}" for op in opciones],
        })
        st.dataframe(df_matriz, hide_index=True, use_container_width=True)

st.markdown('---')
st.caption('UNRC - Gustavo A. Madero | LCDN 3er Sem | 2026-1')
