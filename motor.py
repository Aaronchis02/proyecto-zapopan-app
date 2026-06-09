import os
import glob
import requests
import unicodedata
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

GITHUB_URL = 'https://raw.githubusercontent.com/Aar%C3%B3nchis02/proyecto-zapopan/main/'
AÑO_COSTO_FIRA = 2026

COSTOS_POR_HECTAREA = {
    'Maíz grano': 40_676, 'Frijol': 28_049, 'Calabacita': 265_092,
    'Mango': 46_586, 'Aguacate': 106_011.50, 'Guayaba': 50_000,
    'Avena forrajera en verde': 26_514, 'Agave': 337_487, 'Camote': 11_605.88,
    'Cebolla': 179_048, 'Limón': 55_119, 'Lima': 55_119, 'Coliflor': 95_000,
    'Rábano': 45_000, 'Betabel': 45_000, 'Ciruela': 82_000,
    'Sorgo grano': 28_188, 'Maíz forrajero en verde': 24_538,
    'Nopalitos': 35_000, 'Pastos y praderas': 18_500, 'Tomate verde': 165_000,
}

SUELOS_IDEALES = {
    'Maíz grano': ['Feozem', 'Luvisol', 'Fluvisol'],
    'Frijol': ['Feozem', 'Fluvisol'], 'Calabacita': ['Feozem', 'Fluvisol'],
    'Mango': ['Feozem', 'Regosol'], 'Aguacate': ['Feozem', 'Fluvisol'],
    'Guayaba': ['Feozem', 'Regosol'],
    'Avena forrajera en verde': ['Feozem', 'Luvisol'],
    'Agave': ['Regosol', 'Litosol'], 'Camote': ['Feozem', 'Fluvisol'],
    'Cebolla': ['Feozem', 'Fluvisol'], 'Coliflor': ['Feozem', 'Fluvisol'],
    'Lima': ['Feozem', 'Regosol'], 'Limón': ['Feozem', 'Regosol'],
    'Rábano': ['Feozem', 'Fluvisol'], 'Betabel': ['Feozem', 'Fluvisol'],
    'Ciruela': ['Feozem', 'Regosol'], 'Sorgo grano': ['Feozem', 'Luvisol'],
    'Maíz forrajero en verde': ['Feozem', 'Luvisol'],
    'Nopalitos': ['Regosol', 'Litosol', 'Feozem'],
    'Pastos y praderas': ['Feozem', 'Luvisol', 'Regosol', 'Fluvisol'],
    'Tomate verde': ['Feozem', 'Fluvisol'],
}

AÑOS_MADURACION = {
    'Agave': 6, 'Aguacate': 4, 'Ciruela': 4, 'Guayaba': 3,
    'Lima': 3, 'Limón': 3, 'Mango': 4, 'Nopalitos': 1, 'Pastos y praderas': 1,
}

SUELOS = {
    '1': ('Regosol',  'Tierra clara o arenosa, drena rápido'),
    '2': ('Feozem',   'Tierra negra u oscura, suave y fértil'),
    '3': ('Litosol',  'Tierra muy delgada, con roca superficial'),
    '4': ('Luvisol',  'Tierra rojiza o arcillosa, retiene humedad'),
    '5': ('Fluvisol', 'Tierra fresca y húmeda, cercana a agua'),
}


def normalizar(texto):
    if not isinstance(texto, str):
        return texto
    texto = unicodedata.normalize('NFD', texto)
    texto = ''.join(c for c in texto if unicodedata.category(c) != 'Mn')
    return texto.lower().strip()


COSTOS_NORM = {normalizar(k): v for k, v in COSTOS_POR_HECTAREA.items()}
SUELOS_NORM = {normalizar(k): v for k, v in SUELOS_IDEALES.items()}
MADURACION_NORM = {normalizar(k): v for k, v in AÑOS_MADURACION.items()}


def descargar_datos_github(carpeta_destino='.'):
    for año in range(2014, 2025):
        nombre = f'SIAP_Jalisco_{año}.csv'
        ruta = os.path.join(carpeta_destino, nombre)
        if not os.path.exists(ruta):
            r = requests.get(GITHUB_URL + nombre, timeout=30)
            if r.status_code == 200:
                with open(ruta, 'wb') as f:
                    f.write(r.content)
    ruta_inpp = os.path.join(carpeta_destino, 'INPP_INEGI.csv')
    if not os.path.exists(ruta_inpp):
        r = requests.get(GITHUB_URL + 'INPP_INEGI.csv', timeout=30)
        if r.status_code == 200:
            with open(ruta_inpp, 'wb') as f:
                f.write(r.content)


def cargar_datos(carpeta='.'):
    archivos = sorted(glob.glob(os.path.join(carpeta, 'SIAP_Jalisco_*.csv')))
    tablas = []
    for a in archivos:
        df_t = pd.read_csv(a, encoding='utf-8', low_memory=False)
        df_t = df_t.rename(columns={
            'Nomcultivo Sin Um': 'Nomcultivo',
            'Precio': 'Preciomediorural',
        })
        tablas.append(df_t)
    df = pd.concat(tablas, ignore_index=True)
    df['Nommunicipio'] = df['Nommunicipio'].astype(str).str.upper().str.strip()
    df['Nommodalidad'] = df['Nommodalidad'].astype(str).str.capitalize().str.strip()
    df['Nomcicloproductivo'] = df['Nomcicloproductivo'].astype(str).str.strip()
    for col in ['Sembrada', 'Rendimiento', 'Preciomediorural', 'Valorproduccion']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def calcular_inflacion(archivo_inpp='INPP_INEGI.csv'):
    inpp = pd.read_csv(archivo_inpp, encoding='utf-16-le')
    inpp.columns = ['Periodo', 'Area', 'Indice', 'Extra'][:len(inpp.columns)]
    inpp = inpp[inpp['Periodo'].str.match(r'^\d{4}/\d{2}$', na=False)].copy()
    inpp['Indice'] = pd.to_numeric(inpp['Indice'], errors='coerce')
    inpp['Año'] = inpp['Periodo'].str[:4].astype(int)
    inpp_filtro = inpp[(inpp['Año'] >= 2014) & (inpp['Año'] <= 2024)].copy()
    inpp_anual = inpp_filtro.groupby('Año')['Indice'].mean()
    tasas = inpp_anual.pct_change().dropna()
    return tasas.mean(), inpp_anual


def analizar_cultivo(datos_cultivo, hectareas, presupuesto, tipo_suelo,
                     tasa_inflacion=0.0459,
                     anio_proyeccion=2027, anio_base=2014):
    cultivo = datos_cultivo['Nomcultivo'].iloc[0]
    ciclo_actual = datos_cultivo['Nomcicloproductivo'].iloc[0]
    if ciclo_actual == 'Primavera-Verano':
        cultivo_completo = f"{cultivo} (PV)"
    elif ciclo_actual == 'Otoño-Invierno':
        cultivo_completo = f"{cultivo} (OI)"
    else:
        cultivo_completo = cultivo

    cultivo_norm = normalizar(cultivo)
    if cultivo_norm not in COSTOS_NORM:
        return {'cultivo': cultivo_completo, 'descartado': True, 'motivo': 'Sin datos de costo'}

    costo_ha = COSTOS_NORM[cultivo_norm]
    inversion_inicial = costo_ha * hectareas
    puede_escala_completa = inversion_inicial <= presupuesto
    hectareas_costeables = min(hectareas, presupuesto / costo_ha)

    if costo_ha > presupuesto:
        return {'cultivo': cultivo_completo, 'descartado': True,
                'motivo': f'Costo/ha (${costo_ha:,.0f}) excede presupuesto'}

    # FIX BUG MAÍZ: promediar duplicados por año antes de la regresión
    datos = datos_cultivo.dropna(subset=['Preciomediorural', 'Rendimiento'])
    datos = datos.groupby('Anio', as_index=False).agg({
        'Preciomediorural': 'mean',
        'Rendimiento': 'mean',
        'Sembrada': 'mean',
    }).sort_values('Anio')

    if len(datos) < 3:
        return {'cultivo': cultivo_completo, 'descartado': True, 'motivo': 'Menos de 3 años de datos'}

    x_calc = datos['Anio'].values - anio_base
    y = datos['Preciomediorural'].values.astype(float)
    rendimiento_ha = float(datos['Rendimiento'].mean())

    X = np.column_stack((np.ones(len(x_calc)), x_calc, x_calc**2))
    Beta = np.linalg.pinv(X.T @ X) @ X.T @ y
    b0, b1, b2 = Beta

    def P(t): return b0 + b1*t + b2*t**2
    def dP(t): return b1 + 2*b2*t

    t_obj = anio_proyeccion - anio_base
    precio_proyectado = float(P(t_obj))
    tendencia = float(dP(t_obj))

    precio_min_hist = float(np.min(y))
    precio_max_hist = float(np.max(y))
    proyeccion_sospechosa = False
    motivo_alerta = ''
    if precio_proyectado < 0:
        proyeccion_sospechosa = True
        motivo_alerta = 'Proyección negativa'
    elif precio_proyectado > precio_max_hist * 1.5:
        proyeccion_sospechosa = True
        motivo_alerta = f'Proyección {precio_proyectado/precio_max_hist:.1f}x sobre el máximo'

    def IP(t): return b0*t + (b1*t**2)/2 + (b2*t**3)/3

    ultimo_anio = int(datos['Anio'].max())
    t_inicio_int = ultimo_anio - anio_base
    rango_anios = t_obj - t_inicio_int

    costo_unitario_ton = inversion_inicial / (rendimiento_ha * hectareas)
    t_costo_base = AÑO_COSTO_FIRA - anio_base

    def Costo(t):
        return costo_unitario_ton * (1 + tasa_inflacion) ** (t - t_costo_base)

    valor_acumulado = float(IP(t_obj) - IP(t_inicio_int))
    if rango_anios > 0:
        precio_promedio_continuo = valor_acumulado / rango_anios
    else:
        precio_promedio_continuo = float(np.mean(y))

    precio_pesimista = float(np.min(y))
    precio_probable = float(np.mean(y))
    precio_optimista = float(np.max(y))

    media_precio = float(np.mean(y))
    desviacion_precio = float(np.std(y, ddof=1))
    coef_variacion = (desviacion_precio / media_precio) * 100 if media_precio > 0 else 0

    # IC 95% con T de Student (más correcto para muestra pequeña)
    n_años = len(y)
    error_estandar = desviacion_precio / np.sqrt(n_años)
    t_valor = float(stats.t.ppf(0.975, df=n_años - 1))
    ic_inferior = media_precio - (t_valor * error_estandar)
    ic_superior = media_precio + (t_valor * error_estandar)

    produccion_total = rendimiento_ha * hectareas
    util_pesimista = (produccion_total * precio_pesimista) - inversion_inicial
    util_probable = (produccion_total * precio_probable) - inversion_inicial
    util_optimista = (produccion_total * precio_optimista) - inversion_inicial

    riesgos = {
        'Falta de liquidez': 0.03,
        'Volatilidad de precios y mercado': 0.05,
        'Riesgos agroclimáticos': 0.04,
        'Inseguridad / robo': 0.02,
        'Altos costos de insumos': 0.01,
    }
    porcentaje_total_riesgos = sum(riesgos.values())

    ingresos_estimados = produccion_total * precio_promedio_continuo
    utilidad_bruta = ingresos_estimados - inversion_inicial
    provision_riesgos = max(utilidad_bruta, 0) * porcentaje_total_riesgos
    utilidad_neta = utilidad_bruta - provision_riesgos

    punto_equilibrio = inversion_inicial / precio_promedio_continuo if precio_promedio_continuo > 0 else float('inf')

    suelos_buenos = SUELOS_NORM.get(cultivo_norm, [])
    suelo_ok = tipo_suelo in suelos_buenos
    es_perenne = ciclo_actual == 'Perennes'
    años_maduracion = MADURACION_NORM.get(cultivo_norm, 1)
    utilidad_neta_anualizada = utilidad_neta / años_maduracion

    return {
        'cultivo': cultivo_completo, 'descartado': False,
        'inversion_inicial': inversion_inicial, 'rendimiento_ha': rendimiento_ha,
        'costo_ha': costo_ha,
        'x_calc': x_calc, 'y': y, 'b0': b0, 'b1': b1, 'b2': b2,
        'anio_base': anio_base, 'anio_proyeccion': anio_proyeccion, 't_obj': t_obj,
        'P_func': P, 'IP_func': IP, 'Costo_func': Costo,
        'precio_proyectado': precio_proyectado, 'tendencia': tendencia,
        'ultimo_anio': ultimo_anio, 't_inicio_int': t_inicio_int,
        'precio_promedio_continuo': precio_promedio_continuo,
        'costo_unitario_ton': costo_unitario_ton,
        'precio_pesimista': precio_pesimista, 'precio_probable': precio_probable,
        'precio_optimista': precio_optimista,
        'util_pesimista': util_pesimista, 'util_probable': util_probable,
        'util_optimista': util_optimista,
        'utilidades_escenarios_anuales': [util_pesimista/años_maduracion, util_probable/años_maduracion, util_optimista/años_maduracion],
        'riesgos': riesgos, 'porcentaje_total_riesgos': porcentaje_total_riesgos,
        'ingresos_estimados': ingresos_estimados,
        'utilidad_bruta': utilidad_bruta, 'provision_riesgos': provision_riesgos,
        'utilidad_neta': utilidad_neta,
        'utilidad_neta_anualizada': utilidad_neta_anualizada,
        'punto_equilibrio': punto_equilibrio,
        'produccion_total': produccion_total,
        'suelos_buenos': suelos_buenos, 'suelo_ok': suelo_ok,
        'es_perenne': es_perenne, 'años_maduracion': años_maduracion,
        'media_precio': media_precio, 'desviacion_precio': desviacion_precio,
        'coef_variacion': coef_variacion,
        'ic_inferior': ic_inferior, 'ic_superior': ic_superior,
        'proyeccion_sospechosa': proyeccion_sospechosa, 'motivo_alerta': motivo_alerta,
        'puede_escala_completa': puede_escala_completa,
        'hectareas_costeables': hectareas_costeables,
    }


def aplicar_criterios_decision(resultados, alpha=0.5):
    opciones = [{'nombre': r['cultivo'], 'utilidades': r['utilidades_escenarios_anuales']} for r in resultados]
    opciones.append({'nombre': 'No hacer nada', 'utilidades': [0.0, 0.0, 0.0]})

    for op in opciones:
        op['maximax'] = max(op['utilidades'])
        op['maximin'] = min(op['utilidades'])
        op['hurwicz'] = alpha * max(op['utilidades']) + (1 - alpha) * min(op['utilidades'])
        op['laplace'] = sum(op['utilidades']) / len(op['utilidades'])

    n_esc = len(opciones[0]['utilidades'])
    mejores = [max(op['utilidades'][j] for op in opciones) for j in range(n_esc)]
    for op in opciones:
        op['savage'] = max([mejores[j] - op['utilidades'][j] for j in range(n_esc)])

    return {
        'opciones': opciones,
        'ganadores': {
            '1. Optimista (Maximax)': max(opciones, key=lambda x: x['maximax'])['nombre'],
            '2. Pesimista (Maximin)': max(opciones, key=lambda x: x['maximin'])['nombre'],
            f'3. Realismo (Hurwicz, α={alpha})': max(opciones, key=lambda x: x['hurwicz'])['nombre'],
            '4. Probabilidades iguales (Laplace)': max(opciones, key=lambda x: x['laplace'])['nombre'],
            '5. Arrepentimiento minimax (Savage)': min(opciones, key=lambda x: x['savage'])['nombre'],
        },
        'alpha': alpha,
    }


def procesar_cultivos(df_historico, hectareas, presupuesto, modalidad,
                     ciclo, tipo_suelo, alpha, tasa_inflacion):
    if ciclo == 'Ambas':
        ciclos_validos = ['Primavera-Verano', 'Otoño-Invierno', 'Perennes']
    else:
        ciclos_validos = [ciclo, 'Perennes']

    df_filtrado = df_historico[
        (df_historico['Nommunicipio'] == 'ZAPOPAN') &
        (df_historico['Nommodalidad'] == modalidad) &
        (df_historico['Nomcicloproductivo'].isin(ciclos_validos))
    ]
    cultivos_viables = df_filtrado.groupby(
        ['Nomcultivo', 'Nomcicloproductivo']
    ).filter(lambda x: x['Anio'].nunique() >= 3)

    resultados = []
    descartados = []
    for (cultivo, ciclo_c), df_c in cultivos_viables.groupby(['Nomcultivo', 'Nomcicloproductivo']):
        r = analizar_cultivo(df_c, hectareas, presupuesto, tipo_suelo, tasa_inflacion=tasa_inflacion)
        if r.get('descartado'):
            descartados.append(r)
        else:
            resultados.append(r)

    decision = aplicar_criterios_decision(resultados, alpha=alpha) if resultados else None
    return resultados, descartados, decision


def armar_portafolio(resultados, hectareas, presupuesto):
    rentables = [r for r in resultados if r['utilidad_neta'] > 0]
    no_rentables = [r for r in resultados if r['utilidad_neta'] <= 0]
    corto = [r for r in rentables if not r['es_perenne']]
    perennes = [r for r in rentables if r['es_perenne']]
    corto_ord = sorted(corto, key=lambda x: x['utilidad_neta_anualizada'], reverse=True)
    perennes_ord = sorted(perennes, key=lambda x: x['utilidad_neta_anualizada'], reverse=True)

    principal = corto_ord[0] if len(corto_ord) >= 1 else None
    secundario = corto_ord[1] if len(corto_ord) >= 2 else None
    estabilizador = None
    estabilizador_es_perenne = False
    if len(corto_ord) >= 3:
        estabilizador = corto_ord[2]
    elif perennes_ord:
        estabilizador = perennes_ord[0]
        estabilizador_es_perenne = True

    portafolio = [
        ('Principal (60%)', principal, 0.60),
        ('Secundario (30%)', secundario, 0.30),
        ('Estabilizador (10%)', estabilizador, 0.10),
    ]

    hectareas_restantes = hectareas
    presupuesto_restante = presupuesto
    posiciones_data = []

    for i, (etiqueta, cultivo_data, pct) in enumerate(portafolio):
        if cultivo_data is None:
            continue
        es_ultimo = all(c is None for _, c, _ in portafolio[i+1:])
        if es_ultimo:
            inversion_asignada = presupuesto_restante
        else:
            inversion_asignada = min(presupuesto * pct, presupuesto_restante)
        costo_por_ha = cultivo_data['costo_ha']
        if es_ultimo:
            hectareas_por_tierra = hectareas_restantes
        else:
            hectareas_por_tierra = hectareas * pct
        hectareas_por_dinero = inversion_asignada / costo_por_ha
        hectareas_finales = min(hectareas_por_dinero, hectareas_por_tierra, hectareas_restantes)
        inversion_real = hectareas_finales * costo_por_ha

        proporcion = hectareas_finales / hectareas if hectareas > 0 else 0
        ingresos_pos = cultivo_data['ingresos_estimados'] * proporcion
        costo_pos = inversion_real
        utilidad_bruta_pos = ingresos_pos - costo_pos
        provision_pos = max(utilidad_bruta_pos, 0) * cultivo_data['porcentaje_total_riesgos']
        utilidad_neta_pos = utilidad_bruta_pos - provision_pos
        utilidad_anual_pos = utilidad_neta_pos / cultivo_data['años_maduracion']

        hectareas_restantes -= hectareas_finales
        presupuesto_restante -= inversion_real
        posiciones_data.append({
            'etiqueta': etiqueta, 'cultivo': cultivo_data['cultivo'],
            'hectareas': hectareas_finales, 'inversion': inversion_real,
            'ingresos': ingresos_pos,
            'utilidad_bruta': utilidad_bruta_pos,
            'provision_riesgos': provision_pos,
            'utilidad_neta': utilidad_neta_pos,
            'utilidad_anual': utilidad_anual_pos,
            'pct': pct, 'es_perenne': cultivo_data['es_perenne'],
            'años_maduracion': cultivo_data['años_maduracion'],
            'es_estabilizador_perenne': (etiqueta.startswith('Estabilizador') and estabilizador_es_perenne),
        })

    cultivos_en_portafolio = {p['cultivo'] for p in posiciones_data}
    perennes_no_usados = [r for r in perennes_ord if r['cultivo'] not in cultivos_en_portafolio]

    total_ingresos = sum(p['ingresos'] for p in posiciones_data)
    total_costos = sum(p['inversion'] for p in posiciones_data)
    total_utilidad_bruta = sum(p['utilidad_bruta'] for p in posiciones_data)
    total_provision = sum(p['provision_riesgos'] for p in posiciones_data)
    total_utilidad_neta = sum(p['utilidad_neta'] for p in posiciones_data)
    total_utilidad_anual = sum(p['utilidad_anual'] for p in posiciones_data)

    return {
        'posiciones': posiciones_data, 'rentables': rentables,
        'no_rentables': no_rentables, 'perennes_complementarios': perennes_no_usados,
        'hectareas_usadas': hectareas - hectareas_restantes,
        'hectareas_totales': hectareas,
        'presupuesto_usado': presupuesto - presupuesto_restante,
        'presupuesto_total': presupuesto,
        'hectareas_restantes': hectareas_restantes,
        'presupuesto_restante': presupuesto_restante,
        'er_ingresos': total_ingresos,
        'er_costos_produccion': total_costos,
        'er_utilidad_bruta': total_utilidad_bruta,
        'er_provision_riesgos': total_provision,
        'er_utilidad_neta': total_utilidad_neta,
        'er_utilidad_anual': total_utilidad_anual,
    }


# ============================================================
# GRÁFICAS POR SEPARADO
# ============================================================

def grafica_regresion(r):
    """Gráfica 1: Regresión polinomial."""
    fig, ax = plt.subplots(figsize=(10, 6))
    t_rango = np.linspace(0, r['t_obj'] + 1, 100)
    precios_curva = r['P_func'](t_rango)

    ax.scatter(r['x_calc'] + r['anio_base'], r['y'],
                color='black', s=80, label='Histórico', zorder=5)
    ax.plot(t_rango + r['anio_base'], precios_curva,
                color='blue', label='Polinomio grado 2', linewidth=2.5)
    linea_tan = r['tendencia'] * (t_rango - r['t_obj']) + r['precio_proyectado']
    ax.plot(t_rango + r['anio_base'], linea_tan,
                color='red', linestyle='--', linewidth=2,
                label=f"Tendencia (m={r['tendencia']:.0f})")
    ax.scatter([r['anio_proyeccion']], [r['precio_proyectado']],
                color='orange', s=180, zorder=6,
                edgecolor='black', label=f"Proyección {r['anio_proyeccion']}")
    ax.set_title('Regresión Polinomial y Tendencia', fontsize=14)
    ax.set_xlabel('Año', fontsize=12)
    ax.set_ylabel('Precio ($/Ton)', fontsize=12)
    ax.legend(fontsize=11, loc='best')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def grafica_margen(r, tasa_inflacion=0.0459):
    """Gráfica 2: Margen sobre costo con inflación."""
    fig, ax = plt.subplots(figsize=(10, 6))
    t_rango = np.linspace(0, r['t_obj'] + 1, 100)
    precios_curva = r['P_func'](t_rango)
    costos_curva = r['Costo_func'](t_rango)

    ax.plot(t_rango + r['anio_base'], precios_curva,
                color='blue', linewidth=2.5, label='Precio P(t)')
    ax.plot(t_rango + r['anio_base'], costos_curva,
                color='red', linewidth=2,
                label=f"Costo con inflación ({tasa_inflacion*100:.1f}%/año)")

    t_int = np.linspace(r['t_inicio_int'], r['t_obj'], 50)
    p_int = r['P_func'](t_int)
    c_int = r['Costo_func'](t_int)
    ax.fill_between(t_int + r['anio_base'], c_int, p_int,
                    where=(p_int > c_int), color='green', alpha=0.5,
                    label='Margen positivo')
    ax.fill_between(t_int + r['anio_base'], c_int, p_int,
                    where=(p_int < c_int), color='red', alpha=0.4,
                    label='Margen negativo')

    ax.axhline(y=r['precio_promedio_continuo'], color='darkgreen', linestyle='--',
                linewidth=1.5,
                label=f"Precio TVM (${r['precio_promedio_continuo']:,.0f}/ton)")
    ax.axvline(x=r['ultimo_anio'], color='gray', linestyle=':', alpha=0.6)
    ax.axvline(x=r['anio_proyeccion'], color='gray', linestyle=':', alpha=0.6)

    ax.set_title(f"Margen sobre costo {r['ultimo_anio']}-{r['anio_proyeccion']}", fontsize=14)
    ax.set_xlabel('Año', fontsize=12)
    ax.set_ylabel('Precio ($/Ton)', fontsize=12)
    ax.legend(fontsize=10, loc='best')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def grafica_punto_equilibrio(r):
    """Gráfica 3: Punto de equilibrio (contabilidad)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Rango de toneladas a graficar (de 0 hasta 1.5x la producción esperada)
    produccion_max = r['produccion_total'] * 1.5
    toneladas = np.linspace(0, produccion_max, 100)

    # Línea de ingresos (creciente): Ingresos = toneladas × precio
    precio = r['precio_promedio_continuo']
    ingresos = toneladas * precio

    # Línea de costos (horizontal): Costo fijo
    costos = np.full_like(toneladas, r['inversion_inicial'])

    # Sombreado: zona de ganancia (verde) y pérdida (roja)
    ax.fill_between(toneladas, costos, ingresos,
                    where=(ingresos >= costos),
                    color='green', alpha=0.3, label='Zona de ganancia')
    ax.fill_between(toneladas, costos, ingresos,
                    where=(ingresos < costos),
                    color='red', alpha=0.3, label='Zona de pérdida')

    # Líneas
    ax.plot(toneladas, ingresos, color='green', linewidth=2.5,
            label=f'Ingresos = Ton × ${precio:,.0f}')
    ax.plot(toneladas, costos, color='red', linewidth=2.5,
            label=f'Costo fijo = ${r["inversion_inicial"]:,.0f}')

    # Marcar el punto de equilibrio
    pe_ton = r['punto_equilibrio']
    pe_pesos = pe_ton * precio
    ax.scatter([pe_ton], [pe_pesos], color='orange', s=200,
               zorder=10, edgecolor='black', linewidth=2,
               label=f'Punto equilibrio = {pe_ton:.1f} Ton')

    # Línea vertical del punto de equilibrio
    ax.axvline(x=pe_ton, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)

    # Línea vertical de la producción esperada
    ax.axvline(x=r['produccion_total'], color='blue', linestyle='--',
               linewidth=1.5, alpha=0.7,
               label=f'Producción esperada = {r["produccion_total"]:.1f} Ton')

    ax.set_title('Punto de Equilibrio', fontsize=14)
    ax.set_xlabel('Toneladas producidas', fontsize=12)
    ax.set_ylabel('Pesos ($)', fontsize=12)
    ax.legend(fontsize=10, loc='best')
    ax.grid(alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    return fig


def grafica_distribucion(r):
    """Gráfica 4: Distribución del precio con IC 95%."""
    fig, ax = plt.subplots(figsize=(10, 6))
    años_hist = r['x_calc'] + r['anio_base']

    ax.axhspan(r['ic_inferior'], r['ic_superior'],
                color='lightblue', alpha=0.5,
                label=f"IC 95%: [${r['ic_inferior']:,.0f}, ${r['ic_superior']:,.0f}]")
    ax.scatter(años_hist, r['y'], color='black', s=100,
                label='Precios históricos', zorder=5, edgecolor='white', linewidth=1.5)
    ax.axhline(y=r['media_precio'], color='blue', linewidth=2.5,
                label=f"Media μ = ${r['media_precio']:,.0f}")
    ax.axhline(y=r['media_precio'] + r['desviacion_precio'],
                color='blue', linestyle=':', linewidth=1.8,
                label=f"μ + σ = ${r['media_precio']+r['desviacion_precio']:,.0f}")
    ax.axhline(y=r['media_precio'] - r['desviacion_precio'],
                color='blue', linestyle=':', linewidth=1.8,
                label=f"μ - σ = ${r['media_precio']-r['desviacion_precio']:,.0f}")

    ax.set_title(f"Distribución del Precio (CV = {r['coef_variacion']:.1f}%)", fontsize=14)
    ax.set_xlabel('Año', fontsize=12)
    ax.set_ylabel('Precio ($/Ton)', fontsize=12)
    ax.legend(fontsize=10, loc='best')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig
