#%%%
import os
from glob import glob
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from uncertainties import ufloat, unumpy
from scipy.optimize import curve_fit
#%% Lector Templog
def lector_templog(path):
    '''
    Busca archivo *templog.csv en directorio especificado.
    muestras = False plotea solo T(dt). 
    muestras = True plotea T(dt) con las muestras superpuestas
    Retorna arrys timestamp,temperatura 
    '''
    data = pd.read_csv(path,sep=';',header=5,
                            names=('Timestamp','T_CH1','T_CH2'),usecols=(0,1,2),
                            decimal=',',engine='python') 
    temp_CH1  = pd.Series(data['T_CH1']).to_numpy(dtype=float)
    temp_CH2  = pd.Series(data['T_CH2']).to_numpy(dtype=float)
    timestamp = np.array([datetime.strptime(date,'%Y/%m/%d %H:%M:%S') for date in data['Timestamp']]) 
    
    time = np.array([(t-timestamp[0]).total_seconds() for t in timestamp])
    return timestamp,time,temp_CH1, temp_CH2
#%% Transicion de fase
def detectar_TF_y_plot(t,T,T_central=0,delta_T=0.2,umbral_dTdt=0.15,min_puntos=5,plot=True,identif=None):
    """
    Detecta mesetas de transición de fase en una curva Temperatura vs Tiempo
    y opcionalmente genera un gráfico con la región identificada.

    La meseta se define como una región donde:
    - La temperatura se mantiene dentro de un intervalo alrededor de T_central
    - La derivada temporal |dT/dt| es menor que un umbral dado
    - Los puntos cumplen continuidad temporal (segmentos consecutivos)
    - La longitud del segmento supera un mínimo de puntos (min_puntos)

    Parámetros
    ----------
    t : array_like
        Tiempo [s]
    T : array_like
        Temperatura [°C]
    T_central : float, opcional
        Temperatura central de la transición (default: 0 °C)
    delta_T : float, opcional
        Tolerancia en temperatura (± delta_T) (default: 0.2 °C)
    umbral_dTdt : float, opcional
        Umbral máximo para |dT/dt| [°C/s] (default: 0.15 °C/s)
    min_puntos : int, opcional
        Número mínimo de puntos consecutivos para validar una meseta (default: 5)
    plot : bool, opcional
        Si True, genera la figura con los resultados (default: True)

    Retorna
    -------
    mesetas : list of dict
        Lista de mesetas detectadas. Cada elemento contiene:
        - "t_inicio" : tiempo inicial [s]
        - "t_fin"    : tiempo final [s]
        - "duracion" : duración de la meseta [s]
        - "T_media"  : temperatura media en la meseta [°C]

    fig : matplotlib.figure.Figure o None
        Figura generada (si plot=True)
    ax : matplotlib.axes.Axes o None
        Eje de temperatura
    ax2 : matplotlib.axes.Axes o None
        Eje de derivada dT/dt

    Notas
    -----
    - La derivada dT/dt se calcula mediante diferencias finitas (np.gradient).
    - La segmentación en bloques continuos evita identificar puntos aislados
      (ruido) como mesetas físicas.
    - El método es especialmente útil en experimentos térmicos donde la
      transición de fase se manifiesta como una meseta (ej: fusión del agua).
    - Para datos ruidosos se recomienda suavizar previamente la señal de temperatura."""
    dT_dt = np.gradient(T, t) # --- Derivada ---


    mask = ((T > (T_central - delta_T)) & (T < (T_central + delta_T)) & (np.abs(dT_dt) < umbral_dTdt))     # --- Filtro ---

    idx = np.where(mask)[0]

    if len(idx) == 0:
        return [], None, None, None

    segmentos = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)     # --- Segmentos continuos ---

    mesetas = []
    for seg in segmentos:
        if len(seg) >= min_puntos:
            t_ini = t[seg[0]]
            t_fin = t[seg[-1]]

            mesetas.append({"t_inicio": t_ini,"t_fin": t_fin,
                "duracion": t_fin - t_ini,"T_media": np.mean(T[seg])})

    if plot:
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10,7),sharex=True, constrained_layout=True)

        ax.plot(t, T, '.-', label='Temperatura')
        ax2.plot(t, dT_dt, '.-', label='dT/dt')

        # Umbrales derivada
        ax2.axhline(umbral_dTdt, color='k', ls='--')
        ax2.axhline(-umbral_dTdt, color='k', ls='--')

        # --- Mesetas ---
        for i, m in enumerate(mesetas):
            mask_m = (t >= m["t_inicio"]) & (t <= m["t_fin"])

            label = f'T. Fase ({m["duracion"]:.1f} s)' if i == 0 else None

            # Curva resaltada
            ax.plot(t[mask_m], T[mask_m], 'g-', lw=3, label=label)

            # Sombreado en ambos plots
            ax.axvspan(m["t_inicio"], m["t_fin"], color='g', alpha=0.2)
            ax2.axvspan(m["t_inicio"], m["t_fin"], color='g', alpha=0.2)

        # --- Labels ---
        ax.set_ylabel('T (°C)')
        ax2.set_ylabel('dT/dt (°C/s)')
        ax2.set_xlabel('t (s)')
        ax.set_title(identif+'\nTransición de fase S-L')

        for a in (ax, ax2):
            a.grid()
            a.legend()

        # --- Inset (primera meseta) ---
        if mesetas:
            m = mesetas[0]
            mask_m = (t >= m["t_inicio"]) & (t <= m["t_fin"])

            axin = ax.inset_axes([0.5, 0.1, 0.45, 0.45])
            axin.plot(t, T, 'k-')
            axin.plot(t[mask_m], T[mask_m], 'g-', lw=2)

            axin.axhline(T_central - delta_T, ls='--', color='k')
            axin.axhline(T_central + delta_T, ls='--', color='k')

            axin.set_xlim(m["t_inicio"] - 5, m["t_fin"] + 5)
            axin.set_ylim(T_central - 2*delta_T, T_central + 2*delta_T)

            axin.grid()
            ax.indicate_inset_zoom(axin)

        return mesetas, fig, ax, ax2

    return mesetas, None, None, None
#%% 1 Sumergida simple

paths_1 = glob('1_2mL_sumergida_simple/**/*.csv', recursive=True)
paths_1.sort()

# 100% CPA
fig100, (ax,ax2) =plt.subplots(2,1,figsize=(10,7),constrained_layout=True,sharey=True,sharex=False)

for i,e in enumerate(paths_1):
    if 'CPA100_RT' in e:
        _,t,T,_ = lector_templog(paths_1[i])
        ax.plot(t,T,'.-',label=f'{os.path.basename(e)[14:-4]}',alpha=0.8)
        print('Ploteando: ',os.path.basename(e)[14:-4])
    elif 'CPA100' in e:
        _,t,T,_ = lector_templog(paths_1[i])
        ax2.plot(t,T,'.-',label=f'{os.path.basename(e)[14:-4]}',alpha=0.8)
        print('Ploteando: ',os.path.basename(e)[14:-4])
ax.set_title('CPA100 - expuesto a RT',loc='left')
ax2.set_title('CPA100 - calentado en BT',loc='left')

for a in ax,ax2:
    a.grid()
    a.set_ylabel('T (°C)')
    a.set_xlabel('t (s)')
    a.axhline(y=0,c='k',lw=0.8,label='T = 0°C')
    a.axhline(-43,c='k',ls='--',lw=0.8,label='T$_m$ = -43°C')
    a.axhline(-121,c='k',ls='-.',lw=0.8,label='T$_g$ = -121°C')
    a.legend(loc='best',ncol=2)
ax.set_xlim(0,)
ax.set_xlim(0,1400)
ax2.set_xlim(0,300)
    
plt.suptitle('1 Sumergida simple - 2 mL - 100% CPA',fontsize=16)
plt.show()


# 090 - 080% CPA
fig090080, (ax,ax2) =plt.subplots(2,1,figsize=(10,7),constrained_layout=True,sharey=True,sharex=True)

for i,e in enumerate(paths_1):
    if 'CPA090' in e:
        _,t,T,_ = lector_templog(paths_1[i])
        ax.plot(t,T,'.-',label=f'{os.path.basename(e)[14:-4]}',alpha=0.8)
        print('Ploteando: ',os.path.basename(e)[14:-4])
    elif 'CPA080' in e:
        _,t,T,_ = lector_templog(paths_1[i])
        ax2.plot(t,T,'.-',label=f'{os.path.basename(e)[14:-4]}',alpha=0.8)
        print('Ploteando: ',os.path.basename(e)[14:-4])
ax.set_title('CPA090 - calentado en BT',loc='left')
ax2.set_title('CPA080 - calentado en BT',loc='left')

for a in ax,ax2:
    a.grid()
    a.set_ylabel('T (°C)')
    a.axhline(y=0,c='k',lw=0.8,label='T = 0°C')
    a.axhline(-43,c='k',ls='--',lw=0.8,label='T$_m$ = -43°C')
    a.axhline(-121,c='k',ls='-.',lw=0.8,label='T$_g$ = -121°C')
    a.legend(loc='best',ncol=2)
ax2.set_xlabel('t (s)')
ax2.set_xlim(0,300)
plt.suptitle('1 Sumergida simple - 090% CPA',fontsize=16)
plt.show()

# CPA000 = Agua
t_agua_all, T_agua_all = [],[]
fig000, ax =plt.subplots(figsize=(10,4),constrained_layout=True,sharey=True,sharex=False)

for i,e in enumerate(paths_1):
    if 'CPA000' in e:
        _,t_agua,T_agua,_ = lector_templog(paths_1[i])
        t_agua_all.append(t_agua)
        T_agua_all.append(T_agua)
        ax.plot(t_agua,T_agua,'.-',label=f'{os.path.basename(e)[14:-4]}',alpha=0.8)
        print('Ploteando: ',os.path.basename(e)[14:-4])

ax.set_title('Agua (CPA000) - calentado en BT',loc='left')

ax.grid()
ax.set_ylabel('T (°C)')
ax.set_xlabel('t (s)')
ax.axhline(y=0,c='k',lw=0.8,label='T = 0°C')
ax.axhline(-43,c='k',ls='--',lw=0.8,label='T$_m$ = -43°C')
ax.axhline(-121,c='k',ls='-.',lw=0.8,label='T$_g$ = -121°C')
ax.legend(loc='best',ncol=2)
ax.set_xlim(0,300)

plt.suptitle('1 Sumergida simple - 2 mL - 0% CPA 100% Agua',fontsize=16)
plt.show()

TFase_agua_1,figTF0, _,_ = detectar_TF_y_plot(t_agua_all[0], T_agua_all[0],T_central=-0.5,delta_T=1,umbral_dTdt=0.15,identif='Agua 1')
TFase_agua_2,figTF1, _,_ = detectar_TF_y_plot(t_agua_all[1], T_agua_all[1],T_central=-0.5,delta_T=1,umbral_dTdt=0.15,identif='Agua 2')
TFase_agua_3,figTF2, _,_ = detectar_TF_y_plot(t_agua_all[2], T_agua_all[2],T_central=-0.5,delta_T=1,umbral_dTdt=0.15,identif='Agua 3')
# Guardo figuras de configuracion 1 - sumergida simple y transicion de fase en agua
figs=[fig100,fig090080,fig000]
names=['CPA100','CPA090-080','Agua']
for i,e in enumerate(zip(figs,names)):
    e[0].savefig(f'1_{e[1]}.png',dpi=300)

figsTF=[figTF0,figTF1,figTF2]
namesTF=['agua_TF0','agua_TF1','agua_TF2']
for i,e in enumerate(zip(figsTF,namesTF)):
    e[0].savefig(f'1_{e[1]}.png',dpi=300)


#%% 2  Configuracion 2: 
paths_2 = glob('2_2mL_sumergida/**/*.csv', recursive=True)
paths_2.sort()

#100% CPA & 90% CPA

fig2100, (ax,ax2) =plt.subplots(2,1,figsize=(10,7),constrained_layout=True,sharey=True,sharex=True)

for i,e in enumerate(paths_2):
    if 'CPA100' in e:
        _,t,T,_ = lector_templog(paths_2[i])
        ax.plot(t,T,'.-',label=f'{os.path.basename(e)[14:-4]}',alpha=0.8)
        print('Ploteando: ',os.path.basename(e)[14:-4])
    elif 'CPA090' in e:
        _,t,T,_ = lector_templog(paths_2[i])
        ax2.plot(t,T,'.-',label=f'{os.path.basename(e)[14:-4]}',alpha=0.8)
        print('Ploteando: ',os.path.basename(e)[14:-4])
ax.set_title('CPA100 - calentado en BT',loc='left')
ax2.set_title('CPA090 - calentado en BT',loc='left')

for a in ax,ax2:
    a.grid()
    a.set_ylabel('T (°C)')
    a.axhline(y=0,c='k',lw=0.8,label='T = 0°C')
    a.axhline(-43,c='k',ls='--',lw=0.8,label='T$_m$ = -43°C')
    a.axhline(-121,c='k',ls='-.',lw=0.8,label='T$_g$ = -121°C')
    a.legend(loc='best',ncol=2)
    a.set_xlim(0,300)
ax2.set_xlabel('t (s)')
    
plt.suptitle('2 Sumergida - 2 mL - 100% CPA',fontsize=16)
plt.show()

# Agua 
t_agua_all, T_agua_all = [],[]

fig2000, ax =plt.subplots(figsize=(10,4),constrained_layout=True,sharey=True,sharex=False)
for i,e in enumerate(paths_2):
    if 'Agua' in e:
        _,t_agua,T_agua,_ = lector_templog(paths_2[i])
        t_agua_all.append(t_agua)
        T_agua_all.append(T_agua)
        ax.plot(t_agua,T_agua,'.-',label=f'{os.path.basename(e)[14:-4]}',alpha=0.8)
        print('Ploteando: ',os.path.basename(e)[14:-4])

ax.set_title('Agua (CPA000) - calentado en BT',loc='left')

ax.grid()
ax.set_ylabel('T (°C)')
ax.set_xlabel('t (s)')
ax.axhline(y=0,c='k',lw=0.8,label='T = 0°C')
ax.axhline(-43,c='k',ls='--',lw=0.8,label='T$_m$ = -43°C')
ax.axhline(-121,c='k',ls='-.',lw=0.8,label='T$_g$ = -121°C')
ax.legend(loc='best',ncol=2)
ax.set_xlim(0,400)

plt.suptitle('2 Sumergida - 2mL - 0% CPA 100% Agua',fontsize=16)
plt.show()
TFase_agua_1,figTF3, _, _ = detectar_TF_y_plot(t_agua_all[0], T_agua_all[0],T_central=-0.5,delta_T=1,umbral_dTdt=0.2,identif='Agua 1')
TFase_agua_2,figTF4, _, _ = detectar_TF_y_plot(t_agua_all[1], T_agua_all[1],T_central=-0.5,delta_T=1,umbral_dTdt=0.2,identif='Agua 2')

figs2=[fig2100,fig2000]
names2=['CPA100-090','Agua']
for i,e in enumerate(zip(figs2,names2)):
    e[0].savefig(f'2_{e[1]}.png',dpi=300)

figsTF2=[figTF3,figTF4]
namesTF2=['agua_TF3','agua_TF4']
for i,e in enumerate(zip(figsTF2,namesTF2)):
    e[0].savefig(f'2_{e[1]}.png',dpi=300)

# %% 3 - Usando vapor LN2
paths_3 = glob('3_2mL_vapor/**/*.csv', recursive=True)
paths_3.sort()

fig3100, ax =plt.subplots(figsize=(10,4),constrained_layout=True,sharey=True,sharex=False)
for i,e in enumerate(paths_3):
    if 'CPA100' in e:
        _,t,T,_ = lector_templog(paths_3[i])
        ax.plot(t,T,'.-',label=f'{os.path.basename(e)[14:-4]}',alpha=0.8)
        print('Ploteando: ',os.path.basename(e)[14:-4])

ax.set_title('CPA100 - 2mL enfriado con vapor -  calentado en BT',loc='left')

ax.grid()
ax.set_ylabel('T (°C)')
ax.set_xlabel('t (s)')
ax.axhline(y=0,c='k',lw=0.8,label='T = 0°C')
ax.axhline(-43,c='k',ls='--',lw=0.8,label='T$_m$ = -43°C')
ax.axhline(-121,c='k',ls='-.',lw=0.8,label='T$_g$ = -121°C')
ax.legend(loc='best',ncol=2)
ax.set_xlim(0,1000)
plt.savefig('3_CPA100_vapor.png',dpi=300)
plt.show()
#%% 4 Sumergida 1 mL

paths_4 = glob('4_1mL_sumergida/**/*.csv', recursive=True)
paths_4.sort()

# 100% CPA
fig4100, (ax,ax2,ax3) =plt.subplots(3,1,figsize=(11,10),constrained_layout=True,sharey=True,sharex=False)

for i,e in enumerate(paths_4):
    if 'CPA100' in e:
        _,t,T,_ = lector_templog(paths_4[i])
        ax.plot(t,T,'.-',label=f'{os.path.basename(e)[14:-4]}',alpha=0.8)
        print('Ploteando: ',os.path.basename(e)[14:-4])
    elif 'CPA050' in e:
        _,t,T,_ = lector_templog(paths_4[i])
        ax2.plot(t,T,'.-',label=f'{os.path.basename(e)[14:-4]}',alpha=0.8)
        print('Ploteando: ',os.path.basename(e)[14:-4])
    
    elif 'agua' in e:
        _,t,T,_ = lector_templog(paths_4[i])
        ax3.plot(t,T,'.-',label=f'{os.path.basename(e)[14:-4]}',alpha=0.8)
        print('Ploteando: ',os.path.basename(e)[14:-4])
        
        
ax.set_title('CPA100 - 1 mL - calentado en BT',loc='left')
ax2.set_title('CPA050 - 1mL - calentado en BT',loc='left')
ax3.set_title('Agua - 1mL - calentado en BT',loc='left')

for a in ax,ax2,ax3:
    a.grid()
    a.set_ylabel('T (°C)')
    a.set_xlabel('t (s)')
    a.axhline(y=0,c='k',lw=0.8,label='T = 0°C')
    a.axhline(-43,c='k',ls='--',lw=0.8,label='T$_m$ = -43°C')
    a.axhline(-121,c='k',ls='-.',lw=0.8,label='T$_g$ = -121°C')
    a.legend(loc='best',ncol=2)
ax.set_xlim(0,300)
ax2.set_xlim(0,700)
ax3.set_xlim(0,600)
    
plt.suptitle('4 Sumergida - 1 mL - 100% CPA',fontsize=16)
plt.savefig('4_1mL_sumergida.png',dpi=300)
plt.show()


# %%
