# -*- coding: utf-8 -*-
"""
Created on Thu May 28 18:51:55 2020

@author: hernan
"""

#%% EXPORTADO DE QGIS

import numpy as np
import smopy
import matplotlib.pyplot as plt

#%%
# mq=plt.imread('ComposiciónQgis.png')
mq=plt.imread('mm3.png')

mr = "Marcadores_lezama_test2.csv" # CAPA DE PUNTOS EXPORTADOS DE QGIS


latPts, lonPts = np.loadtxt(mr,
                     dtype=float,
                     delimiter=',',
                     skiprows=1,
                     usecols=[0,1]).T



nombrePts = np.loadtxt(mr,
                     dtype=str,
                     delimiter=',',
                     skiprows=1,
                     usecols=[3]).T


struct=np.array( [latPts.reshape(-1,1) , lonPts.reshape(-1,1) , nombrePts.reshape(-1,1)]).T



#%%

cr=mr = "calles13.csv"  #1623 callles




Way_type,lanes,maxspeed,name,oneway,surface,bicycle= np.loadtxt(mr,
                                                 dtype=str,
                                                 delimiter=',',
                                                 skiprows=1,
                                                 usecols=[3,4,5,6,7,8,9]).T



# oneway.shape
# name[680]
Way_type[680]
#%% 
ar = "20161113192738.txt"


lat, lon,ele = np.loadtxt(ar,
                     dtype=float,
                     delimiter=',',
                     skiprows=3,
                     usecols=[1,2,3]).T

# map = smopy.Map((min(lat), min(lon), max(lat), max(lon)), z=19)  #[default]
# map = smopy.Map((min(lat), min(lon), max(lat), max(lon)), z=19, tileserver="http://a.tile.stamen.com/toner/{z}/{x}/{y}.png") # [stamen toner (b/w high contrast)]
# map = smopy.Map((min(lat), min(lon), max(lat), max(lon)), z=19, tileserver="http://c.tile.stamen.com/watercolor/{z}/{x}/{y}.png")   #[watercolor look]
map = smopy.Map((min(lat), min(lon), max(lat), max(lon)), z=19, tileserver="https://tiles.wmflabs.org/bw-mapnik/{z}/{x}/{y}.png")  # [grayscale]



    
    
# map.show_ipython()

x, y = map.to_pixels(-34.629390, -58.370360)# coord camara  -34.629390, -58.370360
xr, yr = map.to_pixels(lat,lon)

#%%
import pandas as pd
from pandas import ExcelWriter
# df = pd.DataFrame({'Id': [1, 3, 2, 4],
#                    'Nombre': ['Juan', 'Eva', 'María', 'Pablo'],
#                    'Apellido': ['Méndez', 'López', 'Tito', 'Hernández']})
# df = df[['Id', 'Nombre', 'Apellido']]
# writer = ExcelWriter('ejemplo.xlsx')
# df.to_excel(writer, 'Hoja de datos', index=False)
# writer.save()

df = pd.DataFrame({'Id': np.array(range(3655)),
                   'x':lat,
                   'y': lon,
                   'z':ele})
df = df[['Id', 'x', 'y','z']]
writer = ExcelWriter('ejemplo3.xlsx')
df.to_excel(writer, 'Hoja de datos', index=False)
writer.save()


#%%


qq1,qq2=map.to_pixels( latPts, lonPts)


ax = map.show_mpl(figsize=(5,5))

# ax.plot(d1, d2, '.g', ms=5, mew=2);
ax.plot(xr, yr, 'xb', ms=2, mew=2,label='Recorrido del auto');
ax.plot(x, y, 'og', ms=12, mew=2,label='Camara FE');
ax.plot(qq1, qq2, 'or', ms=5, mew=2,label='QGIS points');

ax.legend()

plt.figure()

plt.imshow(mq,label='mapa exportado de Qgis')
plt.legend()


