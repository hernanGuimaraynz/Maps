# -*- coding: utf-8 -*-
"""
Created on Fri May  8 16:46:26 2020

@author: hernan
"""
import cv2

import numpy as np
import matplotlib.pyplot as pt
import datetime
import time
from mpl_toolkits.mplot3d import Axes3D

ar = "20161113192738.txt"


lat, lon, ele, spe = np.loadtxt(ar,
                     dtype=float,
                     delimiter=',',
                     skiprows=3,
                     usecols=[1,2,3,6]).T


# %% World Geodetic System 1984 (WGS 84)
# según wikipedia https://en.wikipedia.org/wiki/Geodetic_datum
# WGS 84 Defining Parameters Parameter 
a = 6378137.0 # semi mayor axis [m]
f_1 = 298.257223563 # Reciprocal of flattening

# WGS 84 derived geometric constants
b = 6356752.3142  # Semi-minor axis m
e2 = 	6.69437999014e-3  # First eccentricity squared
e22 = 6.73949674228e-3  # Second eccentricity squared

#%%

times0 = np.loadtxt(ar,
                   dtype=str,
                   delimiter=',',
                   skiprows=3,
                   usecols=[0])
times = [datetime.datetime.strptime(t,'%Y-%m-%dT%H:%M:%SZ') for t in times0]

times = np.array([time.mktime(t.timetuple()) + t.microsecond / 1.0e6
                      for t in times])
# restar 3 horas porque esta en UTC para que este en hora de argentina
times -= 3*60*60
# calculate time difference for speed estiamtion
inter = times[1:] - times[:-1] # [secs]


# %% to rectangular coordinates:
# https://en.wikipedia.org/wiki/Reference_ellipsoid
latRad, lonRad = (np.deg2rad(lat), np.deg2rad(lon))
a2 = a**2
b2 = b**2
cLat = np.cos(latRad)
sLat = np.sin(latRad)
# radius of curvature in the prime vertical
N = a2 / np.sqrt(a2 * cLat**2 + b2 * sLat**2)
Nele = N + ele

X =  Nele * cLat * np.cos(lonRad)
Y=  Nele * cLat * np.sin(lonRad)
Z= ( (b2/a2) * N + ele ) * sLat


# calculate displacement [m]
dispRect = np.sqrt((X[1:]-X[:-1])**2 +
                   (Y[1:]-Y[:-1])**2 +
                   (Z[1:]-Z[:-1])**2)

# calculate speed with rectangular coordinates
speRect = dispRect / inter

# %% cargar datos de encoder
# tiempos en microsegundos dt1 y dt2 son incertezas y Tau es el intervalo
dt1, dt2, Tau = np.loadtxt("m1_seba.txt").T
timesEnc0 = np.loadtxt("mtime_seba.txt")
vtime = np.loadtxt("vtime_seba.txt")

timesEnc = [datetime.datetime(np.int(t[0]),
                              np.int(t[1]),
                              np.int(t[2]),
                              np.int(t[3]),
                              np.int(t[4]),
                              np.int(t[5]),
                              np.int((t[5] - np.int(t[5]))*1e6)) for t in timesEnc0]

timesEnc = np.array([time.mktime(t.timetuple()) + t.microsecond / 1.0e6
                     for t in timesEnc],dtype=np.float64)

## remove data not relevant
#dejar = Tau < 1e-5

# asuming 30+-1cm radius wheel calculate speed in m/s
Ttemp=(Tau - dt2)


speEnc = (2e6 * np.pi * 0.3) / Tau[np.nonzero(Ttemp)]



speEncMax = (2e6 * np.pi * 0.31) / Ttemp[np.nonzero(Ttemp)]
Ttemp2=(Tau + dt1)
speEncMin = (2e6 * np.pi * 0.29) / Ttemp2[np.nonzero(Ttemp)]

# pt.figure()
# pt.plot(times-1,spe*3.6,'ko-',markersize=3,label="Velocidad directa de GPS") # en m/s
# pt.plot(times[1:]-1,speRect*3.6,'bo-',markersize=3,label='Diferencia de posiciones GPS')
# pt.plot(timesEnc[np.nonzero(Ttemp)], speEnc*3.6,'ro-',markersize=3,label='Encoder, r=30+-1cm')
# pt.plot(timesEnc[np.nonzero(Ttemp)], speEncMax*3.6,'r')
# pt.plot(timesEnc[np.nonzero(Ttemp)], speEncMin*3.6,'r')
# #pt.ylim([0,50])
# #pt.xlim([1.4790764e9+0,1.4790764e9+270])
# pt.legend()
# pt.xlabel('tiempo segundos desde 1979')
# pt.ylabel('velocidad en km/h')
# pt.title("restando 1seg a tiempo GPS")
# pt.show()



# fig = pt.figure()
# ax = fig.gca(projection='3d')
# ax.plot(X,Y,Z,'-*')
#%%

#coordenadas gps de la camara

latC,lonC,eleC=-34.629358, -58.370357,15.7


latRadC, lonRadC = (np.deg2rad(latC), np.deg2rad(lonC))
a2 = a**2
b2 = b**2
cLatC = np.cos(latRadC)
sLatC= np.sin(latRadC)
# radius of curvature in the prime vertical
NC = a2 / np.sqrt(a2 * cLatC**2 + b2 * sLatC**2)
NeleC = NC + eleC

XC =  NeleC * cLatC * np.cos(lonRadC)
YC=  NeleC * cLatC * np.sin(lonRadC)
ZC= ( (b2/a2) * NC + eleC ) * sLatC


# pt.figure()

# pt.plot(X,Y,'-*')
# pt.plot(np.array([XC]),np.array([YC]),'*r')

Pts=np.array([X,Y]).T

dist=np.linalg.norm(Pts - np.array([XC,YC]),axis=1)



radioCercania =16

CCcord= dist <= radioCercania 

PtsCerca=Pts[CCcord]


circle1 = pt.Circle((XC, YC), radioCercania, color='g', fill=False)

fig, ax = pt.subplots() # note we must use plt.subplots, not plt.subplot
# (or if you have an existing figure)
# fig = plt.gcf()
# ax = fig.gca()

ax.add_artist(circle1)
ax.plot(X,Y,'-*',label='Trazas de GPS para el recorrido del auto')
ax.plot(PtsCerca[:,0],PtsCerca[:,1],'*g',label='Coord. cercanas a la camara')
ax.plot(np.array([XC]),np.array([YC]),'*r',label='Posicion de la camara')
ax.legend()

#%%


TiemposCerca=times0[CCcord]

# print('Tiempo Auto pasando MIN= ' + str(min(TiemposCerca)))

# print( 'Tiempo Auto pasando MAX= ' + str(max(TiemposCerca)))




