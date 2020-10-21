# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 12:47:49 2020

@author: hernan
"""

import matplotlib as mpl
# mpl.rcParams['agg.path.chunksize'] = 10000



import numpy as np
import smopy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ar='nodos_curvas_nivel_tif_grande_Recortado_lezama_xyz.csv'

# ar='nodos_3-6.csv'

# ar='xyz_nodos_tif_google.csv'
# ar='xyz_tif_grande_0p001.csv'
# ar='C:/Users/hernan/Desktop/Doctorado-datos/mis datos/arg/xyz_lezama.csv'


# lat, lon,ele = np.loadtxt(ar,
#                       dtype=float,
#                       delimiter=',',
#                       skiprows=3,
#                       usecols=[3,2,0]).T


# eleNZ= ele >0
# ele7=ele[eleNZ]
# lat7=lat[eleNZ]
# lon7=lon[eleNZ]

# ar='C:/Users/hernan/Desktop/Doctorado-datos/mis datos/arg/xyz_caba.csv'


#%%



# ar='C:/Users/hernan/Desktop/Doctorado-datos/mis datos/arg/puntos_lezama.csv'


ar='C:/Users/hernan/Desktop/Doctorado-datos/datosElevacion.csv'


# ar='C:/Users/hernan/Desktop/Doctorado-datos/mis datos/arg/Curvas de nivel/test_mde_IGN_puntos_xyz.csv'

lat, lon,ele = np.loadtxt(ar,
                      dtype=float,
                      delimiter=',',
                      skiprows=3,
                      usecols=[3,2,1]).T



# ele= np.concatenate((ele6 , ele7),axis=0)
# lon= np.concatenate((lon6 , lon7),axis=0)
# lat= np.concatenate((lat6 , lat7),axis=0)

# ar='xyz_rggedness.csv'
# lat, lon,ele = np.loadtxt(ar,
#                      dtype=float,
#                      delimiter=',',
#                      skiprows=3,
#                      usecols=[3,2,1]).T


# lat=lat[::3381]
# lon=lon[::3381]
# ele=ele[::3381]

# lonMax=-58.366743000
# lonMin=-58.372777000
# latMax=-34.625234000
# latMin=-34.631781000

# latQ= (lat>= latMin)



#  lat11=lat[latQ]
# lat11.shape






#  World Geodetic System 1984 (WGS 84)
# según wikipedia https://en.wikipedia.org/wiki/Geodetic_datum
# WGS 84 Defining Parameters Parameter 
a = 6378137.0 # semi mayor axis [m]
f_1 = 298.257223563 # Reciprocal of flattening

# WGS 84 derived geometric constants
b = 6356752.3142  # Semi-minor axis m
e2 = 	6.69437999014e-3  # First eccentricity squared
e22 = 6.73949674228e-3  # Second eccentricity squared


R = 6371.0088  # earth mean radious

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








ar2 = "20161113192738.txt"


lat2, lon2, ele2, spe2 = np.loadtxt(ar2,
                     dtype=float,
                     delimiter=',',
                     skiprows=3,
                     usecols=[1,2,3,6]).T

latRad2, lonRad2 = (np.deg2rad(lat2), np.deg2rad(lon2))
a2 = a**2
b2 = b**2
cLat2 = np.cos(latRad2)
sLat2 = np.sin(latRad2)
# radius of curvature in the prime vertical
N2 = a2 / np.sqrt(a2 * cLat2**2 + b2 * sLat2**2)
Nele2 = N2 + ele2

X2 =  Nele2 * cLat2 * np.cos(lonRad2)
Y2=  Nele2 * cLat2 * np.sin(lonRad2)
Z2= ( (b2/a2) * N2+ ele2 ) * sLat2




ar3 ='C:/Users/hernan/Desktop/Doctorado-datos/mis datos/arg/Curvas de nivel/semaforos_lezama.csv'


lat3, lon3 = np.loadtxt(ar3,
                     dtype=float,
                     delimiter=',',
                     skiprows=3,
                     usecols=[1,0]).T


ele3=np.ones((1,lat3.size))*5.0

ele3=ele3[0]
latRad3, lonRad3 = (np.deg2rad(lat3), np.deg2rad(lon3))
a3 = a**2
b3 = b**2
cLat3 = np.cos(latRad3)
sLat3 = np.sin(latRad3)
# radius of curvature in the prime vertical
N3 = a3 / np.sqrt(a3 * cLat3**2 + b3 * sLat3**2)
Nele3= N3 + ele3

X3 =  Nele3 * cLat3 * np.cos(lonRad3)
Y3=  Nele3 * cLat3 * np.sin(lonRad3)
Z3= ( (b3/a3) * N3+ ele3 ) * sLat3





fig = plt.figure()
ax = fig.gca(projection='3d')









ax.plot(X,Y,Z,'*g',label='calles')
ax.plot(X2,Y2,Z2,'r-*',label='recorrido')
ax.plot(X3,Y3,Z3,'b*',label='semaforos')


# zdirs = (None, 'x', 'y', 'z', (1, 1, 0), (1, 1, 1))
# xs = (1, 4, 4, 9, 4, 1)
# ys = (2, 5, 8, 10, 1, 2)
# zs = (10, 3, 8, 9, 1, 8)

# for zdir, x, y, z in zip(zdirs, xs, ys, zs):
#     label = '(%d, %d, %d), dir=%s' % (x, y, z, zdir)
#     ax.text(x, y, z, label, zdir)


# for i, txt in enumerate(Z):
#     ax.text(X, Y, Z, Z)

# plt.show()
ax.legend()







#%%
plt.figure()

plt.hist(ele.flatten(),200,label='curvas de nivel')


plt.hist(ele2.flatten(),200,label='recorrido')
plt.legend()


print(ele.shape)
print(ele2.shape)



#%%



#pendiente calle 

#  desde el norte ,donde empieza parque lezama (calle de la izquierda pegada al parque)

# altura :20m




#calle defensa(donde esta la camara)  altura : 7m


# dy=(16-6)

# dx=309


# m=dy/dx

# def f1(x,m,b):
#     return x*m+b



# x=range(0,310)




# plt.plot(x, [f1(i,m,6) for i in x])






#%%

# import ee
# import ee.mapclient
# import matplotlib.pyplot as plt


# # ee.Authenticate()


# ee.Initialize()
# # image = ee.Image()
# # print(image.getInfo())
# # -34.629390, -58.370360

# ee.mapclient.centerMap(-58.370360, -34.629390, 15)





# srtm = ee.Image('srtm90_v4')

# visParams={'min':0,'max':2000,'palette':['black','blue','purple','cyan','green','yellow','red']}

# ee.mapclient.addToMap(srtm,visParams,'SRTM')










# dataset = ee.Image('CGIAR/SRTM90_V4');

# elevation = dataset.select('elevation');
# slope = ee.Terrain.slope(elevation);
# ee.mapclient.centerMap(-58.370360, -34.629390, 16);
# Map.addLayer(slope, {min: 0, max: 60}, 'slope');


# print(elevation.getInfo())


# collection=(ee.ImageCollection("MODIS/006/MCD15A3H"))


# Select the median pixel.
# image1 = collection.median()

# Select the red, green and blue bands


# image = image1.select('B3', 'B2', 'B1')
# ee.mapclient.addToMap(image, {'gain': [1.4, 1.4, 1.1]})






















