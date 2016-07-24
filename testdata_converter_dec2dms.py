import csv
from geo import Geo
from math import modf

def dec2dms(dec) :
    pm = "" if(0<=dec) else "-"
    dec, d = modf(abs(dec))
    dec, m = modf(dec*60.0)
    s = dec*60.0
    d_dms = '%d'%d if(0<d) else ""
    m_dms = '%02d'%m if(0<d) else '%d'%m if(0<m) else ""
    s_dms = '%018.15f'%s if(0<m or 0<d) else "%.15f"%s
    return pm + d_dms + m_dms + s_dms

f = open("testdata_dec.csv", 'rb')
dataReader = csv.reader(f)

for row in dataReader:
    lt1_dec = float(row[0])
    lg1_dec = (float(row[1])+180) %360 -180
    lt1_dms = dec2dms(lt1_dec)
    lg1_dms = dec2dms(lg1_dec)

    lt2_dec = float(row[2])
    lg2_dec = (float(row[3])+180) %360 -180
    lt2_dms = dec2dms(lt2_dec)
    lg2_dms = dec2dms(lg2_dec)

    print lt1_dms, lg1_dms, lt2_dms, lg2_dms
