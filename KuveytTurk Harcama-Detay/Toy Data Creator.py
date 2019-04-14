#Import Librares
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import random
import time

#Oluşturulacak Değerler
#KartNo
#Tarih
#Ücret
#Bizim Kategorimiz

def random_with_N_digits(n):
    range_start = 10**(n-1)
    range_end = (10**n)-1
    return randint(range_start, range_end)

def strTimeProp(start, end, format, prop):
    """Get a time at a proportion of a range of two formatted times.

    start and end should be strings specifying times formated in the
    given format (strftime-style), giving an interval [start, end].
    prop specifies how a proportion of the interval to be taken after
    start.  The returned time will be in the specified format.
    """

    stime = time.mktime(time.strptime(start, format))
    etime = time.mktime(time.strptime(end, format))

    ptime = stime + prop * (etime - stime)

    return time.strftime(format, time.localtime(ptime))


def randomDate(start, end, prop):
    return strTimeProp(start, end, '%m/%d/%Y', prop)

z = random_with_N_digits(16)

def Randomdigits(n):
    sectorlist = ['Hizmet', 'Fatura', 'Yemek', 'Alışveriş', 'Giyim']
    cardnumber,prices,sectorx,date = [],[],[],[]
    for i in range(0,n):
        if randint(0,1) == 1:
            continue
        else:
            prices.append(abs(int(np.random.normal(1,100))))
            cardnumber.append(z)
            sectorx.append(sectorlist[randint(0,len(sectorlist)-1)])
            date.append(randomDate("11/1/2018", "4/1/2019", random.random()))

    return prices,cardnumber,sectorx, date

prices,cardnumber,sectorx,date = Randomdigits(4000)

xdata = {"Date":date,"CardNumber":cardnumber,"Prices":prices,"Hizmet":sectorx}
number3 = pd.DataFrame(xdata)
number3.Date = pd.to_datetime(number3.Date)
number3.set_index("Date",inplace=True)

number3.Prices.plot()
plt.show()

#Gerekli Çıktıları aldığımız için burası inaktif
#number3.to_excel("number5.xlsx")

