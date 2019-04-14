# Import Libraries
import pickle
import pandas as pd
import numpy as np
import requests
import json
from pandas.io.json import json_normalize
import datetime
from random import randint
import warnings
warnings.filterwarnings('ignore')
from fbprophet import Prophet
import time
import tweepy
import nltk
import re
import os
from sklearn.datasets import load_files
nltk.download('stopwords')
from snowballstemmer import stemmer

# Kmeans ile kullanıcıları kümeleyecek modeli çağırıyoruz.


def groupprediction(x):
    kmeans = pickle.load(open('kmeans.sav', 'rb'))
    x = np.array(x)
    x = x.reshape(1, -1)
    return kmeans.predict(x)

# Api ile istek atıp dataları alabiliriz.


def GetData(cardnumber):
    API_ENDPOINT = "https://apitest.kuveytturk.com.tr/prep/v1/cards/carddetail"
    headers = {
        'Content-Type': "application/json",
        'Authorization': "Bearer 5dbe96a0451c334e72b8665811e8ec1348ab72203a70f27d28a0b5ea99e215a3"
    }
    # data to be sent to api
    data = {
        "request": {
            "cardnumber": str(cardnumber)
        }
    }
    data = json.dumps(data)
    # sending post request and saving response as response object
    r = requests.post(url=API_ENDPOINT, headers=headers, data=data)
    pastebin_url = r.text
    pastebin_url = pastebin_url[9:-51]
    return json.loads(pastebin_url)

# Bu kısımda Api'dan aldığımız IsIntermTransactions'u işliyoruz.


def test():
    if pd.DataFrame.from_dict(json_normalize(newcustomer), orient='columns')[["CardProductCode", "IsIntermTransactions"]].values[0][1] == True:
        return 1
    else:
        return 0

# Bu kısımda Api'dan aldığımız CardProductCode'u işliyoruz.


def labelencoder(newcustomer):
    labelencoder = pickle.load(open('labelencoder.sav', 'rb'))
    x = [pd.DataFrame.from_dict(json_normalize(newcustomer), orient='columns')[["CardProductCode", "IsIntermTransactions"]].values[0][0]]
    return labelencoder.transform(x)


# Bu kısımda gruplama yapacağız, şube verileri api'dan çekilemediği için bize verilen data ile eşleştirdik.
def GetGroup(newcustomer, cardnumber):
    print(cardnumber)
    sube = [190, 181, 31, 31, 26, 25, 24, 23, 22, 21]
    if cardnumber == "4025900213283250":
        array = np.array([labelencoder(newcustomer), 15, test()])
    elif cardnumber == "4025900288779450":
        array = np.array([labelencoder(newcustomer), 114, test()])
    elif cardnumber == "4025900418055430":
        array = np.array([labelencoder(newcustomer), 219, test()])
    elif cardnumber == "4025900517881690":
        array = np.array([labelencoder(newcustomer), 292, test()])
    else:
        array = np.array([labelencoder(newcustomer), sube[randint(0, len(sube) - 1)], test()])
    return groupprediction(array)

# Transactions verisini işlemek için ön işleme


def dataMonth(data):
    data = data.resample("M").sum()
    data["Date"] = data.index
    data.index = range(0, len(data))
    return data

# Transactions verisini fit ettiriyoruz.


def Fitting(df):
    my_model = Prophet()
    my_model.fit(df)

    future_dates = my_model.make_future_dataframe(periods=3, freq="M")
    forecast = my_model.predict(future_dates)

    forecastnew = forecast['ds']
    forecastnew2 = forecast['yhat']
    forecastnew = pd.concat([forecastnew, forecastnew2], axis=1)
    forecastnew = forecastnew[len(forecastnew) - 3:]
    return forecastnew

# Time series verisinin çıktısını alıyoruz.


def TimeSeries(datapath):
    datax = pd.read_excel(datapath, date_parser=[0])
    dataAlisveris = datax[datax.Hizmet == "Alışveriş"].copy()
    dataAlisveris.drop(["CardNumber", "Hizmet"], axis=1, inplace=True)
    dataAlisveris.set_index("Date", inplace=True)
    dataYemek = datax[datax.Hizmet == "Yemek"][["Date", "Prices"]].copy()
    dataYemek.set_index("Date", inplace=True)
    dataGiyim = datax[datax.Hizmet == "Giyim"][["Date", "Prices"]].copy()
    dataGiyim.set_index("Date", inplace=True)
    dataFatura = datax[datax.Hizmet == "Fatura"][["Date", "Prices"]].copy()
    dataFatura.set_index("Date", inplace=True)
    dataHizmet = datax[datax.Hizmet == "Hizmet"][["Date", "Prices"]].copy()
    dataHizmet.set_index("Date", inplace=True)
    dataAlisveris = dataMonth(dataAlisveris)
    dataYemek = dataMonth(dataYemek)
    dataGiyim = dataMonth(dataGiyim)
    dataFatura = dataMonth(dataFatura)
    dataHizmet = dataMonth(dataHizmet)
    dfAlisveris = dataAlisveris.rename(columns={'Date': 'ds', 'Prices': 'y'})
    dfYemek = dataYemek.rename(columns={'Date': 'ds', 'Prices': 'y'})
    dfGiyim = dataGiyim.rename(columns={'Date': 'ds', 'Prices': 'y'})
    dfFatura = dataFatura.rename(columns={'Date': 'ds', 'Prices': 'y'})
    dfHizmet = dataHizmet.rename(columns={'Date': 'ds', 'Prices': 'y'})
    dfAlisveris = Fitting(dfAlisveris)
    dfYemek = Fitting(dfYemek)
    dfGiyim = Fitting(dfGiyim)
    dfFatura = Fitting(dfFatura)
    dfHizmet = Fitting(dfHizmet)
    return dataYemek, dataGiyim, dataFatura, dataHizmet, dataAlisveris, dfYemek, dfGiyim, dfFatura, dfHizmet, dfAlisveris

# Twitter verisi çekilip, sentimental analysis ile çıktıları alınıyor.


def Twitter(username="kuveytturk"):
    with open('turkce-stop-words.txt') as file:
        stw = file.read()
    stw = stw.split()
    stw = [s.lower() for s in stw]
    stop = stw

    # Modellerin İçeri Alınması
    kokbul1 = stemmer('turkish')
    filename = 'model.sav'
    filenamev2 = 'vectorizer.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    loaded_vectorizer = pickle.load(open(filenamev2, 'rb'))

    def preprocessing(text):
        text = text.lower()
        # get rid of non-alphanumerical characters
        text = re.sub(r'\W', ' ', text)
        # get rid of spaces
        text = re.sub(r'\s+', ' ', text)
        # Correct mistakes
        # and do the stemming
        return " ".join([word for word in kokbul1.stemWords(text.split()) if word not in stop])

    def predicting(x):
        test_sample = []
        for i in range(len(x)):
            test_sample.append(preprocessing(x[i]))
        sample = loaded_vectorizer.transform(test_sample).toarray()
        result = loaded_model.predict(sample)
        return result

    def Graph(result):
        labels = ['Olumsuz', 'Nötr', 'Olumlu']
        sizes = [(result == 0).sum(), (result == 1).sum(), (result == 2).sum()]
        colors = ['Red', 'gold', 'yellowgreen']
        patches, texts = plt.pie(sizes, colors=colors, shadow=True, startangle=90)
        plt.legend(patches, labels, loc="best")
        plt.axis('equal')
        plt.tight_layout()
        return plt

    def GetTweets(username):
        # Twitter API Settings
        auth = tweepy.OAuthHandler("b31NqruIj0m3D6mzOk4glEfz7", "yAku57PMlQ9V6MVxUrrzkGxI4izrwGCzvI8Q5OwPwyFeLCR0oT")
        auth.set_access_token("352938901-hH3mCRnw7ir8acB7oFQwfsu9gaboZeu20Hbm2jWi", "t2vYixvZemUibVI95QuapcqwEUCITki7xWFK6DjLTvGce")
        api = tweepy.API(auth)

        # Get Tweets
        tweets = []
        fetched_tweets = api.user_timeline(screen_name=username, count=100, include_rts=True)

        for tweet in fetched_tweets:
            tweets.append(preprocessing(tweet.text))

        sentiment = {'olumsuz': 0, 'notr': 1, 'olumlu': 2}
        sentimentters = {0: 'Olumsuz', 1: 'Notr', 2: 'Olumlu'}
        inv_sentiment = {v: k for k, v in sentiment.items()}

        result = predicting(tweets)

        sentimentters = {0: 'Olumsuz', 1: 'Notr', 2: 'Olumlu'}
        x, y, c = [], [], []

        x = list(pd.DataFrame(result)[0].map(sentimentters).value_counts().values)
        y = list(pd.DataFrame(result)[0].map(sentimentters).value_counts().index)
        c = pd.DataFrame(x, y, columns=[0]).T
        return c
    c = GetTweets(username)
    return c


# Test Çekimi
newcustomer = GetData("4025900288779450")
print("Test Class-0", GetGroup(newcustomer, "4025900288779450"))

# TEST MÜŞTERİLERİ VERİLERİ
# 4025900418055430 - Class 3
# 4025900213283250 - Class 2
# 4025900288779450 - Class 0
# 4025900517881690 - Class 1

# CLASS AÇIKLAMALARI
# Grup 0 Öğrenciler
# Grup 3 Emekli-Esnaf
# Grup 1 Beyaz-Altın Yakalılar
# Grup 2 Mavi Yakalılar

# Transactions datası işleniyor, bu veri localden çekiliyor, çünkü kredi kartı verileri işleniyor.
dataYemek, dataGiyim, dataFatura, dataHizmet, dataAlisveris, dfYemek, dfGiyim, dfFatura, dfHizmet, dfAlisveris = TimeSeries("number2.xlsx")


def TwitterEnCok(twitterdata):
    return twitterdata.columns[list(twitterdata.iloc[0]).index(sorted(twitterdata.iloc[0])[-2])]


# Test Twitter Verileri
twitterdata = Twitter("uzay00")
print("Twitter Test", TwitterEnCok(twitterdata))


def Checker(datanormal, datatahmin):
    return datanormal.Prices.values[-1] > datatahmin.yhat.values[0]


pathlist = ["number2.xlsx", "number3.xlsx", "number4.xlsx", "number5.xlsx"]
for i in pathlist:
    path = i
    dataYemek, dataGiyim, dataFatura, dataHizmet, dataAlisveris, dfYemek, dfGiyim, dfFatura, dfHizmet, dfAlisveris = TimeSeries(datapath=path)

    print({"Dosya": path,
           "Yemek": Checker(dataYemek, dfYemek),
           "Alışveriş": Checker(dataAlisveris, dfAlisveris),
           "Fatura": Checker(dataFatura, dfFatura),
           "Giyim": Checker(dataGiyim, dfGiyim),
           "Hizmet": Checker(dataHizmet, dfHizmet)})

cardnumber = "4025900288779450"
newcustomer = GetData(cardnumber)
Group = GetGroup(newcustomer, cardnumber)
#N2 - YEMEK-ALIŞVERİŞ-FATURA
#N3 - ALIŞVERİŞ-GİYİM-HİZMET
#N4 - ALIŞVERİŞ-GİYİM-HİZMET
#N5 - ALIŞVERİŞ-GİYİM-HİZMET


def degerlendir(Group):
    if path == "number2.xlsx" and Group == np.array([0]):
        return ["Yemek-Dominos.png", "Alisveris-Trendyol.png", "Fatura-İski.png"]
    elif path == "number2.xlsx" and Group == np.array([1]):
        return ["Yiyecek-Starbucks.png", "Alışveriş-Macrocenter.png", "Fatura-CKElektrik.png"]
    elif path == "number2.xlsx" and Group == np.array([2]):
        return ["Yiyecek-Arbys.png", "Alışveriş-A101.png", "Fatura-İski.png"]
    elif path == "number2.xlsx" and Group == np.array([3]):
        return ["Yemek-Erikli.png", "Alışveriş-A101.png", "Fatura-İski.png"]
    elif path == "number3.xlsx" and Group == np.array([0]):
        return ["Alışveriş-A101.png", "Giyim-Levis.png", "Hizmetler-Cinemaximum.png"]
    elif path == "number3.xlsx" and Group == np.array([1]):
        return ["Alışveriş-Macrocenter.png", "Giyim-Polo.png", "Hizmet-PO.png"]
    elif path == "number3.xlsx" and Group == np.array([2]):
        return ["Alışveriş-Migros.png", "Giyim-Levis.png", "Hizmet-MNG.png"]
    elif path == "number3.xlsx" and Group == np.array([3]):
        return ["Alışveriş-A101.png", "Giyim-LCWaikiki.png", "Hizmet-MNG.png"]
    elif path == "number4.xlsx" and Group == np.array([0]):
        return ["Alışveriş-Migros.png", "Giyim-Polo.png", "Hizmetler-Pegasus.png"]
    elif path == "number4.xlsx" and Group == np.array([1]):
        return ["Alışveriş-Macrocenter.png", "Giyim-Polo.png", "Hizmetler-THY.png"]
    elif path == "number4.xlsx" and Group == np.array([2]):
        return ["Hizmetler-Cinemaximum.png", "Giyim-Hotiç.png", "Hizmet-MNG.png"]
    elif path == "number4.xlsx" and Group == np.array([3]):
        return ["Alışveriş-A101.png", "Giyim-LCWaikiki.png", "Hizmet-MNG.png"]
    elif path == "number5.xlsx" and Group == np.array([0]):
        return ["Alışveriş-Migros.png", "Giyim-Polo.png", "Hizmetler-Pegasus.png"]
    elif path == "number5.xlsx" and Group == np.array([1]):
        return ["Alışveriş-Macrocenter.png", "Giyim-Polo.png", "Hizmetler-THY.png"]
    elif path == "number5.xlsx" and Group == np.array([2]):
        return ["Hizmetler-Cinemaximum.png", "Giyim-Hotiç.png", "Hizmet-MNG.png"]
    elif path == "number5.xlsx" and Group == np.array([3]):
        return ["Alışveriş-A101.png", "Giyim-LCWaikiki.png", "Hizmet-MNG.png"]


degerlendir(Group)
