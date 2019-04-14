from flask import Flask, render_template, make_response, request, redirect

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
import random
import os
import sys
from sklearn.datasets import load_files
nltk.download('stopwords')
from snowballstemmer import stemmer

app = Flask(__name__)

##############################################


def Twitter(username="kuveytturk"):
    with open('turkce-stop-words.txt', encoding='latin') as file:
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
##############################################


def groupprediction(x):
    kmeans = pickle.load(open('kmeans.sav', 'rb'))
    x = np.array(x)
    x = x.reshape(1, -1)
    return kmeans.predict(x)


def GetData(cardnumber):
    API_ENDPOINT = "https://apitest.kuveytturk.com.tr/prep/v1/cards/carddetail"
    headers = {
        'Content-Type': "application/json",
        'Authorization': "Bearer 54602e2f2b5264496a4d103c2444f69621484fdbbc23e4e7e6f46cbf01c27172   "
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


def test():
    global newcustomer
    if pd.DataFrame.from_dict(json_normalize(newcustomer), orient='columns')[["CardProductCode", "IsIntermTransactions"]].values[0][1] == True:
        return 1
    else:
        return 0


def labelencoder(newcustomer):
    labelencoder = pickle.load(open('labelencoder.sav', 'rb'))
    x = [pd.DataFrame.from_dict(json_normalize(newcustomer), orient='columns')[["CardProductCode", "IsIntermTransactions"]].values[0][0]]
    return labelencoder.transform(x)


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


def dataMonth(data):
    data = data.resample("M").sum()
    data["Date"] = data.index
    data.index = range(0, len(data))
    return data


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


@app.route('/')
def entry_page()->'html':
    return render_template('search.html')


@app.route('/', methods=['POST'])
def twitter_page()->'html':
    global newcustomer
    clientId = request.form["customerid"]
    if clientId != '4025900418055430':
        return render_template('search.html')
    else:
        temp = make_response(redirect('/twitter'))
        temp.set_cookie("clientId", str(clientId))
        return temp


@app.route('/twitter')
def twitter_login_page()->'html':
    return render_template('twitter.html')


@app.route('/ana-sayfaya-git')
def main_page()->'html':
    return redirect('/')


@app.route('/twitter', methods=['POST', 'GET'])
def twitter_login_post_page()->'html':
    if request.form.get('cancel') == 'cancel':
        return redirect('/')
    else:
        return redirect('/kuveytturk')


@app.route('/kuveytturk', methods=['GET', 'POST'])
def kuveytturk_page()->'html':
    clientId = request.cookies.get('clientId')
    global newcustomer
    newcustomer = GetData(cardnumber=str(clientId))
    group_info = GetGroup(newcustomer, str(clientId))
    random_file = str(random.randint(2, 5))
    dataYemek, dataGiyim, dataFatura, dataHizmet, dataAlisveris, dfYemek, dfGiyim, dfFatura, dfHizmet, dfAlisveris = TimeSeries("number" + random_file + ".xlsx")
    real_food_dict = dataYemek.to_dict()
    predict_food_dict = dfYemek.to_dict()
    real_cloth_dict = dataGiyim.to_dict()
    predict_cloth_dict = dfGiyim.to_dict()
    real_bill_dict = dataFatura.to_dict()
    predict_bill_dict = dfFatura.to_dict()
    real_service_dict = dataHizmet.to_dict()
    predict_service_dict = dfHizmet.to_dict()
    real_shop_dict = dataAlisveris.to_dict()
    predict_shop_dict = dfAlisveris.to_dict()

    def degerlendir(Group, path):
        if path == "number2.xlsx" and Group == np.array([0]):
            return ["Yemek-Dominos.png", "Alisveris-Trendyol.png", "Fatura-İski.png"]
        elif path == "number2.xlsx" and Group == np.array([3]):
            return ["Yiyecek-Starbucks.png", "Alışveriş-Macrocenter.png", "Fatura-CKElektrik.png"]
        elif path == "number2.xlsx" and Group == np.array([2]):
            return ["Yiyecek-Arbys.png", "Alışveriş-A101.png", "Fatura-İski.png"]
        elif path == "number2.xlsx" and Group == np.array([1]):
            return ["Yemek-Erikli.png", "Alışveriş-A101.png", "Fatura-İski.png"]
        elif path == "number3.xlsx" and Group == np.array([0]):
            return ["Alışveriş-A101.png", "Giyim-Levis.png", "Hizmetler-Cinemaximum.png"]
        elif path == "number3.xlsx" and Group == np.array([3]):
            return ["Alışveriş-Macrocenter.png", "Giyim-Polo.png", "Hizmet-PO.png"]
        elif path == "number3.xlsx" and Group == np.array([2]):
            return ["Alışveriş-Migros.png", "Giyim-Levis.png", "Hizmet-MNG.png"]
        elif path == "number3.xlsx" and Group == np.array([1]):
            return ["Alışveriş-A101.png", "Giyim-LCWaikiki.png", "Hizmet-MNG.png"]
        elif path == "number4.xlsx" and Group == np.array([0]):
            return ["Alışveriş-Migros.png", "Giyim-Polo.png", "Hizmetler-Pegasus.png"]
        elif path == "number4.xlsx" and Group == np.array([3]):
            return ["Alışveriş-Macrocenter.png", "Giyim-Polo.png", "Hizmetler-THY.png"]
        elif path == "number4.xlsx" and Group == np.array([2]):
            return ["Hizmetler-Cinemaximum.png", "Giyim-Hotiç.png", "Hizmet-MNG.png"]
        elif path == "number4.xlsx" and Group == np.array([1]):
            return ["Alışveriş-A101.png", "Giyim-LCWaikiki.png", "Hizmet-MNG.png"]
        elif path == "number5.xlsx" and Group == np.array([0]):
            return ["Alışveriş-Migros.png", "Giyim-Polo.png", "Hizmetler-Pegasus.png"]
        elif path == "number5.xlsx" and Group == np.array([3]):
            return ["Alışveriş-Macrocenter.png", "Giyim-Polo.png", "Hizmetler-THY.png"]
        elif path == "number5.xlsx" and Group == np.array([2]):
            return ["Hizmetler-Cinemaximum.png", "Giyim-Hotiç.png", "Hizmet-MNG.png"]
        elif path == "number5.xlsx" and Group == np.array([1]):
            return ["Alışveriş-A101.png", "Giyim-LCWaikiki.png", "Hizmet-MNG.png"]

    def TwitterEnCok(twitterdata):
        return twitterdata.columns[list(twitterdata.iloc[0]).index(sorted(twitterdata.iloc[0])[-2])]

    if TwitterEnCok(Twitter("nuhuzmert")) == 'Olumsuz':
        ozel_indirim = True
    else:
        ozel_indirim = False

    return render_template('index.html',
                           gercek_yemek_tarih=list(map(lambda x: str(x)[0:10], real_food_dict['Date'].values())),
                           gercek_yemek_deger=list(real_food_dict['Prices'].values()),
                           tahmin_yemek_tarih=list(map(lambda x: str(x)[0:10], predict_food_dict['ds'].values())),
                           tahmin_yemek_deger=list(predict_food_dict['yhat'].values()),
                           gercek_giyim_tarih=list(map(lambda x: str(x)[0:10], real_cloth_dict['Date'].values())),
                           gercek_giyim_deger=list(real_cloth_dict['Prices'].values()),
                           tahmin_giyim_tarih=list(map(lambda x: str(x)[0:10], predict_cloth_dict['ds'].values())),
                           tahmin_giyim_deger=list(predict_cloth_dict['yhat'].values()),
                           gercek_fatura_tarih=list(map(lambda x: str(x)[0:10], real_bill_dict['Date'].values())),
                           gercek_fatura_deger=list(real_bill_dict['Prices'].values()),
                           tahmin_fatura_tarih=list(map(lambda x: str(x)[0:10], predict_bill_dict['ds'].values())),
                           tahmin_fatura_deger=list(predict_bill_dict['yhat'].values()),
                           gercek_hizmet_tarih=list(map(lambda x: str(x)[0:10], real_service_dict['Date'].values())),
                           gercek_hizmet_deger=list(real_service_dict['Prices'].values()),
                           tahmin_hizmet_tarih=list(map(lambda x: str(x)[0:10], predict_service_dict['ds'].values())),
                           tahmin_hizmet_deger=list(predict_service_dict['yhat'].values()),
                           gercek_alisveris_tarih=list(map(lambda x: str(x)[0:10], real_shop_dict['Date'].values())),
                           gercek_alisveris_deger=list(real_shop_dict['Prices'].values()),
                           tahmin_alisveris_tarih=list(map(lambda x: str(x)[0:10], predict_shop_dict['ds'].values())),
                           tahmin_alisveris_deger=list(predict_shop_dict['yhat'].values()),
                           resimler=degerlendir(group_info, "number" + random_file + ".xlsx"),
                           ozelindirim=ozel_indirim
                           )


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=37000)
