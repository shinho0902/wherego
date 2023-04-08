from django.shortcuts import render
from django.conf import settings
# Create your views here.
from .predict import pred_picture
import cv2
import numpy
import pandas as pd
import requests
from django.conf import settings
from haversine import haversine
import json
from django.http import HttpResponseRedirect
from django.urls import reverse

client_id = settings.NAVER_API
client_secret = settings.NAVER_SECRET

def index(request):
    return render(request, 'imageupload/index.html')


def result(request):
    if request.method == "POST":
        if request.session.get("center"):
            f = request.FILES["img"]
            myfile = f.read()
            image = cv2.imdecode(numpy.frombuffer(myfile , numpy.uint8), cv2.COLOR_BGR2RGB)
            dst1 = cv2.resize(image, (299, 299))
            img = numpy.array(dst1)
            result = pred_picture(img)

            # 좌표 (경도, 위도)
            coords = str(request.session.get("center")[1])+","+str(request.session.get("center")[0])
            output = "json"
            orders = 'addr'
            endpoint = "https://naveropenapi.apigw.ntruss.com/map-reversegeocode/v2/gc"
            url = f"{endpoint}?coords={coords}&output={output}&orders={orders}"

            # 헤더
            headers = {
                "X-NCP-APIGW-API-KEY-ID": client_id,
                "X-NCP-APIGW-API-KEY": client_secret,
            }

            # 요청
            res = requests.get(url, headers=headers)
            region = res.json()['results'][0]['region']['area2']["name"]
            stores = pd.read_csv("imageupload/static/csv/real_real_final_cafeList.csv")
            in_region_stores = stores.loc[(stores['address'].str.contains(region)) & (stores['y_pred'] == str(result))]
            in_region_stores.fillna("", inplace=True)
            if len(in_region_stores) >=2 :
                center_point = (request.session.get("center")[0],request.session.get("center")[1])
                subset = in_region_stores[['lat', 'lng']]
                tuples = [tuple(x) for x in subset.values]
                in_region_stores["coordinates"] = tuples
        
                in_region_stores["distance"] = in_region_stores.apply(lambda row:haversine(center_point,row["coordinates"]),axis=1)

                in_region_stores = in_region_stores.sort_values(by=["distance"])
                if len(in_region_stores) > 6 :
                    in_region_stores = in_region_stores.iloc[0:6]
                    in_region_stores = in_region_stores.to_dict('records')
                else:
                    in_region_stores = in_region_stores.iloc[:len(in_region_stores)]
                    in_region_stores = in_region_stores.to_dict('records')
                    
            elif len(in_region_stores)==1:
                in_region_stores = in_region_stores.to_dict('records')
            
            return render(request, 'imageupload/result.html',{'stores':in_region_stores, 'len_stores':len(in_region_stores),'label_result':str(result)})
