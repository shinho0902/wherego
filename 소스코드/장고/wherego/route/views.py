from django.shortcuts import render
from django.conf import settings
# Create your views here.
import pandas as pd
import math
from haversine import haversine
import json
from decimal import Decimal
from django.http import JsonResponse  
# Create your views here.

client_id = settings.NAVER_API
odsay_key=settings.ODSAY_API

def index(request):
    data = pd.read_csv("route/static/csv/station_coordinate.csv")
    if request.method == 'POST':
        dots = []
        lons = request.POST.getlist('lon')
        lats = request.POST.getlist('lat')
        center_point_lng = Decimal('0.0')
        center_point_lat = Decimal('0.0')
        for i in range(len(lons)):
            center_point_lng += Decimal(lons[i])
            center_point_lat += Decimal(lats[i])
            dots.append((float(Decimal(lats[i])),float(Decimal(lons[i]))))

        center_point_lng = center_point_lng/Decimal(len(lons))
        center_point_lat = center_point_lat/Decimal(len(lons))
        center_point = (center_point_lat,center_point_lng)
        subset = data[['lat', 'lng']]
        tuples = [tuple(x) for x in subset.values]
        data["coordinates"] = tuples
        
        data = data.dropna(axis=0)
        
        data["distance"] = data.apply(lambda row:haversine(center_point,row["coordinates"]),axis=1)
        
        subway_list=data.nsmallest(1, 'distance').name
        subway_df=data.nsmallest(3, 'distance')
        subway_df = subway_df.to_dict("records")
        subway_json = json.dumps(subway_df)

        
        request.session["center"]=[subway_df[0]['lat'],subway_df[0]['lng']]
        request.session["center_name"] = list(subway_list)[0]
        
        def ccw(p1, p2, p3):
            return p1[0]*(p2[1] - p3[1]) + p2[0]*(p3[1] - p1[1]) + p3[0]*(p1[1] - p2[1])

        def monotoneChain(dots):
            dots.sort()

            lower = []
            for d in dots:
                while len(lower) >= 2 and ccw(lower[-2], lower[-1], d) < 0:
                    lower.pop()
                lower.append(d)

            upper = []
            for d in reversed(dots):
                while len(upper) >= 2 and ccw(upper[-2], upper[-1], d) < 0:
                    upper.pop()
                upper.append(d)
            return lower[:-1] + upper[:-1]
        client = {}
        polygons = {}
        if len(lats) > 3:
            res = monotoneChain(dots)
            count = 1
            for x,y in res:
                polygons[str(count)] = [x,y]
                count+=1
        else:
            count = 1
            for x,y in dots:
                polygons[str(count)] = [x,y]
                count+=1
        
        count = 1
        for x,y in dots:
            client[str(count)] = [x,y]
            count+=1
        
        context = {
                'NAVER_API_KEY': client_id,
                'ODSAY_API_KEY': odsay_key,
                'subway_list': list(subway_list)[0],
                'subway_json':subway_json,
                'clients':json.dumps(client),
                'polygons':json.dumps(polygons)
            }

        
        return render(request,'route/index.html',context)
    else:
        context = {
                'NAVER_API_KEY': client_id,
                'ODSAY_API_KEY': odsay_key,
                'subway_list': {},
                'subway_json':{},
                'clients':{},
                'polygons':{}
            }
        return render(request,'route/index.html',context)
    
    # data = pd.read_csv("route\station_coordinate.csv")
    
    # center_point_lng = 127.10733328074403
    # center_point_lat = 37.3672570356506
    # center_point = (center_point_lat,center_point_lng)

    # # data["difference_x"] = data["lat"] - center_point_x
    # # data["difference_y"] = data["lng"] - center_point_y

    # # data["distance"] = (data["difference_x"]**2 + data["difference_y"]**2)
    # # data['distance'] = data['distance'].apply(math.sqrt)

    # # subway_list = list(data.nsmallest(3, 'distance').name)
    
    # subset = data[['lat', 'lng']]
    # tuples = [tuple(x) for x in subset.values]
    # data["coordinates"] = tuples
    
    # data = data.dropna(axis=0)
    
    # data["distance"] = data.apply(lambda row:haversine(center_point,row["coordinates"]),axis=1)
    
    # subway_list=data.nsmallest(3, 'distance').name
    
    # subway_df=data.nsmallest(3, 'distance')
    # subway_df = subway_df.to_dict("records")
    # subway_json = json.dumps(subway_df)
    
    # context = {
    #         'NAVER_API_KEY': naver_client_id,
    #         'ODSAY_API_KEY': odsay_key,
    #         'subway_list': subway_list,
    #         'subway_json':subway_json,
    #     }

    
    # return render(request,'route/index.html',context)



def center(request):
    client_id = settings.NAVER_API
    context = {
            'API_KEY': client_id,
        }
    return render(request, 'route/center.html', context)

