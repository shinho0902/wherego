from django.shortcuts import render
from django.conf import settings
from account.models import User, Group
from .models import Message
import pandas as pd
import io   
from .text_recsys import text_rec_run
from PIL import Image
import base64
from io import StringIO
import cv2
import numpy as np
import requests
from account.decorator import login_required


client_id = settings.NAVER_API
client_secret = settings.NAVER_SECRET

# Create your views here.
@login_required
def index(request):
    groups = Group.objects.filter(members__username__exact=request.session["user"]).all()
    members = ""
    for group in groups:
        members = "("
        users = group.members.all()
        for user in users:
            members += user.username+", "
        members = members[:-2]+")"
        group.memnames = members
        
    return render(request, 'chat/index.html',{"groups":groups})


def enterroom(request):
        group = Group.objects.filter(id=request.POST["group"]).all()
        group = group[0]
        request.session.modified = True
        messages = Message.objects.filter(room=group).all()
        users = group.members.all()
        return render(request, "chat/room.html", {"room_name":group.chattingid,
                                                "messages":messages,
                                                "users":users,
                                                "groupname":group.name})


def getaddress(coords):
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
    spot = res.json()['results'][0]['region']['area2']['name'].split()[1] + " " + res.json()['results'][0]['region']['area3']['name']
    return spot


def kakaochat(request):
    chat = request.FILES["kakaochat"].read().decode('utf-8', 'ignore')
    kakao = pd.read_csv(io.StringIO(chat),
                    sep='\t', engine='python', encoding='utf-8')

    coords = str(request.session.get("center")[1])+","+str(request.session.get("center")[0])
    spot = getaddress(coords)
    result = text_rec_run(spot, kakao)
    dfs = []
    for img, df in result:
        imgs = []
        for i in range(3):
            im = img[i].to_image()
            im = np.array(im)  
            opencv_image=cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
            opencv_image = cv2.resize(opencv_image,(1170,780))
            ret, frame_buff = cv2.imencode('.jpg', opencv_image)
            frame_b64 = base64.b64encode(frame_buff)
            imgs.append(frame_b64.decode("utf-8"))
            
        df["img"] = imgs
        dfs.append(df)
        
    result_list = []
    catego = ["식당", "카페", "술집", "가게"]
    for i in range(4):
        content = {'category':catego[i],
                   'content':dfs[i].to_dict('records')}
        result_list.append(content)
    return render(request, 'chat/result.html',{'storeses':result_list, 'len_stores':len(df)})


def groupchat(request):
    messages = Message.objects.filter(room__chattingid__exact=request.POST["chatid"]).order_by("date").all()
    messages_df = [f.message.strip() for f in messages]
    coords = str(request.session.get("center")[1])+","+str(request.session.get("center")[0])
    spot = getaddress(coords)
    result = text_rec_run(spot, messages_df, "group")
    dfs = []
    for img, df in result:
        imgs = []
        for i in range(3):
            im = img[i].to_image()
            im = np.array(im)  
            opencv_image=cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
            opencv_image = cv2.resize(opencv_image,(1170,780))
            ret, frame_buff = cv2.imencode('.jpg', opencv_image)
            frame_b64 = base64.b64encode(frame_buff)
            imgs.append(frame_b64.decode("utf-8"))
            
        df["img"] = imgs
        dfs.append(df)
        
    result_list = []
    catego = ["식당", "카페", "술집", "가게"]
    for i in range(4):
        content = {'category':catego[i],
                   'content':dfs[i].to_dict('records')}
        result_list.append(content)
    return render(request, 'chat/result.html',{'storeses':result_list, 'len_stores':len(df)})