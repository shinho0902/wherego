from django.shortcuts import render,redirect
from django.http import HttpResponse, HttpResponseRedirect
from .models import User,Group
import bcrypt
from django.shortcuts import render
from django.contrib import messages
from .decorator import already_login,login_required
from django.http import JsonResponse   
from django.views import View
from django.core.exceptions import ValidationError
from django.core.validators import validate_email
from django.contrib.sites.shortcuts import get_current_site
from django.utils.http import urlsafe_base64_encode,urlsafe_base64_decode
from django.core.mail import EmailMessage
from django.utils.encoding import force_bytes, force_str
from .token import account_activation_token
from .text import message
from django.conf import settings
from django.core import serializers
from .generateroom import makeroom
import requests
import os
import json


kakao_login_api = settings.KAKAO_LOGIN
client_id = settings.NAVER_API
kakao_secret = settings.KAKAO_SECRET
# Create your views here.
@already_login
def signup(request):
    if request.method == 'POST':

        user = User.objects.create(username=request.POST.get('id'),password = bcrypt.hashpw(request.POST.get('password1').encode('utf-8'), bcrypt.gensalt()).decode(), 
                                   email = request.POST.get('email'), gender = request.POST.get('gender'),name=request.POST.get('username'), 
                                   birthday = request.POST.get('year')+"-"+request.POST.get('month')+"-"+request.POST.get('day'))
        user.save()

        current_site = get_current_site(request) 
        domain       = current_site.domain
        uidb64       = urlsafe_base64_encode(force_bytes(user.pk))
        token        = account_activation_token.make_token(user)
        message_data = message(domain, uidb64, token)

        mail_title = "이메일 인증을 완료해주세요"
        mail_to    = request.POST.get('email')
        email      = EmailMessage(mail_title, message_data, to=[mail_to])
        email.send()
        return redirect('account:email')

def email(request):
    return render(request, 'account/email.html')

@already_login
def agree(request):
    if request.method == 'POST':
        return render(request, 'account/signup.html')
    return render(request, 'account/agree.html')

@already_login
def signin(request):
    if request.method == 'POST':
        user_id = request.POST.get('username')
        user_pw = request.POST.get('password').encode('utf-8')
        try:
            user = User.objects.get(email = user_id)
            result = bcrypt.checkpw(user_pw,user.password.encode('utf-8'))
            if result:
                if user.is_active==False:
                    messages.warning(request,"이메일 인증이 완료되지 않았습니다.")
                else:
                    request.session['user'] = user.username
                    return redirect('main:index')
            else:
                messages.warning(request,"ID 혹은 비밀번호 오류입니다.")
        except:
            messages.warning(request,"등록된 이메일이 없습니다.")


    return render(request, "account/login.html")


def id_check(request):
    if request.method == 'POST':
        try:
            user = User.objects.get(username=request.POST.get('id'))
        except Exception as e:
            user = None
        if user:
            result = {
            'result':'no',
            }
        else:
            result = {
                'result':"ok",
            }
        return JsonResponse(result)


def email_check(request):
    if request.method == 'POST':
        try:
            user = User.objects.get(email=request.POST.get('email'))
        except Exception as e:
            user = None
        if user:
            result = {
            'result':'no',
            }
        else:
            result = {
                'result':"ok",
            }
    return JsonResponse(result)

@login_required
def logout(request):
    for key in list(request.session.keys()):
        del request.session[key]
    return redirect('main:index')


class Activate(View):
    def get(self, request, uidb64, token):
        try:
            uid  = force_str(urlsafe_base64_decode(uidb64))
            user = User.objects.get(pk=uid)
            
            if account_activation_token.check_token(user, token):
                user.is_active = True
                user.save()

                return redirect('account:login')
        
            return JsonResponse({"message" : "AUTH FAIL"}, status=400)

        except ValidationError:
            return JsonResponse({"message" : "TYPE_ERROR"}, status=400)
        except KeyError:
            return JsonResponse({"message" : "INVALID_KEY"}, status=400)


def makegroup(request):
    if request.method=="POST":
        friends = request.POST.getlist('friends')
        friends.append(request.session['user'])
        friends_list = []
        for friend in friends:
            friends_list.append(User.objects.get(username=friend.strip()))
        group = Group.objects.create(name=request.POST.get("groupname"), chattingid= makeroom())
        group.members.set(friends_list)
        group.save()
        return redirect('account:mypage')
    
    
def autoid(request):
    if request.method=="POST":
        try:
            user = User.objects.filter(username=request.POST.get('id')).all()
        except Exception as e:
            user = None
        if user:
            result = {
                'result': serializers.serialize('json', user),
            }
        else:
            result = {
                'result':[],
            }
        
        return JsonResponse(result)
    
@login_required
def mypage(request):
    user = User.objects.get(username = request.session["user"])
    groups = Group.objects.filter(members__username__exact=request.session["user"]).all()
    return render (request, "account/mypage.html", {"user":user, "groups":groups})
    
@login_required
def deleteid(request):
    user = User.objects.get(username = request.session["user"])
    user.delete()
    for key in list(request.session.keys()):
        del request.session[key]
    return redirect('main:index')
    
@login_required
def outgroup(request):
    group = Group.objects.get(pk = request.POST["deletegroupname"])
    group.members.remove(User.objects.get(username=request.session['user']))
    return redirect('account:mypage')
    
    
def kakao_login(request):
    REDIRECT_URI = "http://127.0.0.1:8000/account/kakao/callback"
    return redirect(
        f"https://kauth.kakao.com/oauth/authorize?client_id={kakao_login_api}&redirect_uri={REDIRECT_URI}&response_type=code"
    )
    
def kakao_callback(request):
    try:
        #print(request.GET)
        code            = request.GET.get("code")
        redirect_uri    = "http://127.0.0.1:8000/account/kakao/callback"
        token_request   = requests.get(
            f"https://kauth.kakao.com/oauth/token?grant_type=authorization_code&client_id={kakao_login_api}&client_secret={kakao_secret}&redirect_uri={redirect_uri}&code={code}"
        )
        token_json      = token_request.json()
        #print(token_json)

        error           = token_json.get("error", None)

        if error is not None:
            return render(request, "account/error.html" ,{"message":"카카오 로그인 오류"})  

        access_token    = token_json.get("access_token")
        
        user_info_response = requests.get('https://kapi.kakao.com/v2/user/me', headers={"Authorization": f'Bearer ${access_token}'})
        data = user_info_response.json()
        
        if "email" in data["kakao_account"].keys():
            user = User.objects.filter(email = data["kakao_account"]["email"])
            if user:
                user = user.first()
                if user.is_active==False:
                    return render(request, "account/error.html" ,{"message":"이메일 인증이 완료되지 않았습니다."}) 
                else:
                    request.session['user'] = user.username
                    return redirect('main:index')
            else:
                return render(request, "account/error.html" ,{"message":"해당 카카오메일로 가입한 계정이 없습니다."}) 
        else:
             return render(request, "account/error.html" ,{"message":"카카오 이메일 제공 동의를 진행해주세요."})  
        
    except KeyError:
         return render(request, "account/error.html" ,{"message":"카카오 로그인 오류"})      
            