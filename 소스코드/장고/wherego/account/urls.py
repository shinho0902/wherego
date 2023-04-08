from django.urls import path

from . import views

app_name = "account" 

urlpatterns = [
    path('', views.signup, name='signup'),
    path('login/', views.signin, name="login"),
    path('agree/', views.agree, name="agree"),
    path('idcheck/', views.id_check,name="id_check"),
    path('emailcheck/', views.email_check,name="email_check"),
    path('logout/', views.logout,name="logout"),
    path('email/', views.email,name="email"),
    path('kakao/', views.kakao_login,name="kakao-login"),
    path('kakao/callback/', views.kakao_callback,name="kakao_callback"),
    path('autoid/', views.autoid,name="autoid"),
    path('mypage/', views.mypage,name="mypage"),
    path('deleteid/', views.deleteid,name="deleteid"),
    path('outgroup/', views.outgroup,name="outgroup"),
    path('makegroup/', views.makegroup,name="makegroup"),
    path('activate/<str:uidb64>/<str:token>', views.Activate.as_view())
]