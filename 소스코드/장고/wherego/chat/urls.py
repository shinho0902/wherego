from django.urls import path

from . import views
app_name = "chat" 

urlpatterns = [
    path('', views.index, name='index'),
    path('enterroom/', views.enterroom, name="enterroom"), 
    path('groupchat/', views.groupchat, name="groupchat"),
    path('kakaochat/', views.kakaochat, name="kakaochat"),
]
