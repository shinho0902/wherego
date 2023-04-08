from django.urls import path

from . import views


app_name = "route" 
urlpatterns = [
    path('', views.index, name='index'),
    path('center/', views.center, name='center'),
]