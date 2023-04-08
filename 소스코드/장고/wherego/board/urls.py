from django.urls import path

from . import views
app_name = "board" 

urlpatterns = [
    path('<int:page>/', views.index, name='index'),
    path('detail/<int:id>/<int:num>/', views.detail, name="detail"),
    path('write/', views.write, name="write"),
    path('update/<int:id>/', views.update, name="update"),
    path('delete/<int:id>/', views.delete, name="delete")
]
