from django.shortcuts import render
from django.conf import settings
# Create your views here.
def index(request):
    return render(request, 'main/index.html')

def center(request):
    client_id = settings.NAVER_API
    context = {
            'API_KEY': client_id,
        }
    return render(request, 'main/center.html', context)
