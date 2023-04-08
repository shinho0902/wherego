
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('route/', include('route.urls')),
    path('chat/', include('chat.urls')),
    path('account/', include('account.urls')),
    path('board/', include('board.urls')),
    path('imageupload/', include('imageupload.urls')),
    path('', include('main.urls')),
]

from django.conf.urls.static import static
from django.conf import settings

# MEDIA_URL로 들어오는 요청에 대해 MEDIA_ROOT 경로를 탐색한다.
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)