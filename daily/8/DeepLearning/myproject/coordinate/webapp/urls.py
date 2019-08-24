from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^getCoord$', views.getCoord, name='getCoord'),
]
