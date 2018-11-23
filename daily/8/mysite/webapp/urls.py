from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^checkface$', views.checkface, name='checkface'),
    url(r'^addface$', views.addface, name='addface'),
    url(r'^listfaces$', views.listfaces, name='listfaces'),
    url(r'^removeface', views.removeface, name='removeface'),
]
