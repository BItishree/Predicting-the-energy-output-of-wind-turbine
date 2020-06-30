from django.urls import path
from . import views

urlpatterns = [
    path('',views.myform, name='myform'),

]