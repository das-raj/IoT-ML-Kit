from django.contrib import admin
from django.urls import path, include
from . import views

app_name = "website"

urlpatterns = [
	path('', views.index, name='index'),
	path('form/', views.input_form, name='form'),
	# path('opt/event/', views.uploadEvents, name='opt-event'),
	# path('opt/announcement/', views.uploadAnn, name='opt-announcement'),
	# path('opt/placement/', views.uploadPlacement, name='opt-placement'),
	# path('', views.indexpage, name='index'),
	# path('login/', views.LoginFormView, name='index'),
	# path('announcement/', views.announcement, name='announcement'),
	# path('opt/announcement/', views.announcement, name='opt-announcement'),
	# path('event/', views.event, name='event'),
	# path('opt/event/', views.event, name='opt-event'),
]
