from django.urls import path
from . import views

urlpatterns = [
    path("webcam/", views.webcam, name="webcam"),
    path(
        "capture_and_recommend/",
        views.capture_and_recommend,
        name="capture_and_recommend"
    ),
]
