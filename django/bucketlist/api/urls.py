from django.urls import path, include
from . import views
from rest_framework import routers
from rest_framework.urlpatterns import format_suffix_patterns

router = routers.DefaultRouter()

router.register(r'bucketlist', views.BucketlistViewSet, base_name="create")

urlpatterns = [
    path('', include(router.urls))
]

# urlpatterns = format_suffix_patterns(urlpatterns)
