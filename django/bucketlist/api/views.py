from django.shortcuts import render
from rest_framework import viewsets, status
from .models import Bucketlist
from .serializers import BucketlistSerializer
from rest_framework.response import Response
from rest_framework.decorators import action
from django.shortcuts import get_object_or_404


# Create your views here.
class BucketlistViewSet(viewsets.ViewSet):
    def create(self, request):
        serializer = BucketlistSerializer(data=request.data)
        if serializer.is_valid():
            payload = serializer.save()

            return Response({
                'status': status.HTTP_201_CREATED,
                'msg': '{} created'.format(payload.name),
                'data': serializer.data
            }, status.HTTP_201_CREATED)
        else:
            return Response(serializer.errors, status.HTTP_400_BAD_REQUEST)

    @staticmethod
    def list(self):
        queryset = Bucketlist.objects.all()
        serializer = BucketlistSerializer(queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None):
        queryset = Bucketlist.objects.all()
        data = get_object_or_404(queryset, pk=pk)
        serializer = BucketlistSerializer(data)
        return Response(serializer.data)

    def update(self, request, pk=None):
        bucketlist = get_object_or_404(Bucketlist.objects.all(), pk=pk)
        data = request.data
        serializer = BucketlistSerializer(instance=bucketlist, data=data, partial=True)
        if serializer.is_valid(raise_exception=True):
            bucketlist_updated = serializer.save()
            return Response({
                'status': status.HTTP_200_OK,
                'msg': "'{}' data has been updated successfully".format(bucketlist_updated.name),
                'bucketlist': serializer.data
            })
        else:
            return Response(serializer.errors, status.HTTP_400_BAD_REQUEST)

    def destroy(self, request, pk=None):
        bucketlist = get_object_or_404(Bucketlist.objects.all(), pk=pk)
        bucketlist.delete()
        return Response({
            'status': status.HTTP_204_NO_CONTENT,
            'msg': "{} deleted".format(bucketlist.name)
        }, status.HTTP_204_NO_CONTENT)
