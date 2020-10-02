from django.test import TestCase

# Create your tests here.
from django.test import TestCase
from .models import Bucketlist
from rest_framework.test import APIClient
from django.urls import reverse
from rest_framework import status


class ModelTestCase(TestCase):
    def setUp(self):
        self.bucketlist_name = "Learn testing",
        self.bucketlist = Bucketlist(name=self.bucketlist_name)

    def test_model_can_create_a_bucketlist(self):
        old_count = Bucketlist.objects.count()
        self.bucketlist.save()
        new_count = Bucketlist.objects.count()
        self.assertNotEqual(old_count, new_count)


class ViewTestCase(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.bucketlist_data = {'name': 'Go home'}
        self.response = self.client.post(
            'api/bucketlist',
            self.bucketlist_data,
            format="json"
        )

    def test_api_can_create_a_bucketlist(self):
        self.assertEqual(self.response.status_code, status.HTTP_201_CREATED)

    def test_api_can_get_a_bucketlist(self):
        bucketlist = Bucketlist.objects.get()
        response = self.client.get(
            'bucketlist', kwargs={'pk': bucketlist.id},
            format="json"
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertContains(response, bucketlist)

    def test_api_can_update_bucketlist(self):
        bucketlist = Bucketlist.objects.get()
        change_bucketlist = {'name': 'Something new'}
        res = self.client.put(
            reverse('bucketlist', kwargs={'pk': bucketlist.id}),
            change_bucketlist, format="json"
        )

        self.assertEqual(res.status_code, status.HTTP_200_OK)

    def test_api_can_delete_bucketlist(self):
        bucketlist = Bucketlist.objects.get()
        response = self.client.delete(
            reverse('bucketlist', kwargs={'pk': bucketlist.id}),
            format='json', follow=True
        )

        self.assertEquals(response.status_code, status.HTTP_204_NO_CONTENT)
