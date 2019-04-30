from django.urls import reverse
from rest_framework.test import APITestCase, APIClient
from rest_framework.views import status
from .models import Songs
from .serializers import SongsSerializer

# Create your tests here.
class BaseViewTest(APITestCase):
    client = APIClient()

    @staticmethod
    def create_song(title="", artist=""):
        if title != "" and artist != "":
            Songs.objects.create(title=title, artist=artist)

    def setUp(self):
        # add test data
        self.create_song("like glue", "sean paul")
        self.create_song("simple song", "konshens")
        self.create_song("love is wicked", "brick and lace")
        self.create_song("jam rock", "damien marley")

class GetAllSongsTest(BaseViewTest):
    def test_get_all_songs(self):
        # hit the API endpoint
        url = reverse("songs-all", kwargs={'version': 'v1'})
        response = self.client.get(url)

        # fetch the data from db
        expected = Songs.objects.all()
        serialized = SongsSerializer(expected, many=True)
        self.assertEqual(response.data. serialized.data)
        self.assertEqual(response.status_code. status.HTTP_200_OK)