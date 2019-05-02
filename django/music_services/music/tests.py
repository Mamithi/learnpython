import json
from django.urls import reverse
from rest_framework.test import APITestCase, APIClient
from rest_framework.views import status
from .models import Songs
from .serializers import SongsSerializer

# Create your tests here.
class SongsModelTest(APITestCase):
    def setUp(self):
        self.song = Songs.objects.create(
            title="The anthem",
            artist="People"
        )

    def test_song(self):
        self.assertEqual(self.song.title, "The anthem")
        self.assertEqual(self.song.artist, "People")
        self.assertEqual(str(self.song), "The anthem - People")

class BaseViewTest(APITestCase):
    client = APIClient()

    @staticmethod
    def create_song(title="", artist=""):
        if title != " " and artist != " ":
            Songs.objects.create(title=title, artist=artist)

    def make_a_request(self, kind="post", **kwargs):
        if kind== "post":
            return self.client.post(
                reverse(
                    "songs-list-create",
                    kwargs={
                        "version": kwargs["version"]
                    }
                ),
                data = json.dumps(kwargs["data"]),
                content_type='application/json'
            )
        elif kind == "put":
            return self.client.put(
                reverse(
                    "songs-detail",
                    kwargs={
                        "version": kwargs["version"],
                        "pk": kwargs["id"]
                    }
                ),
                data=json.dumps(kwargs["data"]),
                content_type='application/json'
            )
        else:
            return None

    def fetch_a_song(self, pk=0):
        return self.client.get(
            reverse(
                "songs-detail",
                kwargs={
                    "version": "v1",
                    "pk": pk
                }
            )
        )

    def delete_a_song(self, pk=0):
        return self.client.delete(
            reverse(
                "songs-detail",
                kwargs={
                    "version": "v1",
                    "pk": pk
                }
            )
        )

    def setUp(self):
        # add test data
        self.create_song("like glue", "sean paul")
        self.create_song("simple song", "konshens")
        self.create_song("love is wicked", "brick and lace")
        self.create_song("jam rock", "damien marley")

        self.valid_data = {
            "title": "test_song",
            "artist": "test artist"
        }

        self.invalid_data = {
            "title": "",
            "artist": ""
        }

        self.valid_song_id = 1
        self.invalid_song_id = 100

class GetAllSongsTest(BaseViewTest):
    def test_get_all_songs(self):
        # hit the API endpoint
        url = reverse("songs-list-create", kwargs={"version": "v1"})
        response = self.client.get(url)

        # fetch the data from db
        expected = Songs.objects.all()
        serialized = SongsSerializer(expected, many=True)
        self.assertEqual(response.data, serialized.data)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

class GetASingleSongsTest(BaseViewTest):
    def test_get_a_song(self):

        # Testing with a song that exists
        response = self.fetch_a_song(self.valid_song_id)
        expected = Songs.objects.get(pk=self.valid_song_id)
        serialized = SongsSerializer(expected)
        self.assertEqual(response.data, serialized.data)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        # Testing with a song that does not exist
        response = self.fetch_a_song(self.invalid_song_id)
        self.assertEqual(
            response.data["message"],
            "Song with id: 100 does not exist"
        )
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

class AddSongsTest(BaseViewTest):
    def test_create_a_song(self):
        response = self.make_a_request(
            kind="post",
            version="v1",
            data=self.valid_data
        )
        self.assertEqual(response.data, self.valid_data)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        response = self.make_a_request(
            kind="post",
            version="v1",
            data=self.invalid_data
        )
        self.assertEqual(
            response.data["message"],
            "Both title and artist are required to add a song"
        )

class UpdateSongsTest(BaseViewTest):
    def test_update_a_song(self):
        response =  self.make_a_request(
            kind="put",
            version="v1",
            id=2,
            data=self.valid_data
        )

        self.assertEqual(response.data, self.valid_data)
        self.assertEqual(response.status, status.HTTP_200_OK)

        response = self.make_a_request(
            kind="put",
            version="v1",
            id=3,
            data=self.invalid_data
        )

        self.assertEqual(
            response.data["message"],
            "Both title and artist are required to add a song"
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

class DeleteSongsTest(BaseViewTest):
    def test_delete_a_song(self):
        response = self.delete_a_song(1)
        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)

        response = self.delete_a_song(100)
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
