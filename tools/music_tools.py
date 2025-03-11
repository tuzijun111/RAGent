import requests
import base64
import time
import os
from collections import Counter
from smolagents import Tool


class MusicTool(Tool):
    name = "get_top_genres_by_country"
    description = "Gets the most common music genres in a country using Spotify APIs and then returns them as comma-separated list string."
    inputs = {'country': {'type': 'string', 'description': 'The name of the country to get the genres for (e.g., "Argentine").'}}
    output_type = "string"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.access_token, self.token_expiration = self._get_spotify_token()
        print(self.access_token)
        self.is_initialized = False

    def forward(self, country: str) -> str:
        results = self.get_top_genres_by_country(country)
        if len(results) == 0:
            raise Exception("No results found! Please check it's a valid country name.")
        # postprocessed_results = ', '.join(results)
        return results

    def _get_spotify_token(self):
        """Obtain a new Spotify access token using the Client Credentials Flow."""
        url = "https://accounts.spotify.com/api/token"
        headers = {
            "Authorization": "Basic " + base64.b64encode(f"{os.environ.get('SPOTIFY_CLIENT_ID')}:{os.environ.get('SPOTIFY_CLIENT_SECRET')}".encode()).decode()
        }
        data = {"grant_type": "client_credentials"}

        response = requests.post(url, headers=headers, data=data)
        if response.status_code == 200:
            token_data = response.json()
            return token_data["access_token"], time.time() + token_data["expires_in"]
        else:
            raise Exception(f"Failed to get token: {response.json()}")

    def _ensure_valid_token(self):
        """Refresh the token if it's expired."""

        if time.time() >= self.token_expiration:
            self.access_token, self.token_expiration = self._get_spotify_token()

    def _make_request(self, url, params=None):
        """Make an API request, refreshing the token if needed."""
        self._ensure_valid_token()

        headers = {"Authorization": f"Bearer {self.access_token}"}

        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 401:  # Unauthorized (Token expired)
            self.access_token, self.token_expiration = self._get_spotify_token()
            headers = {"Authorization": f"Bearer {self.access_token}"}
            response = requests.get(url, headers=headers, params=params)

        if response.status_code != 200:
            print(f"API Error: {response.status_code}, {response.json()}")
            return None

        return response.json()

    def _search_playlists_by_country(self, country_name, limit=3):
        """Search for country-specific playlists."""
        url = 'https://api.spotify.com/v1/search'
        params = {'q': country_name, 'type': 'playlist', 'limit': limit}
        data = self._make_request(url, params)

        if not data or "playlists" not in data:
            return []

        return [item['id'] for item in data['playlists'].get('items', []) if item]

    def _get_playlist_tracks(self, playlist_id):
        """Get tracks from a playlist."""
        url = f'https://api.spotify.com/v1/playlists/{playlist_id}/tracks'
        data = self._make_request(url)

        if not data or "items" not in data:
            return []

        return [item['track']['artists'][0]['id'] for item in data['items'] if item and "track" in item]

    def _get_artist_genres(self, artist_id):
        """Retrieve genres associated with an artist."""
        url = f'https://api.spotify.com/v1/artists/{artist_id}'
        data = self._make_request(url)

        return data.get("genres", []) if data else []

    def get_top_genres_by_country(self, country_name: str) -> str:
        """A tool that gets the most common music genres in a country.
        Args:
            country_name: The name of the country to get the genres for (e.g., 'Argentine').
        """
        playlist_ids = self._search_playlists_by_country(country_name, limit=5)
        genre_counter = Counter()

        for playlist_id in playlist_ids:
            artist_ids = self._get_playlist_tracks(playlist_id)
            for artist_id in artist_ids[:25]:  # Limit to 25 per playlist
                genres = self._get_artist_genres(artist_id)
                if genres:
                    genre_counter.update(genres)

        return ', '.join([name for name, _ in genre_counter.most_common(5)])