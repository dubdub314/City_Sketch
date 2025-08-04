import requests

class GoogleMapsStub:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def geocode(self, name: str):
        # TODO: implement actual Google Geocoding request and parse lat/lng
        return None
