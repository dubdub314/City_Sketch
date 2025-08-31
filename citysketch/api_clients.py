class GoogleMapsStub:
    """Placeholder: fill with your real Google Maps API calls if you do not have RealMap shapefiles.
    Use it only when RealMap is unavailable. """
    def __init__(self, api_key: str):
        self.api_key = api_key
    def geocode(self, name: str):
        # Return fake coordinates for now
        return {'name': name, 'lat': 0.0, 'lng': 0.0}
