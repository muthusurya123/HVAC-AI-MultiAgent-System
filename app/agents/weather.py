import requests
import os
from dotenv import load_dotenv


class WeatherAgent:

    def __init__(self):

        # Load environment variables
        load_dotenv()

        self.api_key = os.getenv("WEATHER_API_KEY")
        self.city = os.getenv("CITY")

        self.url = f"https://api.openweathermap.org/data/2.5/forecast?q={self.city}&appid={self.api_key}&units=metric"


    def get_weather_forecast(self):

        response = requests.get(self.url)

        if response.status_code != 200:
            raise Exception("Weather API request failed")

        data = response.json()

        # Safety check
        if "list" not in data:
            raise Exception("Invalid weather API response")

        temps = []
        humidity = []

        # Get next 24 hours (3-hour interval × 8)
        for item in data["list"][:8]:

            temps.append(item["main"]["temp"])
            humidity.append(item["main"]["humidity"])

        return temps, humidity