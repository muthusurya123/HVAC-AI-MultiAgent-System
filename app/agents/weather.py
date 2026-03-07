import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


class WeatherAgent:

    def __init__(self):
        self.api_key = os.getenv("WEATHER_API_KEY")
        self.city = os.getenv("CITY")

        print("Loaded API KEY:", self.api_key)
        print("Loaded CITY:", self.city)

    def get_weather_forecast(self):

        url = f"https://api.openweathermap.org/data/2.5/forecast?q={self.city}&appid={self.api_key}&units=metric"

        response = requests.get(url)

        data = response.json()

        print("API RESPONSE:", data)

        # If API failed
        if "list" not in data:
            raise Exception("Weather API failed. Check API key or city name.")

        temps = []
        humidity = []

        # OpenWeather returns data every 3 hours
        # 8 entries = 24 hours
        for item in data["list"][:8]:
            temps.append(item["main"]["temp"])
            humidity.append(item["main"]["humidity"])

        return temps, humidity