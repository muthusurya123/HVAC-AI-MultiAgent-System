from agents.weather import WeatherAgent

weather = WeatherAgent()

temps, humidity = weather.get_weather_forecast()

print("\nWeather Forecast Test")
print("----------------------")

print("Temperatures:", temps)
print("Humidity:", humidity)