"""Real-world anchored road-traffic emission factors.

The values below are engineering estimates in grams per vehicle-kilometre for
urban traffic. They are intended for student-project analytics when a full
regulatory model such as EPA MOVES or COPERT is not available locally.

Source basis:
* EPA typical passenger vehicle CO2 page: about 400 g CO2 per mile, which is
  about 249 g/km for an average gasoline passenger vehicle.
* EPA MOVES: official U.S. on-road model covering CO2, CO, NOx, PM, VOC and
  other pollutants by vehicle class, speed, fuel and age.
* EEA/EMEP road-transport guidebook: European emission-inventory methodology
  for road transport pollutants.
* WHO ambient air-quality guidance: health framing for PM, NO2, SO2, CO and
  ozone-forming pollutants.

Because exact emissions depend on fuel, engine standard, vehicle age, gradient,
temperature and driving cycle, the app labels outputs as estimates and keeps the
factor table visible in the dashboard and documentation.
"""
from __future__ import annotations

POLLUTANTS = ["co2", "co", "nox", "pm25", "pm10", "hc", "voc", "so2", "ch4", "n2o"]

# g/km by detected class. Bicycle is zero tailpipe emissions.
EMISSION_FACTORS_G_PER_KM = {
    "car": {
        "co2": 249.0,
        "co": 1.60,
        "nox": 0.18,
        "pm25": 0.010,
        "pm10": 0.020,
        "hc": 0.12,
        "voc": 0.10,
        "so2": 0.006,
        "ch4": 0.018,
        "n2o": 0.012,
    },
    "motorcycle": {
        "co2": 95.0,
        "co": 3.20,
        "nox": 0.08,
        "pm25": 0.012,
        "pm10": 0.018,
        "hc": 0.55,
        "voc": 0.45,
        "so2": 0.003,
        "ch4": 0.030,
        "n2o": 0.004,
    },
    "bus": {
        "co2": 1050.0,
        "co": 2.20,
        "nox": 5.80,
        "pm25": 0.120,
        "pm10": 0.180,
        "hc": 0.45,
        "voc": 0.38,
        "so2": 0.035,
        "ch4": 0.060,
        "n2o": 0.035,
    },
    "truck": {
        "co2": 780.0,
        "co": 1.80,
        "nox": 4.60,
        "pm25": 0.100,
        "pm10": 0.160,
        "hc": 0.38,
        "voc": 0.32,
        "so2": 0.030,
        "ch4": 0.045,
        "n2o": 0.030,
    },
    "bicycle": {
        "co2": 0.0,
        "co": 0.0,
        "nox": 0.0,
        "pm25": 0.0,
        "pm10": 0.0,
        "hc": 0.0,
        "voc": 0.0,
        "so2": 0.0,
        "ch4": 0.0,
        "n2o": 0.0,
    },
}

# 100-year global warming potentials from IPCC values used by EPA pages.
GWP_100 = {
    "co2": 1.0,
    "ch4": 28.0,
    "n2o": 265.0,
}

POLLUTANT_HEALTH = {
    "co2": {
        "name": "Carbon dioxide",
        "unit": "g/km",
        "severity": "Climate",
        "harm": "Main greenhouse gas from fuel combustion; not an urban toxic exposure indicator at roadside levels.",
    },
    "co": {
        "name": "Carbon monoxide",
        "unit": "g/km",
        "severity": "High",
        "harm": "Reduces blood oxygen transport; dangerous in enclosed or poorly ventilated environments.",
    },
    "nox": {
        "name": "Nitrogen oxides",
        "unit": "g/km",
        "severity": "High",
        "harm": "Contributes to NO2 exposure, ozone formation and respiratory irritation.",
    },
    "pm25": {
        "name": "Fine particulate matter",
        "unit": "g/km",
        "severity": "Very high",
        "harm": "Penetrates deep into lungs and is strongly linked with cardiopulmonary risk.",
    },
    "pm10": {
        "name": "Coarse particulate matter",
        "unit": "g/km",
        "severity": "High",
        "harm": "Irritates airways and worsens respiratory disease; includes brake, tyre and road dust.",
    },
    "hc": {
        "name": "Hydrocarbons",
        "unit": "g/km",
        "severity": "Medium",
        "harm": "Unburned fuel compounds; several species contribute to smog and toxic exposure.",
    },
    "voc": {
        "name": "Volatile organic compounds",
        "unit": "g/km",
        "severity": "Medium",
        "harm": "Ozone and secondary aerosol precursor; some compounds are toxic.",
    },
    "so2": {
        "name": "Sulfur dioxide",
        "unit": "g/km",
        "severity": "Medium",
        "harm": "Respiratory irritant and sulfate particle precursor; usually low for low-sulfur fuels.",
    },
    "ch4": {
        "name": "Methane",
        "unit": "g/km",
        "severity": "Climate",
        "harm": "Potent greenhouse gas; converted to CO2-equivalent using 100-year GWP.",
    },
    "n2o": {
        "name": "Nitrous oxide",
        "unit": "g/km",
        "severity": "Climate",
        "harm": "Very high global warming potential; included in CO2-equivalent emissions.",
    },
    "co2e": {
        "name": "CO2 equivalent",
        "unit": "g/km",
        "severity": "Climate",
        "harm": "Combined climate impact of CO2, CH4 and N2O using 100-year GWP.",
    },
}

SOURCE_NOTES = [
    {
        "label": "EPA typical passenger vehicle CO2",
        "url": "https://www.epa.gov/greenvehicles/greenhouse-gas-emissions-typical-passenger-vehicle",
    },
    {
        "label": "EPA MOVES on-road emission model",
        "url": "https://www.epa.gov/moves",
    },
    {
        "label": "EPA/DOE FuelEconomy.gov downloads",
        "url": "https://www.fueleconomy.gov/feg/download.shtml",
    },
    {
        "label": "WHO ambient outdoor air quality and health",
        "url": "https://www.who.int/news-room/fact-sheets/detail/ambient-(outdoor)-air-quality-and-health",
    },
    {
        "label": "EPA global warming potentials",
        "url": "https://www.epa.gov/ghgemissions/understanding-global-warming-potentials",
    },
]
