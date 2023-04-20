import concurrent.futures
import json
import os
import random
import time
import urllib

import pandas as pd
import requests
from bs4 import BeautifulSoup


def get_json(url: str) -> dict:
    """
    Takes a URL from a rightmove search and returns the page's corrosponding json file.
    """
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    relevant_part = soup.find_all("script")
    json_text = relevant_part[5].text.replace("window.jsonModel = ", "")
    required_json = json.loads(json_text)
    return required_json


def json_to_df(json: dict) -> pd.DataFrame:
    """
    Takes a rightmove json and converts the relevant data to a pandas DataFrame.
    """
    df = pd.DataFrame(
        columns=[
            "Price_GBP",
            "Num_Bedrooms",
            "Num_Bathrooms",
            "Latitude",
            "Longitude",
            "Property_Type",
            "Address",
        ]
    )
    for i in range(len(json["properties"])):
        df.loc[i] = [
            json["properties"][i]["price"]["amount"],
            json["properties"][i]["bedrooms"],
            json["properties"][i]["bathrooms"],
            json["properties"][i]["location"]["latitude"],
            json["properties"][i]["location"]["longitude"],
            json["properties"][i]["propertySubType"],
            json["properties"][i]["displayAddress"],
        ]
    return df


def get_region_code(location: str) -> str:
    """
    Takes a location and finds the corrosponding rightmove code for the location.
    """
    response = requests.get(
        "https://www.rightmove.co.uk/house-prices/" + location + ".html"
    )
    soup = BeautifulSoup(response.content, "html.parser")
    relevant_part = soup.find_all("script")[3]
    json_text = relevant_part.text.replace("window.__PRELOADED_STATE__ = ", "")
    required_json = json.loads(json_text)
    return required_json["searchLocation"]["locationId"]


def create_url(
    location: str,
    min_price: str = "",
    max_price: str = "",
    min_bedrooms: str = "",
    max_bedrooms: str = "",
    min_bathrooms: str = "",
    max_bathrooms: str = "",
    radius: str = "",
    index: str = "",
) -> str:
    """
    Takes a variety of housing features and creates the rightmove URL needed to search for houses with the desired features.
    """
    base_url = "https://www.rightmove.co.uk/property-for-sale/find.html?"
    location_code = get_region_code(location)
    params = {
        "searchType": "SALE",
        "locationIdentifier": "REGION^" + location_code,
        "radius": radius,
        "minPrice": min_price,
        "maxPrice": max_price,
        "minBedrooms": min_bedrooms,
        "maxBedrooms": max_bedrooms,
        "minBathrooms": min_bathrooms,
        "maxBathrooms": max_bathrooms,
        "propertyTypes": "detached,semi-detached,terraced",
        "index": index,
    }
    final_url = base_url + urllib.parse.urlencode(params)
    return final_url


def create_url_list(
    location: str,
    min_price: str = "",
    max_price: str = "",
    min_bedrooms: str = "",
    max_bedrooms: str = "",
    min_bathrooms: str = "",
    max_bathrooms: str = "",
    radius: str = "",
) -> list[str]:
    """
    Takes a variety of housing features and creates a list of all rightmove URLs needed to search for houses with the desired features.
    """

    def get_index(i: int) -> str:
        # Creates the url for the page number corresponding to the given index.
        page_number = str(24 * (i - 1))
        return create_url(
            location,
            min_price,
            max_price,
            min_bedrooms,
            max_bedrooms,
            min_bathrooms,
            max_bathrooms,
            radius,
            index=page_number,
        )

    url_list = []
    threads = 30
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        executor = executor.map(get_index, range(43))
        for url in executor:
            url_list.append(url)
    return url_list


def download_jsons(url_list: list[str]) -> list[dict[str, str]]:
    """
    Takes a list of rightmove URLs and returns a list of their corrosponding json files.
    """
    threads = min(30, len(url_list))
    json_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        executor = executor.map(get_json, url_list)
        for json in executor:
            json_list.append(json)
        return json_list


def create_df(jsons: list[dict[str, str]]) -> pd.DataFrame:
    """
    Takes a list of rightmove jsons and creates a pandas DataFrame with the desired information.
    """
    df = pd.DataFrame(
        columns=[
            "Price_GBP",
            "Num_Bedrooms",
            "Num_Bathrooms",
            "Latitude",
            "Longitude",
            "Property_Type",
            "Address",
        ]
    )
    for json in jsons:
        new_df = json_to_df(json)
        df = pd.concat([df, new_df], ignore_index=True, axis=0)
    return df.drop_duplicates()


def create_table(
    location: str,
    min_price: str = "",
    max_price: str = "",
    min_bedrooms: str = "",
    max_bedrooms: str = "",
    min_bathrooms: str = "",
    max_bathrooms: str = "",
    radius: str = "",
    property_type: str = "",
) -> pd.DataFrame:
    """
    Creates a pandas DataFrame of all rightmove data corrosponding to the input information.
    """
    return create_df(
        download_jsons(
            create_url_list(
                location,
                min_price,
                max_price,
                min_bedrooms,
                max_bedrooms,
                min_bathrooms,
                max_bathrooms,
                radius,
                property_type,
            )
        )
    )


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans a pandas DataFrame of housing data. Assumes houses with no data on bathrooms has 1 bathroom.
    Converts numerical data to integer type. Removes details of properties that aren't detached, semi-detached or terraced.
    Renames columns to aid readability.
    """
    data["Num_Bathrooms"] = data["Num_Bathrooms"].fillna(1)
    data["Num_Bathrooms"] = data["Num_Bathrooms"].astype("int")
    data["Num_Bedrooms"] = data["Num_Bedrooms"].astype("int")
    data["Price_GBP"] = data["Price_GBP"].astype("int")
    data = pd.get_dummies(data, columns=["Property_Type"])
    required_columns = [
        "Price_GBP",
        "Num_Bedrooms",
        "Num_Bathrooms",
        "Latitude",
        "Longitude",
        "Address",
        "Property_Type_Detached",
        "Property_Type_Semi-Detached",
        "Property_Type_Terraced",
    ]
    data = data[required_columns]
    data = data[
        data["Property_Type_Detached"]
        + data["Property_Type_Semi-Detached"]
        + data["Property_Type_Terraced"]
        == 1
    ]
    data = data.rename(
        {
            "Property_Type_Detached": "Detached",
            "Property_Type_Semi-Detached": "Semi-Detached",
            "Property_Type_Terraced": "Terraced",
        },
        axis="columns",
    )
    return data


def etl(city_list: list[str]) -> pd.DataFrame:
    """
    Takes a list of cities, then performs an ETL process to create a pandas DataFrame contaning property data from each city.
    """
    for city in city_list:
        data = create_table(city)
        cleaned_data = clean_data(data)
        directory = os.getcwd()
        os.makedirs(directory + "\\data\\", exist_ok=True)
        #is_exist = os.path.exists(directory + "\\data\\")
        #if not is_exist:
        #    os.makedirs(directory + "\\data\\")
        cleaned_data.to_csv(directory + "\\data\\" + city.lower(), index=False)
        time.sleep(random.randint(5, 10))
        print(city + " is loaded")


city_list = [
    "Birmingham",
    "Bristol",
    "Cardiff",
    "Edinburgh",
    "Glasgow",
    "Leeds",
    "Liverpool",
    "London",
    "Manchester",
    "Newcastle-upon-Tyne",
    "York",
]

etl(city_list)
