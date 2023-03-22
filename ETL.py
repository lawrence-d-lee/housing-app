import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import urllib
import time
import concurrent.futures
import random

def get_json(URL):
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, "html.parser")
    relevant_part = soup.find_all("script")
    json_text = relevant_part[5].text.replace("window.jsonModel = ", "")
    required_json = json.loads(json_text)
    return required_json

def json_to_df(json):
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

def get_region_code(location):
    response = requests.get(
        "https://www.rightmove.co.uk/house-prices/" + location + ".html"
    )
    soup = BeautifulSoup(response.content, "html.parser")
    relevant_part = soup.find_all("script")[3]
    json_text = relevant_part.text.replace("window.__PRELOADED_STATE__ = ", "")
    required_json = json.loads(json_text)
    return required_json["searchLocation"]["locationId"]

def create_url(location, min_price ='', max_price = '', min_bedrooms='', max_bedrooms='', min_bathrooms='', max_bathrooms='', radius='', property_type='', index=''):
    base_url = "https://www.rightmove.co.uk/property-for-sale/find.html?"
    location_code = get_region_code(location)
    params = {'searchType': 'SALE', 'locationIdentifier': 'REGION^' + location_code, 
              "radius": radius, "minPrice":min_price,"maxPrice":max_price, 
              "minBedrooms":min_bedrooms, "maxBedrooms":max_bedrooms, 
              "minBathrooms":min_bathrooms, "maxBathrooms":'', "propertyTypes":"detached,semi-detached,terraced",
             "index":index}
    final_url = base_url + urllib.parse.urlencode(params)
    return final_url

def create_url_list(
    location,
    min_price="",
    max_price="",
    min_bedrooms="",
    max_bedrooms="",
    min_bathrooms="",
    max_bathrooms="",
    radius="",
    property_type="",
):
    def get_index(i):
        return create_url(
            location,
            min_price,
            max_price,
            min_bedrooms,
            max_bedrooms,
            min_bathrooms,
            max_bathrooms,
            radius,
            property_type,
            index=24 * (i - 1))
    url_list=[]
    threads = 30
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        executor = executor.map(get_index, range(43))
        for url in executor:
            url_list.append(url)
    return url_list

def download_jsons(url_list):
    threads = min(30, len(url_list))
    json_list=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        executor = executor.map(get_json, url_list)
        for json in executor:
            json_list.append(json)
        return json_list  
    
def create_df(jsons):
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
    location,
    min_price="",
    max_price="",
    min_bedrooms="",
    max_bedrooms="",
    min_bathrooms="",
    max_bathrooms="",
    radius="",
    property_type="",
):
    return create_df(download_jsons(create_url_list(location,
            min_price,
            max_price,
            min_bedrooms,
            max_bedrooms,
            min_bathrooms,
            max_bathrooms,
            radius,
            property_type)))

def clean_data(data):
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
    data = data.rename({'Property_Type_Detached': 'Detached', 'Property_Type_Semi-Detached': 'Semi-Detached', "Property_Type_Terraced": "Terraced"}, axis='columns')
    return data

def ETL(city_list):
    for city in city_list:
        data = create_table(city)
        cleaned_data = clean_data(data)
        Path = "C:\\Users\\Lawrence\\Desktop\\Housing-Project\\"
        cleaned_data.to_csv(Path + city + "_csv", index=False)
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

#ETL(city_list)