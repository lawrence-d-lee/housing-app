import concurrent.futures
import json
import os
import random
import time
import urllib
from typing import Dict

import pandas as pd
import requests
from bs4 import BeautifulSoup
from etl import *

url = "https://www.rightmove.co.uk/property-for-sale/find.html?searchType=SALE&locationIdentifier=REGION%5E904&insId=1&radius=0.0&minPrice=&maxPrice=&minBedrooms=&maxBedrooms=&displayPropertyType=&maxDaysSinceAdded=&_includeSSTC=on&sortByPriceDescending=&primaryDisplayPropertyType=&secondaryDisplayPropertyType=&oldDisplayPropertyType=&oldPrimaryDisplayPropertyType=&newHome=&auction=false"
#print(type(json_to_df(get_json(url))))
print(type(get_region_code("manchester")))