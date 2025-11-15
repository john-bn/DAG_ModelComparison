import pandas as pd

def major_airports_df():
    """Major CONUS city hubs (ICAO, city, lat, lon)."""
    return pd.DataFrame([
        ("KATL","Atlanta",      33.6367,  -84.4281),
        ("KLAX","Los Angeles",  33.9416, -118.4085),
        ("KORD","Chicago",      41.9742,  -87.9073),
        ("KDFW","Dallas/FortW", 32.8968,  -97.0379),
        ("KDEN","Denver",       39.8617, -104.6731),
        ("KJFK","New York",     40.6413,  -73.7781),
        ("KSFO","San Francisco",37.6213, -122.3790),
        ("KSEA","Seattle",      47.4502, -122.3088),
        ("KLAS","Las Vegas",    36.0840, -115.1537),
        ("KCLT","Charlotte",    35.2140,  -80.9431),
        ("KPHX","Phoenix",      33.4343, -112.0116),
        ("KIAH","Houston",      29.9902,  -95.3368),
        ("KBOS","Boston",       42.3656,  -71.0096),
        ("KMSP","Minneapolis",  44.8848,  -93.2223),
        ("KDTW","Detroit",      42.2124,  -83.3534),
        ("KPHL","Philadelphia", 39.8744,  -75.2424),
        ("KBWI","Baltimore",    39.1754,  -76.6684),
        ("KSLC","Salt Lake",    40.7899, -111.9791),
        ("KSAN","San Diego",    32.7338, -117.1933),
        ("KTPA","Tampa",        27.9755,  -82.5332),
    ], columns=["icao","city","lat","lon"])
