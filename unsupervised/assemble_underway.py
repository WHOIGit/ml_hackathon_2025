import pandas as pd

en617 = pd.read_csv('underway/en617_underway.csv')
en627 = pd.read_csv('underway/en627_underway.csv')

en617['cruise'] = 'EN617'
en627['cruise'] = 'EN627'


selected_columns = [
    "cruise",

    # Decimal Day of Year
    #"datetime_decimaldoy",

    # GPS: Furuno Latitude & Longitude
    "gps_furuno_latitude",
    "gps_furuno_longitude",

    # Meteorology: RMY Bow
    "met_rmy_bow_airtemp",
    "met_rmy_bow_baropressure",
    "met_rmy_bow_relhumidity",

    # Radiation Sensors
    "rad1_lw",
    "rad1_sw",
    "rad2_lw",
    "rad2_sw",

    # TSG1 (most important, non-redundant)
    "tsg2_conductivity",
    "tsg2_salinity",
    "tsg2_temperature",
    "tsg2_soundvelocity"
]

# create a dataset that is mostly en617, but with just a small number of rows from en627
en617_subset = en617.sample(n=2000, random_state=42)
en627_subset = en627.sample(n=10, random_state=42)

en617_with_anomalies = pd.concat([en617_subset, en627_subset], ignore_index=True)[selected_columns]
# shuffle the rows
en617_with_anomalies = en617_with_anomalies.sample(frac=1, random_state=42).reset_index(drop=True)
en617_with_anomalies.to_csv('underway/underway_with_anomalies.csv', index=False)
