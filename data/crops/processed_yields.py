import pandas as pd
import os
from glob import glob
from tqdm import tqdm, trange

def subsample_df(df, num_rows=500):
    try:
        frac =  min(1.0, num_rows / df.shape[0])
        df = df.sample(frac=frac)
    except:
        print(f"File has no values")
    return df

dates = ['04-07', '04-23', '05-09', '05-25',
       '06-10', '06-26', '07-12', '07-28',
       '08-13', '08-29', '09-14', '09-30',
       '10-16']

# root_dir = "latlon"
# columns = ['longitude', 'latitude', 'County', 'State']

# files = glob(f"{root_dir}/*.csv")

# df_final = pd.DataFrame(columns=
#     ["longitude", "latitude", "County", "State"] 
# )
# for file in tqdm(files):
#     i = 0
#     df = pd.read_csv(file)
#     df = df[columns]
#     df_processed = df.copy()
#     df_final = df_final.append(df_processed, sort=True)

# df_final.to_csv("processed_covariates/latlon-500_data_all.csv", index=False)


# """## subsampled latlon"""
# root_dir = "latlon"
# columns = ['longitude', 'latitude', 'County', 'State']

# files = glob(f"{root_dir}/*.csv")

# df_final = pd.DataFrame(columns=
#     ["longitude", "latitude", "County", "State"] 
# )
# for file in tqdm(files):
#     i = 0
#     df = pd.read_csv(file)
#     df = df[columns]
#     df_processed = df.copy()
#     df_processed = subsample_df(df_processed)
#     df_final = df_final.append(df_processed, sort=True)

# # df_final.to_csv("processed_covariates/latlon-500_data.csv", index=False)
# # df_final.to_csv("processed_covariates/latlon-500_data_200points.csv", index=False)
# df_final.to_csv("processed_covariates/latlon-500_data_500points.csv", index=False)

# """## No subsampling"""
# root_dir = "MODIS"
# columns = ['month_date', 'longitude', 'latitude', 'NDVI', 'EVI',
#        'County', 'State']

# files = glob(f"{root_dir}/*.csv")

# df = pd.read_csv(files[0])
# df["month_date"] = df["datetime"].apply(lambda v: v[5:])


# df_final = pd.DataFrame(columns=
#     ["longitude", "latitude", "County", "State"] +
#     [f"NDVI_{date}" for date in dates]+
#     [f"EVI_{date}" for date in dates]
# )
# for file in tqdm(files):
#     i = 0
#     df = pd.read_csv(file)
#     year = df["datetime"].values[0][:4]
#     df["month_date"] = df["datetime"].apply(lambda v: v[5:])
#     df = df[columns]
#     for date in dates:
#         if i == 0:
#             df_processed = df[df["month_date"]==date].copy()
#             df_processed.rename(columns={"NDVI": f"NDVI_{date}", "EVI": f"EVI_{date}"}, inplace=True)
#         else:
#             df_tmp = df[df["month_date"]==date][["longitude", "latitude", "County", "State", "NDVI", "EVI"]]
#             df_tmp.rename(columns={"NDVI": f"NDVI_{date}", "EVI": f"EVI_{date}"}, inplace=True)
#             df_processed = df_processed.merge(df_tmp, on=["longitude", "latitude", "County", "State"])
#         i+=1
#     df_processed["year"] = year
#     df_processed = df_processed.drop(["month_date"], axis=1)
#     df_final = df_final.append(df_processed, sort=True)

# df_final.to_csv("processed_covariates/MOD13Q1-1000-all_data.csv", index=False)

"""## GRIDMET"""

# root_dir = "GRIDMET"
# files = glob(f"{root_dir}/*.csv")
# columns = ['month_date', 'longitude', 'latitude', 'County', 'State']
# bands = ["tmmx", "pr"]
# columns += bands
# df = pd.read_csv(files[0])
# df["month_date"] = df["datetime"].apply(lambda v: v[5:])
# df = df[df["month_date"].isin(dates)]
# # there are too many days, so we just use the same days as MODIS
# df_final = pd.DataFrame(columns=
#     ["month_date", "longitude", "latitude", "County", "State"] +
#     [f"pr_{date}" for date in dates]+
#     [f"tmmx_{date}" for date in dates]
# )
# for file in tqdm(files):
#     i = 0
#     df = pd.read_csv(file)
#     year = df["datetime"].values[0][:4]
#     df["month_date"] = df["datetime"].apply(lambda v: v[5:])
#     df = df[df["month_date"].isin(dates)]
#     df = df[columns]
#     for date in dates:
#         if i == 0:
#             df_processed = df[df["month_date"]==date].copy()
#             df_processed.rename(columns={"pr": f"pr_{date}", 
#                                          "tmmx": f"tmmx_{date}"}, inplace=True)

#         else:
#             df_tmp = df[df["month_date"]==date][["longitude", "latitude", "County", "State", "pr", "tmmx"]]
#             df_tmp.rename(columns={"pr": f"pr_{date}", 
#                                    "tmmx": f"tmmx_{date}"}, inplace=True)
#             df_processed = df_processed.merge(df_tmp)
#         i+=1
#     df_processed["year"] = year
#     df_processed = df_processed.drop(["month_date"], axis=1)
#     df_final = df_final.append(df_processed, sort=True)

# df_final.to_csv("processed_covariates/GRIDMET-all_data.csv", index=False)

# # df_final

# """## MODIS"""
# root_dir = "MODIS"
# columns = ['month_date', 'longitude', 'latitude', 'NDVI', 'EVI',
#        'County', 'State']
# files = glob(f"{root_dir}/*.csv")
# dates = ['04-07', '04-23', '05-09', '05-25',
#        '06-10', '06-26', '07-12', '07-28',
#        '08-13', '08-29', '09-14', '09-30',
#        '10-16']


# df = pd.read_csv(files[0])
# df["month_date"] = df["datetime"].apply(lambda v: v[5:])

# df_final = pd.DataFrame(columns=
#     ["month_date", "longitude", "latitude", "County", "State"] +
#     [f"NDVI_{date}" for date in dates]+
#     [f"EVI_{date}" for date in dates]
# )
# for file in tqdm(files):
#     i = 0
#     df = pd.read_csv(file)
#     year = df["datetime"].values[0][:4]
#     df["month_date"] = df["datetime"].apply(lambda v: v[5:])
#     df = df[columns]
#     for date in dates:
#         if i == 0:
#             df_processed = df[df["month_date"]==date].copy()
#             df_processed.rename(columns={"NDVI": f"NDVI_{date}", "EVI": f"EVI_{date}"}, inplace=True)
#         else:
#             df_tmp = df[df["month_date"]==date][["longitude", "latitude", "County", "State", "NDVI", "EVI"]]
#             df_tmp.rename(columns={"NDVI": f"NDVI_{date}", "EVI": f"EVI_{date}"}, inplace=True)
#             df_processed = df_processed.merge(df_tmp)
#         i+=1
#     df_processed = subsample_df(df_processed)
#     df_processed["year"] = year
#     df_processed = df_processed.drop(["month_date"], axis=1)
#     df_final = df_final.append(df_processed, sort=True)

# df_final.to_csv("processed_covariates/MOD13Q1-1000_data.csv", index=False)

# """## GRIDMET"""
# subsampling doesn't do much since GRIDMET is very large already

# root_dir = "GRIDMET"
# files = glob(f"{root_dir}/*.csv")
# columns = ['month_date', 'longitude', 'latitude', 'County', 'State']
# bands = ["tmmx", "pr"]
# columns += bands
# df = pd.read_csv(files[0])
# df["month_date"] = df["datetime"].apply(lambda v: v[5:])
# df = df[df["month_date"].isin(dates)]

# # there are too many days, so we just use the same days as MODIS
# df_final = pd.DataFrame(columns=
#     ["month_date", "longitude", "latitude", "County", "State"] +
#     [f"pr_{date}" for date in dates]+
#     [f"tmmx_{date}" for date in dates]
# )
# for file in tqdm(files):
#     i = 0
#     df = pd.read_csv(file)
#     year = df["datetime"].values[0][:4]
#     df["month_date"] = df["datetime"].apply(lambda v: v[5:])
#     df = df[df["month_date"].isin(dates)]
#     df = df[columns]
#     for date in dates:
#         if i == 0:
#             df_processed = df[df["month_date"]==date].copy()
#             df_processed.rename(columns={"pr": f"pr_{date}", 
#                                          "tmmx": f"tmmx_{date}"}, inplace=True)

#         else:
#             df_tmp = df[df["month_date"]==date][["longitude", "latitude", "County", "State", "pr", "tmmx"]]
#             df_tmp.rename(columns={"pr": f"pr_{date}", 
#                                    "tmmx": f"tmmx_{date}"}, inplace=True)
#             df_processed = df_processed.merge(df_tmp)
#         i+=1
#     df_processed = subsample_df(df_processed)
#     df_processed["year"] = year
#     df_processed = df_processed.drop(["month_date"], axis=1)
#     df_final = df_final.append(df_processed, sort=True)

# df_final.to_csv("processed_covariates/GRIDMET_data.csv", index=False)

# df_final

# """## Same resolution MODIS-GRIDMET data"""

root_dir = "MODIS-downsampled"
files = glob(f"{root_dir}/*.csv")
columns = ['month_date', 'longitude', 'latitude', 'County', 'State']
bands = ['NDVI', 'EVI']
columns += bands

# there are too many days, so we just use the same days as MODIS
df_final = pd.DataFrame(columns=
    ["month_date", "longitude", "latitude", "County", "State"] +
    [f"NDVI_{date}" for date in dates]+
    [f"EVI_{date}" for date in dates]
)
for file in tqdm(files):
    i = 0
    df = pd.read_csv(file)
    year = df["datetime"].values[0][:4]
    df["month_date"] = df["datetime"].apply(lambda v: v[5:])
    df = df[df["month_date"].isin(dates)]
    df.rename(columns={"County_x": "County", "State_x": "State"}, inplace=True)
    df = df[columns]
    for date in dates:
        if i == 0:
            df_processed = df[df["month_date"]==date].copy()
            df_processed.rename(columns={"pr": f"pr_{date}", 
                                         "tmmx": f"tmmx_{date}",
                                         "NDVI": f"NDVI_{date}",
                                         "EVI": f"EVI_{date}",
                                         }, inplace=True)

        else:
            df_tmp = df[df["month_date"]==date][["longitude", "latitude", "County", "State", "NDVI", "EVI"]]
            df_tmp.rename(columns={"pr": f"pr_{date}", 
                                   "tmmx": f"tmmx_{date}",
                                    "NDVI": f"NDVI_{date}",
                                    "EVI": f"EVI_{date}",
                                   }, inplace=True)
            df_processed = df_processed.merge(df_tmp, on=["longitude", "latitude", "County", "State"])
        i+=1
    df_processed = subsample_df(df_processed)
    df_processed["year"] = year
    df_processed = df_processed.drop(["month_date"], axis=1)
    df_final = df_final.append(df_processed, sort=True)

df_final.to_csv("processed_covariates/MOD13Q1-downsampled_data.csv", index=False)

# df_final

# """## Same resolution MODIS-GRIDMET data upsampled"""

# root_dir = "gdrive/My Drive/MODIS-GRIDMET-sameres"
# files = glob(f"{root_dir}/*reprojected*.csv")
# columns = ['datetime', 'longitude', 'latitude', 'County', 'State']
# bands = ['NDVI', 'EVI', "tmmx", "pr"]
# columns += bands
# df = pd.read_csv(files[0])
# dates = df["datetime"].unique()
# # there are too many days, so we just use the same days as MODIS
# df_final = pd.DataFrame(columns=
#     ["datetime", "longitude", "latitude", "County", "State"] +
#     [f"NDVI_{date}" for date in dates]+
#     [f"EVI_{date}" for date in dates]+
#     [f"pr_{date}" for date in dates]+
#     [f"tmmx_{date}" for date in dates]
# )
# for file in tqdm(files):
#     i = 0
#     df = pd.read_csv(file)
#     df = df[df["datetime"].isin(dates)]
#     df.rename(columns={"County_x": "County", "State_x": "State"}, inplace=True)
#     df = df[columns]
#     for date in dates:
#         if i == 0:
#             df_processed = df[df["datetime"]==date].copy()
#             df_processed.rename(columns={"pr": f"pr_{date}", 
#                                          "tmmx": f"tmmx_{date}",
#                                          "NDVI": f"NDVI_{date}",
#                                          "EVI": f"EVI_{date}",
#                                          }, inplace=True)
#         else:
#             df_tmp = df[df["datetime"]==date][["longitude", "latitude", "County", "State", "NDVI", "EVI", "pr", "tmmx"]]
#             df_tmp.rename(columns={"pr": f"pr_{date}", 
#                                    "tmmx": f"tmmx_{date}",
#                                     "NDVI": f"NDVI_{date}",
#                                     "EVI": f"EVI_{date}",
#                                    }, inplace=True)
#             df_processed = df_processed.merge(df_tmp, on=["longitude", "latitude", "County", "State"])
#         i+=1
#     df_processed = subsample_df(df_processed)
#     df_final = df_final.append(df_processed, sort=True)

# df_final.to_csv("processed_covariates")

# # # df_final

