import time

import pandas as pd
import bodo

import numpy as np
import xgboost as xgb


@bodo.jit
def clean(ddf):
    cols_to_drop_2014 = ['vendor_id', 'store_and_fwd_flag', 'payment_type', 'surcharge', 'mta_tax', 'tip_amount', 'tolls_amount', 'total_amount']
    # cols_to_drop_2015_2016 = ['vendorid', 'store_and_fwd_flag', 'payment_type', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge', 'total_amount']

    # columns_to_rename = cols_to_rename_2014
    # columns_to_drop = cols_to_drop_2014

    # if year == ("2015" or "2016"):
    #     columns_to_rename = cols_to_rename_2015_2016
    #     columns_to_drop = cols_to_drop_2015_2016

    # ddf = ddf.rename(columns=cols_to_rename_2014)

    ddf = ddf.rename(
        columns={
            "tpep_pickup_datetime": "pickup_datetime",
            "tpep_dropoff_datetime": "dropoff_datetime",
            "ratecodeid": "rate_code",
        }
    )

    ddf = ddf.drop(columns=cols_to_drop_2014)

    return ddf


@bodo.jit
def read_data(base_path):
    t = time.time()

    columns = ["vendor_id",
        "pickup_datetime",
        "dropoff_datetime", "passenger_count",
        "trip_distance", "pickup_longitude", "pickup_latitude", "rate_code",
        "store_and_fwd_flag", "dropoff_longitude", "dropoff_latitude", "payment_type",
        "fare_amount", "surcharge", "mta_tax", "tip_amount",
        "tolls_amount", "total_amount"]
  

    df_2014 = pd.read_csv(
        base_path + "2014/yellow_tripdata_2014-01.csv",
        parse_dates=["pickup_datetime", "dropoff_datetime"],
        names=columns,
        header=0,
    )

    df_2014 = clean(df_2014)
    # df_2015 = clean(df_2015)
    # df_2016 = clean(df_2016)
    # taxi_df = pd.concat([df_2014, df_2015, df_2016], ignore_index=True)
    print(f"read time internal, s: {time.time() - t}")
    return df_2014


@bodo.jit
def etl(taxi_df):
    t = time.time()
    taxi_df = taxi_df[
        (taxi_df.fare_amount > 1)
        & (taxi_df.fare_amount < 500)
        & (taxi_df.passenger_count > 0)
        & (taxi_df.passenger_count < 6)
        & (taxi_df.pickup_longitude > -75)
        & (taxi_df.pickup_longitude < -73)
        & (taxi_df.dropoff_longitude > -75)
        & (taxi_df.dropoff_longitude < -73)
        & (taxi_df.pickup_latitude > 40)
        & (taxi_df.pickup_latitude < 42)
        & (taxi_df.dropoff_latitude > 40)
        & (taxi_df.dropoff_latitude < 42)
        & (taxi_df.trip_distance > 0)
        & (taxi_df.trip_distance < 500)
        & ((taxi_df.trip_distance <= 50) | (taxi_df.fare_amount >= 50))
        & ((taxi_df.trip_distance >= 10) | (taxi_df.fare_amount <= 300))
        & (taxi_df.dropoff_datetime > taxi_df.pickup_datetime)
    ]

    taxi_df = taxi_df.reset_index(drop=True)

    taxi_df["day"] = taxi_df["pickup_datetime"].dt.day
    taxi_df["diff"] = taxi_df["dropoff_datetime"].astype("int64") - taxi_df[
        "pickup_datetime"
    ].astype("int64")

    taxi_df["pickup_latitude_r"] = taxi_df["pickup_latitude"] // 0.01 * 0.01
    taxi_df["pickup_longitude_r"] = taxi_df["pickup_longitude"] // 0.01 * 0.01
    taxi_df["dropoff_latitude_r"] = taxi_df["dropoff_latitude"] // 0.01 * 0.01
    taxi_df["dropoff_longitude_r"] = taxi_df["dropoff_longitude"] // 0.01 * 0.01

    taxi_df = taxi_df.drop("pickup_datetime", axis=1)
    taxi_df = taxi_df.drop("dropoff_datetime", axis=1)

    dlon = taxi_df["dropoff_longitude"] - taxi_df["pickup_longitude"]
    dlat = taxi_df["dropoff_latitude"] - taxi_df["pickup_latitude"]
    taxi_df["e_distance"] = dlon * dlon + dlat * dlat

    taxi_df = taxi_df.drop("trip_distance", axis=1)

    # Bodo can't use this columns
    taxi_df = taxi_df.drop(columns=["passenger_count", "rate_code"])

    X_train = taxi_df[taxi_df.day < 25]
    # create a Y_train ddf with just the target variable
    Y_train = X_train[["fare_amount"]]
    # drop the target variable from the training ddf
    # X_train = X_train[X_train.columns.difference(["fare_amount"])]
    X_train = X_train.drop(columns="fare_amount")

    X_test = taxi_df[taxi_df.day >= 25]
    # Create Y_test with just the fare amount
    Y_test = X_test[['fare_amount']]

    # Drop the fare amount from X_test
    X_test = X_test.drop(columns="fare_amount")

    X_test = X_test.drop(columns=["day"])
    X_train = X_train.drop(columns=["day"])


    print(f"etl time internal, s: {time.time() - t}")
    return X_train, Y_train, X_test, Y_test


@bodo.jit  # (cache=True)
def data_processing():
    base_path = "/localdisk/benchmark_datasets/yellow-taxi-dataset/"

    print("\nread_csv ...")
    t = time.time()
    taxi_df = read_data(base_path)
    print("Data read shape: ", taxi_df.shape)
    print(f"read time, s: {time.time() - t}")

    print("\netl ...")
    t = time.time()
    X_train, Y_train, X_test, Y_test = etl(taxi_df)

    print("X train shape: ", X_train.shape)
    print("X test shape: ", X_test.shape)
    print(f"etl time, s: {time.time() - t}")

    return bodo.gatherv(X_train), bodo.gatherv(Y_train), bodo.gatherv(X_test), bodo.gatherv(Y_test)


if __name__ == "__main__":
    print("main ...")

    t = time.time()
    X_train, Y_train, X_test, Y_test = data_processing()

    if bodo.get_rank() == 0:
        print("\nml ...")
        tt = time.time()
        dtrain = xgb.DMatrix(X_train, Y_train)

        trained_model = xgb.train({
            'learning_rate': 0.3,
            'max_depth': 8,
            'objective': 'reg:squarederror',
            'subsample': 0.6,
            'gamma': 1,
            'tree_method':'hist'
            },
            dtrain,
            num_boost_round=100,
            verbose_eval=False,
            evals=[(dtrain, 'train')])

        booster = trained_model
        prediction = pd.Series(booster.predict(xgb.DMatrix(X_test)))
        print(prediction.shape)

        # prediction = prediction.map_partitions(lambda part: cudf.Series(part)).reset_index(drop=True)
        actual = Y_test['fare_amount'].reset_index(drop=True)

        print(f'prediction:\n{prediction.head()}')
        print(f'actual:\n{actual.head()}')

        print(f"ml time, s: {time.time() - tt}")

        # Calculate RMSE
        squared_error = ((prediction-actual)**2)

        # compute the actual RMSE over the full test set
        print(f'RMSE: {np.sqrt(squared_error.mean())}')
        print(f"main time, s: {time.time() - t}")

