import pandas as pd
import math as math
from sklearn.preprocessing import StandardScaler
import operator
import boto3
from io import StringIO

client = boto3.client('s3')
bucket_name = 'flower--bucket'
def get_csvfile(file_name):
    csv_obj = client.get_object(Bucket=bucket_name, Key=file_name)
    body = csv_obj['Body']
    csv_string = body.read().decode('utf-8')
    data = pd.read_csv(StringIO(csv_string))
    return data

def dataframe_col_rename(data):
    data = data.rename(columns={' Mean of the integrated profile': 'Profile_Mean',
                            ' Standard deviation of the integrated profile': 'Profile_SD',
                            ' Excess kurtosis of the integrated profile': 'Profile_Kurtosis',
                            ' Skewness of the integrated profile': 'Profile_Skewness',
                            ' Mean of the DM-SNR curve': 'Curve_Mean',
                            ' Standard deviation of the DM-SNR curve': 'Curve_SD',
                            ' Excess kurtosis of the DM-SNR curve': 'Curve_Kurtosis',
                            ' Skewness of the DM-SNR curve': 'Curve_Skewness'})
    return data

def data_preprocessing(data):
    df_features = data.drop(['target_class'], axis=1)
    df_result = data_normalization(df_features)
    return df_result

def data_normalization(data):
    scaler = StandardScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    return df_normalized

def knn_algorithm(row_to_pred, train_data, neighbour_count):
    distances = list()
    for index, row in train_data.iterrows():
        # Calculate Euclidean distance
        distance = 0
        for i in range(len(row_to_pred) - 1):
            distance += (row_to_pred[i] - row[i]) ** 2
        distance = math.sqrt(distance)
        distances.append({'index': index, 'distance': distance})

    distances.sort(key=operator.itemgetter('distance'))
    return distances[:neighbour_count]

df = get_csvfile('training_data/pulsar_stars.csv')
df = dataframe_col_rename(df)
df_train = data_preprocessing(df)

def predict(file_name):
    print(file_name)
    df_test_original = get_csvfile(file_name)
    df_test_original = dataframe_col_rename(df_test_original)
    df_test = data_preprocessing(df_test_original)
    df_final = df_test_original
    count = 0
    prediction_col = []
    for index, row in df_test.iterrows():
        print('Target row index: {}'.format(index))
        result_list = knn_algorithm(row, df_train, 5)
        output_values = [df.iloc[res['index']].target_class for res in result_list]
        print(output_values)
        prediction = max(output_values, key=output_values.count)
        prediction_col.append(prediction)
        print('prediction: {},  Real value: {}'.format(prediction, df.iloc[index].target_class))
        count += 1
    df_final.insert(8, "Prediction", prediction_col, True)
    print(df_final.head(50))
    #df_final.to_csv(r'Prediction_Results.csv')

    csv_buffer = StringIO()
    df_final.to_csv(csv_buffer, index=False)
    #s3_resource = boto3.resource("s3")
    #s3_resource.Object(bucket_name, 'results/knn_results.csv').put(Body=csv_buffer.getvalue())

    #df_final.to_csv(csv_buffer, index=False)
    client.put_object(Key='results/knn_output.csv', Bucket=bucket_name, Body=csv_buffer.getvalue())

    return "success"




