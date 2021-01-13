from sklearn.ensemble import RandomForestClassifier
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory



def clean_data(data):
    #remove columns which are not needed
    #columns removed are sl_no, ssc_b, hsc_b, salary
    x_df = data.to_pandas_dataframe()
    x_df = x_df.drop(columns=['sl_no','ssc_b','hsc_b','salary'])
    x_df = x_df.dropna()

    # Clean and one hot encode data
    x_df["gender"] = x_df.gender.apply(lambda s: 1 if s == "M" else 0)
    x_df["specialisation"] = x_df.specialisation.apply(lambda s: 1 if s == "Mkt&Fin" else 0)
    x_df["workex"] = x_df.workex.apply(lambda s: 1 if s == "Yes" else 0)
    
    hsc = pd.get_dummies(x_df.hsc_s, prefix="hsc_stream")
    x_df.drop("hsc_s", inplace=True, axis=1)
    x_df = x_df.join(hsc)
    degree = pd.get_dummies(x_df.degree_t, prefix="degree")
    x_df.drop("degree_t", inplace=True, axis=1)
    x_df = x_df.join(degree)
    
    y_df = x_df.pop("status").apply(lambda s: 1 if s == "Placed" else 0)

    return x_df, y_df

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_estimators', type=int, default=100, help="The number of trees in the forest. High value gives better performance.")
    parser.add_argument('--min_samples_split', type=int, default=2, help="The minimum number of samples required to split an internal node")
    args = parser.parse_args()

    run = Run.get_context()
    ws = run.experiment.workspace
    run.log("n_estimators:", np.int(args.n_estimators))
    run.log("Minimum samples split:", np.int(args.min_samples_split))

    # TODO: Create TabularDataset using TabularDatasetFactory
    # Data is located at:
    # "https://raw.githubusercontent.com/webpagearshi/capstone-project/master/starter_file/placement_data_mba.csv"

    dataset_name = 'Placement Dataset'

# Get a dataset by name
    ds = Dataset.get_by_name(workspace=ws, name=dataset_name)
    #ds = TabularDatasetFactory.from_delimited_files(path = 
    #"https://raw.githubusercontent.com/webpagearshi/capstone-project/master/starter_file/placement_data_mba.csv")
    x, y = clean_data(ds)

    # TODO: Split data into train and test sets.

    ### YOUR CODE HERE ###a
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =0.2, random_state=223)

    model = RandomForestClassifier(n_estimators=args.n_estimators, min_samples_split=args.min_samples_split).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)

    run.log("Accuracy", np.float(accuracy)) 
    os.makedirs('outputs', exist_ok = True)

    
    joblib.dump(value = model, filename= 'outputs/model.joblib')

if __name__ == '__main__':
    main()
