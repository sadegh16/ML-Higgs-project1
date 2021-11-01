from  implementations import *

# load data
(yb, input_data, ids)=load_csv_data("train.csv")
# we change the labels from 1 and -1 to 1 and 0
yb=(yb+1)/2
# prepare data ( doing normalization, feature generation, data cleaning )
input_data, valid_columns, mu, sigma = prepare_input_data(input_data,degree=9)
# split input data to two parts of 20% validation and 80% train
train_X, val_X, train_y, val_y = train_val_split(yb, input_data)
# set the validation set
set_val_data(val_X,val_y)

# start loading the test data
(test_y, test_X, test_ids) =load_csv_data("test.csv")
test_X = prepare_test_data(test_X, mu, sigma,degree=9)
# run the best model on test data
weight=least_squares(train_y, train_X)
y_pred = predict_labels(weight,test_X)
create_csv_submission(test_ids,y_pred, "predict.csv")