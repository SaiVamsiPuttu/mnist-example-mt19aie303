# Assignment 6 Submission
# tasks performed:

## TODO: write  a test case to check if model is successfully getting created or not?
def test_model_writing():

    1. create some data

    2. run_classification_experiment(data, expeted-model-file)

    assert os.path.isfile(expected-model-file)


## TODO: write a test case to check fitting on training -- litmus test.

def test_small_data_overfit_checking():

    1. create a small amount of data / (digits / subsampling)

    2. train_metrics = run_classification_experiment(train=train, valid=train)

    assert train_metrics['acc']  > some threshold

    assert train_metrics['f1'] > some other threshold
    
    
 # End result!
 
 [assignment6_output](https://user-images.githubusercontent.com/67168573/136434238-75e7c1ea-da7b-4377-a378-1ad6fc198745.png)
 
