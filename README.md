# Assignment 11 Submission
# Tasks performed
# 1. Create 80:10:10 train:dev:test split of the data.
# 2. Create smaller training sets that use 10,20,30,40,50,60,70,80,90,100% of the full training set.
# 3. For each of the smaller training sets, train the model (including hyper parameter tuning) and test of the same test set that was created in step 1.
# 4. Report the line chart that has on x-axis the percentage of training and y-axis test set macro f1.

## output obtained
</br></br>
![resultplot](https://user-images.githubusercontent.com/67168573/143916413-aeb4c0bf-3edf-40ef-9e2e-29b0b97965d2.png)

### Explanation
As we can see f1 score is improving by increasing the percentage of training data. Macro f1 score is very good measure to find whether model is able to perform well all the class of the data. Hence increasing training data size is increasing model performance of unknown data.

5. Make some observations about the chart and write.

</br></br>

6. Compare the actual predictions on test set using the model trained with  20% training data vs 10% training data, and 30% vs 20% so-on. Find out what will be good metric for comparing predictions of two models and use that for comparison. Note: don't just say "it has higher accuracy" -- that is aggregate metric comparison. What we are after is a "guarantee that is the model trained with larger data likely to be as good or better on each instance?" Hint: Confusion metrics.

# output obtained

![CompPlot](https://user-images.githubusercontent.com/67168573/143916559-04bab05e-14dd-4ac5-87c8-1dc36afcced2.png)\

### Explanation
ROC score is very good metric on multi class equal class distributed datasets. The above plot indicates that increase in training size increasing the ROC score.
