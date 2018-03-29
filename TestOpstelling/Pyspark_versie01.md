################################ IMPORTS ###################################
import pyspark
### LINEAR REGRETION
from pyspark.ml.regression import LinearRegression
### RANDOM FOREST
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
################################ READ DATA ###################################

training = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

################################ ML ###################################

### LINEAR REGRETION /// config only needed on windows + make temp file C: /temp
ss = SparkSession.builder.config("spark.sql.warehouse.dir","file:///C:/temp").appName("LinearRegression").getOrCreate()
# convert data to mllib
input = ss.sparkContext.testFile("regression.txt")
RDDdataLR = input.map(lambda w: w.split(",")).map(lambda x: (floar(x[0]), Vectors.dense(float(x[1])))
# convert RDD to dataframe
ColNames = ["label","feature"]
DFdataLR= RDDdataLR(ColNames)
# split data in to training and test data
trainTestLR=DFdataLR.randomSplit([0.5,0.5])
trainDFLR= trainTestLR[0]
testDFLR= trainTestLR[1]
# LinearRegression model
ModelLR = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
# train model with data
ModelresultLR = ModelLR.fit(trainDFLR)
# predict with test data
PredictLR = ModelresultLR.transform(testDFLR).cashe()
# show prediction
predictionsLR = PredictLR.select("prediction").rdd.map(lambda x: x[0])
labels = fullPredictions.select("label").rdd.map(lambda x: x[0])
predictionzip=predictionsLR.zip(labels).collect()
for prediction in predictionzip:
    print(prediction)


### RANDOM FOREST
# Index labels, adding metadata to the label column.
# Fit on whole dataset to include all labels in index.
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)

# Automatically identify categorical features, and index them.
# Set maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Train a RandomForest model.
rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)

# Convert indexed labels back to original labels.
labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                               labels=labelIndexer.labels)

# Chain indexers and forest in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])

# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
predictions.select("predictedLabel", "label", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))

rfModel = model.stages[2]
print(rfModel)  # summary only
