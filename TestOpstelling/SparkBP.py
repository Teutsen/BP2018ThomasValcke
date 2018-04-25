
# coding: utf-8

# In[1]:


################################ IMPORTS ###################################
import os
import sys
import pyspark
from pyspark.sql import SparkSession
from pyspark import SparkContext
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator, DayLocator, WeekdayLocator, DateFormatter
from dateutil.relativedelta import relativedelta
### LINEAR REGRETION
from pyspark.ml.regression import LinearRegression
### RANDOM FOREST
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import IntegerType, TimestampType
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler


# In[2]:


SpSession = SparkSession     .builder     .master("local[2]")     .appName("BPSpark")     .config("spark.executor.memory", "1g")     .config("spark.cores.max","2")     .config("spark.sql.warehouse.dir", "file:///c:/temp/spark-warehouse")    .getOrCreate()


# In[3]:


#Get the Spark Context from Spark Session    
SpContext = SpSession.sparkContext
try:
    sc = pyspark.SparkContext(appName = 'App')
    spark = SparkSession(sparkContext=sc)
    print ("SparkSession initialized")
except ValueError:
    print ("SparkSession already initialized")


# In[4]:


################################ READ DATA ###################################
test1 = SpSession.read.option("delimiter", ";").option("header", "true").csv("D:/School/STAGE_BP/Stage_BachlerProef/datasets/jams2.csv")


# In[5]:


###test1 = test1[test1.to_numeric(test1.Length, errors='coerce').notnull()]
test1.show()


# In[6]:


# data cleaning: only Gent
#test1 = test1.filter(test1["City"] == "Gent")
#test1 = test1.filter(test1["PubTime"] >= "2017-11-12") 
# df = df.filter(df["PubTime"] <= "2018-03-02") 
test1 = test1.withColumn("Length",test1["Length"].cast(IntegerType()))     .withColumn("Delay",test1["Delay"].cast(IntegerType()))    .withColumn("Speed",test1["Speed"].cast(IntegerType()))    .withColumn("Level",test1["Level"].cast(IntegerType()))    .withColumn("RoadType",test1["RoadType"].cast(IntegerType()))    .withColumn("PubTime",test1["PubTime"].cast(TimestampType())) #"yyyy-MM-dd HH:mm"))


# In[7]:


# CLASSIFICATION
# API: http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.classification.RandomForestClassifier
# Split the data into training and test sets (20% held out for testing)
(trainingData, testData) = test1.randomSplit([0.8, 0.2])

# Concatenate all feature columns into 
# a single feature vector in a new column "rawFeatures".
featuresCols = trainingData.columns
featuresCols.remove('Jclass')#Jclass
vectorAssembler = VectorAssembler(inputCols=featuresCols, outputCol="rawFeatures")


# In[ ]:


test1 =test1.drop('PubTime').collect()


# In[ ]:


test1


# In[ ]:


# This identifies categorical features and indexes them.
vectorIndexer = VectorIndexer(inputCol="rawFeatures",                                      outputCol="features", maxCategories=6)

# Train a RandomForest model.
dt = RandomForestClassifier(numTrees=15,maxDepth=6,labelCol="Jclass")  # ca. 70% ##Jclass/Length

# Assemble the pipeline and apply the model
pipeline = Pipeline(stages=[vectorAssembler,vectorIndexer,dt])
model = pipeline.fit(trainingData)


# In[ ]:


test1.show(2)


# In[ ]:


test1.select('Length').show(10,False)


# In[ ]:


test2 = test1.select("Length","Speed")


# In[ ]:


test2.show(2)


# In[ ]:


################################ ML ###################################
from pyspark.ml.linalg import Vectors
test3 = test1.rdd.map(lambda x: [Vectors.dense(x[0:3]), x[-1]]).toDF(['Length', 'Speed'])
test3.show(5)




# In[ ]:


from pyspark.ml.regression import LinearRegression

# Load training data
##training = spark.read.format("libsvm")\
##    .load("data/mllib/sample_linear_regression_data.txt")

lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

# Fit the model
##lrModel = lr.fit(training)
lrModel = lr.fit(test2)
# Print the coefficients and intercept for linear regression
print("Coefficients: %s" % str(lrModel.coefficients))
print("Intercept: %s" % str(lrModel.intercept))

# Summarize the model over the training set and print out some metrics
trainingSummary = lrModel.summary
print("numIterations: %d" % trainingSummary.totalIterations)
print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
trainingSummary.residuals.show()
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)


# In[ ]:


### RANDOM FOREST
# Index labels, adding metadata to the label column.
# Fit on whole dataset to include all labels in index.
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)

# Automatically identify categorical features, and index them.
# Set maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

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

