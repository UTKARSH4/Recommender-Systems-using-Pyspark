from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import *
from pyspark.sql import SQLContext
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import *
import matplotlib.pyplot as plt
import logging
import logging.config
import sys

from create_spark import create_spark_object
from validate import get_current_date
from ingest import load_files, display_df, df_count

logging.config.fileConfig('properties/configuration/logging.config')


def main():
    try:
        logging.info('I am the main method')
        logging.info('calling spark object')
        spark = create_spark_object()

        #logging.info('object create...', str(spark))
        logging.info('validating spark object')
        get_current_date(spark)

        logging.info('reading file....')
        df = load_files(spark, file_dir='/home/utkarsh/Downloads/Ratings.csv')
        logging.info('displaying the dataframe... ')
        display_df(df,'df')
        logging.info('validating the dataframe... ')
        df_count(df,'df')

        # EDA
        ratings = df.select('Book-Rating').toPandas()
        # review_list = [reviews[i][0] for i in range(len(reviews))]
        # print(ratings['Book-Rating'].tolist())
        print("Unique count check", ratings['Book-Rating'].value_counts())
        print("Null check", ratings['Book-Rating'].isna().sum())
        plt.hist(ratings, alpha=0.5,
                 histtype='stepfilled', color='blue',
                 edgecolor='none')
        plt.ylabel('Frequency')
        plt.xlabel('Ratings')
        # plt.show()

        indexer = [StringIndexer(inputCol=column, outputCol=column + "_index") for column in ['ISBN', 'User-ID']]
        pipeline = Pipeline(stages=indexer)
        transformed = pipeline.fit(df).transform(df)
        # transformed.select(['User-ID', 'ISBN', 'User-ID_index', 'ISBN_index','Book-Rating'])

        (training, test) = transformed.randomSplit([0.8, 0.2])

        als_model = ALS(maxIter=5,
                        regParam=0.09,
                        rank=25,
                        userCol="User-ID_index",
                        itemCol="ISBN_index",
                        ratingCol="Book-Rating",
                        coldStartStrategy="drop",
                        nonnegative=True)

        model = als_model.fit(training)

        eval = RegressionEvaluator(metricName="rmse", labelCol="Book-Rating", predictionCol="prediction")
        predictions = model.transform(test)
        rmse = eval.evaluate(predictions)
        print("RMSE=" + str(rmse))


    except Exception as e:
        logging.error('An error occured ===', str(e))
        sys.exit(1)


if __name__ == '__main__':
    main()
    logging.info('Application done')


