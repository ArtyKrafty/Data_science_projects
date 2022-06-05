import io
import sys

import warnings
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import (DecisionTreeRegressor,
                                       RandomForestRegressor,
                                       GBTRegressor)
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import (ParamGridBuilder, 
                               TrainValidationSplit)
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
from typing import List
import numpy as np
import logging

warnings.filterwarnings("ignore")

# Используйте как путь куда сохранить модель
MODEL_PATH = 'spark_ml_model'
SEED=42

log_classes = {
    'init': 'INIT',
    'metric': 'METRIC',
    'model': 'MODEL',
    'process': 'PROCESSING',
    'complete': 'COMPLETED'
}


def process(spark: SparkSession, input_file: Path, output_file: Path):
    # input_file - путь к файлу с данными для которых нужно предсказать ctr
    # output_file - путь по которому нужно сохранить файл с результатами [ads_id, prediction]
    inputs = spark.read.parquet(input_file)
    _log(log_classes['init'], 'Assets are loaded')
    # грузим модель
    model = PipelineModel.load(MODEL_PATH)
    _log(log_classes['process'], 'Getting predicions')
    # прогоняем инпуты и получаем предсказания
    outputs = model.transform(inputs)
    # сохраняем в csv
    interes = outputs.select('ad_id', 'prediction')
    _log(log_classes['process'], 'Example for preds data')
    interes.show(1)
    # сжимаем до одной партиции
    _log(log_classes['process'], 'Saving predicions')
    interes.coalesce(1) \
           .write.format("com.databricks.spark.csv") \
           .option("header", "true") \
           .save(output_file) \
    # останавливаем сессию
    spark.stop()
    _log(log_classes['complete'], 'Spark session is stoped')


def _log(cat: str, info: str) -> None:
    """для логирования результатов"""
    record = f'{datetime.now()} {cat} {info}'
    print(record)


def main(argv):
    train_data = argv[0]
    print("Input path to train data: " + train_data)
    test_data = argv[1]
    print("Input path to test data: " + test_data)
    spark = _spark_session()
    process(spark, train_data, test_data)


def _spark_session():
    return SparkSession.builder.appName('PySparkMLFitJob').getOrCreate()


if __name__ == "__main__":
    arg = sys.argv[1:]
    if len(arg) != 2:
        sys.exit("Train and test data are require.")
    else:
        main(arg)
