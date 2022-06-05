import io
import sys

import warnings
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler, StandardScaler
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

# Используйте как путь откуда загрузить модель
MODEL_PATH = 'spark_ml_model'
SEED=42

log_classes = {
    'init': 'INIT',
    'metric': 'METRIC',
    'model': 'MODEL',
    'process': 'PROCESSING',
    'complete': 'COMPLETED'
}

def process(spark:SparkSession, train_data: Path, test_data: Path) -> None:
    """основной скрипт по валидации и получению параметров на тестовой выборке
       создает три модели, подбирает параметры и получает окончательные результаты на тестовой выборке
       сохраняет лучшую модель"""
    # читаем train и test
    train = spark.read.parquet(train_data, header=True)
    test = spark.read.parquet(test_data, header=True)
    
    _log(log_classes['init'], 'Train and test are loaded')
    # создаем эвалуатора для подсчета RMSE
    evaluator = RegressionEvaluator(
        predictionCol='prediction', labelCol="ctr", metricName="rmse")
    _log(log_classes['init'], 'Evaluator created. Metric is RMSE')
    # добавляем необходимые трансформации
    vector = VectorAssembler(
        inputCols=train.columns[1:-1], outputCol="features")
    _log(log_classes['init'], f'Features for processing are {train.columns[1:-1]}')
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features",
                        withStd=True, withMean=False)
    # получаем список моделей и параметров
    models = get_models()
    _log(log_classes['init'], 'Models are loaded')
    metrics = []
    best_models = {}
    # перебираем ключи в словаре моделей и производим валидацию и сбор результатов
    for key in models.keys():
        model = models[key]["model"]
        model_name = model.__class__.__name__
        _log(log_classes['process'], f'Working with {model_name}')
        params = models[key]["params"]
        # создаем пайплайн и tvs
        pipeline = Pipeline(stages=[vector, scaler, model])
        tvs = TrainValidationSplit(estimator=pipeline,
                                   estimatorParamMaps=params,
                                   evaluator=evaluator,
                                   trainRatio=0.8)
        _log(log_classes['process'], 'Starting parametrs tuning')
        # собираем метрики
        fit_model = tvs.fit(train)
        best_model = fit_model.bestModel
        best_metric = evaluator.evaluate(best_model.transform(test))
        _log(log_classes['metric'], f'The best metric on test for model {model_name} is {best_metric}')
        metrics.append(best_metric)
        best_models[key] = best_model
    # выбираем лучшую модель
    choose_model = get_best(metrics, best_models)
    _log(log_classes['model'], f'The best model is {model_name}')
    _log(log_classes['metric'], f'The best metric on test is {min(metrics)}')
    choose_model.write().overwrite().save(MODEL_PATH)
    _log(log_classes['complete'], f'Model is saved. Path is {MODEL_PATH}')
    spark.stop()
    _log(log_classes['complete'], 'Spark session is stoped')
    
def get_best(metrics: List[float], best_model:PipelineModel) -> PipelineModel:
    """отбираем лучшую модель"""
    idx = np.argmax(metrics)
    keys_list = list(best_model)
    key = keys_list[idx]
    return best_model[key]


def get_models():
    """создаем модели и фиксируем их сетку параметров для валидации"""
    dt = DecisionTreeRegressor(
        labelCol='ctr', featuresCol="scaled_features", seed=SEED)
    rf = RandomForestRegressor(
        labelCol='ctr', featuresCol="scaled_features", seed=SEED)
    gbt = GBTRegressor(labelCol='ctr', featuresCol="scaled_features", seed=SEED)

    models_info = {
        'dt': {'model': dt,
               'params': ParamGridBuilder()
                 .addGrid(dt.maxDepth, [2, 3, 4, 5])
                 .addGrid(dt.minInfoGain, [0.1, 0.2, 0.4])
                 .build()},
        'rf': {'model': rf,
               'params': ParamGridBuilder()
                 .addGrid(rf.maxDepth, [2, 3, 4, 5])
                 .addGrid(rf.numTrees, [5, 10, 12])
                 .build()},
        'gbt': {'model': gbt,
                'params': ParamGridBuilder()
                  .addGrid(gbt.maxDepth, [2, 3, 4, 5])
                  .addGrid(gbt.maxIter, [15, 20, 25])
                  .build()}
    }
    return models_info

def _log( cat: str, info: str) -> None:
    """дял логирования результатов"""
    record = f'{datetime.now()} {cat} {info}'
    print(record)


def main(argv):
    input_path = argv[0]
    print("Input path to file: " + input_path)
    output_file = argv[1]
    print("Output path to file: " + output_file)
    spark = _spark_session()
    process(spark, input_path, output_file)


def _spark_session():
    return SparkSession.builder.appName('PySparkMLPredict').getOrCreate()


if __name__ == "__main__":
    arg = sys.argv[1:]
    if len(arg) != 2:
        sys.exit("Input and Target path are require.")
    else:
        main(arg)
