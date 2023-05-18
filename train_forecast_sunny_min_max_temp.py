import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow_decision_forests as tfdf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.metrics import Precision, Recall, AUC
from keras_tuner import RandomSearch
from sklearn.model_selection import train_test_split
from common_functions import *

np.set_printoptions(precision=3, floatmode="fixed", suppress=True, linewidth=200)

file = "data_min_max_mean_sorted.csv"
data = pd.read_csv(file)
EPOCHS = 50 #60
FINAL_TRAINING_EPOCHS = 60
MIN_LAYERS = 2
MAX_LAYERS = 3
MIN_NEURONS = 10
MAX_NEURONS = 250
LEARNING_RATES = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
TRIALS = 10 #30
RUNS_PER_TRIAL = 3 #3

def generate_training_data():
    all_measurements = list(group_columns(file))
    X = []
    y_sunny_hours = []
    y_max_temp = []
    y_min_temp = []
    y_below_0 = []
    y_windy = []
    y_rainy = []
    y_temperature = []
    y_solarradiation_max = []
    y_solarradiation_mean = []
    hours_backwards = 24 * 2
    for idx in range(len(all_measurements) - hours_backwards - 24):
        xx = []
        yy_sunny_hours = []
        yy_min_temp = []
        yy_max_temp = []
        yy_below_0 = []
        yy_windy = []
        yy_rainy = []
        yy_temperature = []
        yy_solarradiation_max = []
        yy_solarradiation_mean = []
        for input_row in all_measurements[idx:idx+hours_backwards]:
            xx.extend(input_row)
        for output_data in all_measurements[idx + hours_backwards:idx + hours_backwards + 24]:
            yy_sunny_hours.append(output_data[0])
            yy_min_temp.append(output_data[1])
            yy_max_temp.append(output_data[2])
            yy_below_0.append(output_data[3])
            yy_windy.append(output_data[4])
            yy_rainy.append(output_data[5])
            yy_temperature.append(output_data[6])
            yy_solarradiation_max.append(output_data[13])
            yy_solarradiation_mean.append(output_data[14])
            

        X.append(xx)
        y_sunny_hours.append(yy_sunny_hours)
        y_max_temp.append(max(yy_max_temp))
        y_min_temp.append(min(yy_min_temp))
        y_below_0.append(max(yy_below_0))
        y_windy.append(yy_windy)
        y_rainy.append(yy_rainy)
        y_temperature.append(yy_temperature)
        y_solarradiation_max.append(yy_solarradiation_max)
        y_solarradiation_mean.append(yy_solarradiation_mean)

    X = np.array(X)
    y_sunny_hours = np.array(y_sunny_hours)
    y_max_temp = np.array(y_max_temp)
    y_min_temp = np.array(y_min_temp)
    y_below_0 = np.array(y_below_0)
    y_windy = np.array(y_windy)
    y_rainy = np.array(y_rainy)
    y_temperature = np.array(y_temperature)
    y_solarradiation_max = np.array(y_solarradiation_max)
    y_solarradiation_mean = np.array(y_solarradiation_mean)

    return X, y_sunny_hours, y_min_temp, y_max_temp, y_below_0, y_windy, y_rainy, y_temperature, y_solarradiation_max, y_solarradiation_mean

X, y_sunny_hours, y_min_temp, y_max_temp, y_below_0, y_windy, y_rainy, y_temperature, y_solarradiation_max, y_solarradiation_mean = generate_training_data()
global input_shape
input_shape = X.shape[1:]

models = {
    'wind_model': ['forecast_windy', y_windy],
    'rain_model': ['forecast_rainy', y_rainy],
    'sun_model': ['forecast_sunny_hours', y_sunny_hours],
    'below_0_model': ['forecast_below_0', y_below_0],
    'min_temp_model': ['forecast_min_temp', y_min_temp],
    'max_temp_model': ['forecast_max_temp', y_max_temp],
    'temperature_model': ['forecast_temperature_LSTM', y_temperature],
    'solarradiation_max_model': ['forecast_solarradiation_max_LSTM', y_solarradiation_max],
    'solarradiation_mean_model': ['forecast_solarradiation_mean_LSTM', y_solarradiation_mean]
}

def build_model_multiple_binary(hp):
    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))

    for i in range(hp.Int("num_layers", min_value=MIN_LAYERS, max_value=MAX_LAYERS, step=1)):
        model.add(layers.Dense(units=hp.Int(f"units_{i}", min_value=MIN_NEURONS, max_value=MAX_NEURONS, step=32), activation="relu"))
        
    model.add(layers.Dense(24, activation="sigmoid"))

    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice("learning_rate", values=LEARNING_RATES)
        ),
        loss="binary_crossentropy",
        #metrics=["mae"]
        metrics=["accuracy", Precision(name="precision"), Recall(name="recall"), AUC(name="auc")]
    )
    return model

def build_model_multiple_binary_24_values_LSTM(hp):
    model = keras.Sequential()
    model.add(layers.Input(shape=(48, 17)))

    for i in range(hp.Int("num_layers", min_value=MIN_LAYERS, max_value=MAX_LAYERS, step=1)):
        if i == 0:
            model.add(layers.LSTM(units=hp.Int(f"units_{i}", min_value=MIN_NEURONS, max_value=MAX_NEURONS, step=32),
                                  activation="tanh", return_sequences=True))  # Ensure return_sequences=True for all but the last LSTM layer
        else:
            model.add(layers.LSTM(units=hp.Int(f"units_{i}", min_value=MIN_NEURONS, max_value=MAX_NEURONS, step=32),
                                  activation="tanh", return_sequences=(i != hp.Int("num_layers", min_value=MIN_LAYERS, max_value=MAX_LAYERS, step=1)-1)))

    model.add(layers.Flatten())  # Flatten the output before connecting to the final Dense layer
    model.add(layers.Dense(24, activation="sigmoid"))  # 24 units and "linear" activation

    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice("learning_rate", values=LEARNING_RATES)
        ),
        loss="binary_crossentropy",
        metrics=["accuracy", Precision(name="precision"), Recall(name="recall"), AUC(name="auc")]
    )
    model.summary()
    return model

def build_model_regression_1_value(hp):
    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))

    for i in range(hp.Int("num_layers", min_value=MIN_LAYERS, max_value=MAX_LAYERS, step=1)):
        model.add(layers.Dense(units=hp.Int(f"units_{i}", min_value=MIN_NEURONS, max_value=MAX_NEURONS, step=32), activation="relu"))
        
    model.add(layers.Dense(1, activation="linear"))

    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice("learning_rate", values=LEARNING_RATES)
        ),
        loss="mse",
        metrics=["mae"]
    )
    return model

def build_model_regression_24_values(hp):
    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))

    for i in range(hp.Int("num_layers", min_value=MIN_LAYERS, max_value=MAX_LAYERS, step=1)):
        model.add(layers.Dense(units=hp.Int(f"units_{i}", min_value=MIN_NEURONS, max_value=MAX_NEURONS, step=32), activation="relu"))
        
    model.add(layers.Dense(24, activation="linear"))  # 24 units and "linear" activation

    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice("learning_rate", values=LEARNING_RATES)
        ),
        loss="mse",
        metrics=["mae"]
    )
    return model

def build_model_regression_24_values_LSTM(hp):
    model = keras.Sequential()
    model.add(layers.Input(shape=(48, 17)))

    for i in range(hp.Int("num_layers", min_value=MIN_LAYERS, max_value=MAX_LAYERS, step=1)):
        if i == 0:
            model.add(layers.LSTM(units=hp.Int(f"units_{i}", min_value=MIN_NEURONS, max_value=MAX_NEURONS, step=32),
                                  activation="tanh", return_sequences=True))  # Ensure return_sequences=True for all but the last LSTM layer
        else:
            model.add(layers.LSTM(units=hp.Int(f"units_{i}", min_value=MIN_NEURONS, max_value=MAX_NEURONS, step=32),
                                  activation="tanh", return_sequences=(i != hp.Int("num_layers", min_value=MIN_LAYERS, max_value=MAX_LAYERS, step=1)-1)))

    model.add(layers.Flatten())  # Flatten the output before connecting to the final Dense layer
    model.add(layers.Dense(24, activation="linear"))  # 24 units and "linear" activation

    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice("learning_rate", values=LEARNING_RATES)
        ),
        loss="mse",
        metrics=["mae"]
    )
    return model

def build_model_binary(hp):
    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))

    for i in range(hp.Int("num_layers", min_value=MIN_LAYERS, max_value=MAX_LAYERS, step=1)):
        model.add(layers.Dense(units=hp.Int(f"units_{i}", min_value=MIN_NEURONS, max_value=MAX_NEURONS, step=32), activation="relu"))
        
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice("learning_rate", values=LEARNING_RATES)
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

def evaluatemodels():
    for modelname, modelinfo in models.items():
        model = tf.keras.models.load_model(modelinfo[0])
        test_data = modelinfo[1]
        print("   ")
        print("   ")
        print(modelname,' :-----------------------------------------------------------------------------------------------------------------------------------------')
        loss, accuracy = model.evaluate(X, test_data)
        model.summary()
        
def retrain_models():
    old_metrics = {}
    new_metrics = {}
    for modelname, modelinfo in models.items():
        X_train, X_test, y_train, y_test = train_test_split(X, modelinfo[1], test_size=0.2, random_state=42)
        model = tf.keras.models.load_model(modelinfo[0])
        actual_loss, actual_accuracy = model.evaluate(X_test, y_test)
        old_metrics[modelname] = {'loss': actual_loss, 'accuracy': actual_accuracy}

        optimizer_config = model.optimizer.get_config()
        metrics_config = [metric.get_config() for metric in model.metrics]

        loss = model.loss.__name__
        optimizer = optimizer_config['name']
        metrics = metrics_config[1]['name']

        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        history = model.fit(X_train, y_train, batch_size=32, epochs=100, validation_split=0.2)
        new_loss, new_accuracy = model.evaluate(X_test, y_test)
        new_metrics[modelname] = {'loss': new_loss, 'accuracy': new_accuracy}
        model.save(modelinfo[0]+'retrained')

    for modelname, _ in old_metrics.items():
        print(old_metrics[modelname], new_metrics[modelname])

def train_tune_model(model_name, project_name, objective, y_data, model_function):
    X_train, X_test, y_train, y_test = train_test_split(X, y_data, test_size=0.2, random_state=42)
    forecast_model_tuner = RandomSearch(
        model_function,
        objective=objective,
        max_trials=TRIALS,
        executions_per_trial=RUNS_PER_TRIAL,
        directory=project_name,
        project_name=project_name
    )
    
    forecast_model_tuner.search_space_summary()
    forecast_model_tuner.search(X_train, y_train, epochs=EPOCHS, validation_split=0.2)

    best_model = forecast_model_tuner.get_best_models(num_models=1)[0]
    best_model.build(input_shape)
    best_model.summary()
    best_hyperparameters = forecast_model_tuner.get_best_hyperparameters(num_trials=1)[0]

    best_model.fit(X_train, y_train, epochs=FINAL_TRAINING_EPOCHS)
    try:
        test_loss, test_mae = best_model.evaluate(X_test, y_test)
        print("Test ",objective,": ", test_mae)
    except:
        test_loss, test_accuracy, test_precision, test_recall, test_auc = best_model.evaluate(X_test, y_test)
        print("Test ",objective,": ", test_loss, test_accuracy, test_precision, test_recall, test_auc)
    

    best_model.save(model_name)

#train_tune_model(model_name = 'forecast_sunny_hours', project_name='sunny_hours', objective='val_mae', y_data=y_sunny_hours, model_function=build_model_multiple_binary)
#train_tune_model(model_name = 'forecast_min_temp', project_name='min_temp', objective='val_mae', y_data=y_min_temp, modelt_function=build_model_regression_1_value)
#train_tune_model(model_name = 'forecast_max_temp', project_name='max_temp', objective='val_mae', y_data=y_max_temp, model_function=build_model_regression_1_value)
#train_tune_model(model_name = 'forecast_below_0', project_name='below_0', objective='val_accuracy', y_data=y_below_0, model_function=build_model_binary)
#train_tune_model(model_name = 'forecast_rainy', project_name='rainy', objective='val_accuracy', y_data=y_rainy, model_function=build_model_multiple_binary)
#train_tune_model(model_name = 'forecast_windy', project_name='windy', objective='val_accuracy', y_data=y_windy, model_function=build_model_multiple_binary)
#print(y_rainy.shape)
X = X.reshape(X.shape[0], 48, 17)
#for data in y_rainy:    
#    print(data)
#train_tune_model(model_name = 'forecast_rainy', project_name='rainy', objective='val_accuracy', y_data=y_rainy, model_function=build_model_multiple_binary_24_values_LSTM)
#train_tune_model(model_name = 'forecast_windy', project_name='windy', objective='val_accuracy', y_data=y_windy, model_function=build_model_multiple_binary_24_values_LSTM)
#train_tune_model(model_name = 'forecast_temperature_LSTM', project_name='temperature', objective='val_mae', y_data=y_temperature, model_function=build_model_regression_24_values_LSTM)
#train_tune_model(model_name = 'forecast_solarradiation_max', project_name='solarradiation_max', objective='val_mae', y_data=y_solarradiation_max, model_function=build_model_regression_24_values_LSTM)
#train_tune_model(model_name = 'forecast_solarradiation_mean', project_name='solarradiation_mean', objective='val_mae', y_data=y_solarradiation_mean, model_function=build_model_regression_24_values_LSTM)

#evaluatemodels()
#retrain_models()

#benchmark: 781 - 949 us per step

def build_model():
    model = keras.Sequential()
    
    # LSTM layer with 128 units and input shape (48, 17)
    model.add(layers.LSTM(192, input_shape=(48, 17)))

    # Flatten layer
    model.add(layers.Flatten())

    # Dense layer with 24 units and sigmoid activation
    model.add(layers.Dense(24, activation="sigmoid"))

    # Compile the model
    model.compile(
        optimizer='adam',
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall"), tf.keras.metrics.AUC(name="auc")]
    )

    # Print the model summary
    model.summary()

    return model

# Create the model
X_train, X_test, y_train, y_test = train_test_split(X, y_windy, test_size=0.2, random_state=42)
model = build_model()
model.fit(X_train, y_train, epochs=FINAL_TRAINING_EPOCHS)
test_loss, test_accuracy, test_precision, test_recall, test_auc = model.evaluate(X_test, y_test)
print("Test ", test_loss, test_accuracy, test_precision, test_recall, test_auc)
model.save("test_windy")