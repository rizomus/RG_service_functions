from tensorflow.keras.losses import MAE, MSE
from tensorflow.keras.losses import mean_absolute_percentage_error as MAPE
import matplotlib.pyplot as plt
import numpy as np


def plot_history(history):    
    plt.plot(history.history['loss'], label='LOSS')
    plt.plot(history.history['val_loss'], label='VAL_LOSS')
    plt.legend()
    plt.show()


def add_korr_str(x):         # внесение случайных отклонений ПОСТРОЧНО
    korrs = ((-0.1, 0.1), (-1.5, 1.5), (-1, 1), (-50, 50), (-20, 20))
    x = list(np.transpose(x))                       
    for i in range(5):                     
        x[i] = x[i] + np.random.uniform(*korrs[i], size=len(x[i]))
    x = np.transpose(x) 
    return x


def get_score(pred, y):
    SS_res =  np.square(y - pred).sum()
    SS_tot = np.square(y - y.mean()).sum()
    score = 1 - SS_res / SS_tot
    return score


def get_err(pred, y):        # подсчёт ошибки на чистых тестовых данных
    mae = MAE(y, pred).numpy().mean()            # mean absolute erro
    proc = MAPE(y, pred).numpy().mean()         # mean absolute percentage error
    score = get_score(pred, y)
    print(f'MAE: {mae:.2f}   {proc:.1f} %    score: {score:.2f}')


def normalization(x):
    x_max = []                       
    x_min = []

    for i in range(x.shape[1]):
        x_max.append(x[:,i].max())
        x_min.append(x[:,i].min())
    x_max = np.array(x_max)
    x_min = np.array(x_min)
    
    return (x - x_min) / (x_max - x_min)


def coeff_determination(y_true, y_pred):            # RF score (1 - идеально, 0 - плохо, отрицательные значения - совсем плохо)
    SS_res =  K.sum(K.square(y_true-y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res/(SS_tot + K.epsilon())


def determination_loss(y_true, y_pred):                 # на основе RF score
    SS_res =  K.sum(K.square(y_true-y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return K.abs((1 - SS_res/(SS_tot + K.epsilon())) - 1)