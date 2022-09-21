from tensorflow.keras.losses import MAE, MSE
from tensorflow.keras.losses import mean_absolute_percentage_error as MAPE
import tensorflow.keras.backend as K
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np


def plot_history(history):    
    plt.plot(history.history['loss'], label='LOSS')
    plt.plot(history.history['val_loss'], label='VAL_LOSS')
    plt.legend()
    plt.show()


def plot_predict(y_test, pred):
    fig, ax = plt.subplots(3,1, figsize=(20,10))
    ax[0].set_ylim(bottom=0, top=1)
    ax[1].set_ylim(bottom=0, top=1)
    ax[2].set_ylim(bottom=0, top=1)
    ax[0].plot(y_test.squeeze(), '.b', markersize=2)
    ax[0].legend(['true'])
    ax[1].plot(pred.squeeze(), '.r', markersize=2)
    ax[1].legend(['pred'])
    ax[2].plot(y_test.squeeze(), '.b', markersize=2)
    ax[2].plot(pred.squeeze(), '.r', markersize=2)
    plt.show()

    
def add_korr_str(x):         # внесение случайных отклонений ПОСТРОЧНО
    korrs = ((-0.1, 0.1), (-1.5, 1.5), (-1, 1), (-50, 50), (-20, 20))
    x = list(np.transpose(x))                       
    for i in range(5):                     
        x[i] = x[i] + np.random.uniform(*korrs[i], size=len(x[i]))
    x = np.transpose(x) 
    return x


def regr_warm_fit(x_train, y_train):
    regr = RandomForestRegressor(1, criterion='squared_error', verbose=0, n_jobs=2, warm_start=True)
    y = y_train.squeeze()

    for i in range(100):
        x = add_korr_str(x_train)
        regr.fit(x, y)
        regr.n_estimators += 1

        print(i+1)
        clear_output(wait=True)
    return regr


def get_score(pred, y):
    SS_res =  np.square(y - pred).sum()
    SS_tot = np.square(y - y.mean()).sum()
    score = 1 - SS_res / SS_tot
    return score


def get_err(pred, y):        # подсчёт ошибки на чистых тестовых данных
    y = y.squeeze()
    pred = pred.squeeze()
    mae = MAE(y, pred).numpy().mean()            # mean absolute erro
    proc = MAPE(y, pred).numpy().mean()         # mean absolute percentage error
    score = get_score(pred, y)
    print(f'MAE: {mae:.2f}   {proc:.1f} %    score: {score:.2f}')

    
def show_result(model, x_test, y_true):
    pred = model.predict(x_test)
    get_err(pred, y_true)
    plot_predict(y_true, pred)
    
    
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


def generate_x(x, n=100):
    x_ = [add_korr_str(x) for _ in range(n)]
    x = np.vstack([*x_])
    return x


def generate_y(y, n=100):
    y_ = [y for _ in range(n)]
    y = np.vstack([*y_])
    return y


def get_only_indeces(x, ind):
    a = [x[:,i] for i in ind]
    return np.vstack([*a]).T


def getXYseq(x, y, window=64, step=8, y_seq=False, y_prob=False, threshold=0.1): 
    i = 0
    X = []
    Y = []
    while i + window <= len(x):
        X.append(x[i: i+window])
        if y_prob:
            prob = (y[i: i+window] >= threshold).sum() / window
            Y.append(prob)
        elif y_seq:
            Y.append(y[i: i+window])
        else:
            Y.append(y[i+window-1])
        i += step
    X = np.array(X)
    Y = np.array(Y)
    return X, Y
