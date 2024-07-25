import pandas as pd
import numpy as np
import seaborn as sns
import os

#Regression metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Classification metrics
from sklearn.metrics import (accuracy_score, f1_score,
                            recall_score, precision_score,
                            roc_auc_score)
                            

#1 Получение пути к данным
def path_check(path_name):
  '''Проверка существования пути к файлу'''
  if os.path.exists(os.getcwd()+path_name):
      print('OK - Path exists')
      return os.getcwd()+path_name
  else:
      raise EnvironmentError('Нет пути к данным') 

#2 Оценка количества пропущенных значений
def unn_info_pct(data):
  '''Вывод информации о пропущенных значениях
  Аргумент
    data: pd.DataFrame
  Возвращаемые данные:
    pd.DataFrame с количественным и %-ным представлением о пропущенных значениях'''
  return pd.DataFrame([data.isna().sum(), data.isna().sum() / len(data) * 100]).T

#3 Расчёт размаха признака
def feature_range(data):
    '''Функция расчёта размаха признака
    data: объект pd.Series - признак, размах которого нужно посчитать.
    '''
    return data.max() - data.min()

#4 Поиск дубликатов
def duplicate_search(dataset, cols_search = []):
    '''Поиск дубликатов в датасете
    dataset : pd.DataFrame - датасет, в котором необходимо провести поиск
    cols_search : list - итерируемый объект, в который передаются колонки для поиска'''
    if len(cols_search) == 0:
        cols_search = dataset.columns.tolist()
    else:
        pass
    for i in cols_search:
        print(f'Значения признака {i}:', sorted(dataset[i].unique()))

#5 Функция отображения гистограмм по признакам
def hist_plotting(dat, x:str, hue: list, text, 
                    plots_grid = (2, 2),
                      fig_s = (8, 6), dpi = 100):
    '''Построение гистограмм
    dat : pd.DataFrame;
    x: str - признак, который необходимо визуализировать;
    hue: list - список, содержащий дополнительные признаки для построения гистограмм с различными категориями (!обязательный параметр);
    plots_grid: настройка размеров окон графика;
    fig_s, dpi: настройка размеров и качества графика;
    '''
    fig, axs = plt.subplots(plots_grid[0], plots_grid[1], figsize = fig_s, constrained_layout = True, dpi = dpi)
    if len(hue) < plots_grid[0]*plots_grid[0]:
        pass
    else:
        raise AttributeError('Измените параметр plots_grid для отображения всех графиков.')
    for i, j in enumerate(hue):
        pos = axs[int(np.floor(i/2)), i%2]
        sns.countplot(data = dat, x = x, hue = j, ax=pos)
    plt.suptitle(text)
    plt.show()
    return fig, axs

#6 Фукнция отображения цвета seaborn
def seaborn_color(color:str):
    return sns.xkcd_rgb[color]

#7 Группировка данных и подсчёт % по передаваемым категориям
def pivot_table_analysis(data, indexes):
    '''Функция для расчёта количества объектов по категориям с учётом расчёта процентов
    data: объект pd.DataFrame с данными для расчёта;
    indexes: итерируемый объект list/set, 0 аргумент - верхний уровень группировки, 1 аргумент - нижний уровень группировки'''
    temp1 = pd.pivot_table(data = data, index = indexes, aggfunc = 'count')[data.columns.tolist()[0]]
    temp2 = 100*round(data.groupby(indexes).count()['id'] / data.groupby(indexes[0]).count()[data.columns.tolist()[0]], 4)
    return pd.concat([temp1, temp2], axis = 1)

#8 Расчёт метрик качества регрессии
def regr_metrics(model, data, target_true):
    '''Могут быть проблемы с работой функции, так как она написана под старую версию sklearn'''
    pred_temp = model.predict(data) 
    metrics = pd.DataFrame({'Metrics':['R2_score', 'RMSE_score', 'MAE_score', 'SMAPE_Score'],
                            'Scores': [r2_score(target_true.values, pred_temp),
                           mean_squared_error(target_true.values, pred_temp, squared=False),
                           mean_absolute_error(target_true.values, pred_temp),
                           smape_score(target_true.values, pred_temp)]})
    return metrics

#9 Построение графиков остатков для анализа моделей регрессии
def residuals_plotting(true, predict):
    fig, axs = plt.subplots(1, 2, figsize = (10, 6), constrained_layout = True, dpi = 120)
    plt.suptitle('Исследование остатков')
    residuals = true - predict
    #график 1 Распределение остатков
    sns.histplot(residuals, kde=True, ax = axs[0])
    axs[0].set_xlabel('Величина ошибки модели')
    axs[0].set_ylabel('Количество ошибок')
    axs[0].set_title('Распределение остатков')
    axs[0].vlines([np.mean(residuals), np.median(residuals)],
           ymin = 0, ymax = 130, colors=['red', 'green'], linewidths = [2, 2])
    
    #график 2 Исследование дисперсии остатков RMSE
    sns.scatterplot(x = predict, y = np.sqrt((residuals)**2), color = 'Seagreen')
    axs[1].set_xlabel('Прогнозная величина')
    axs[1].set_ylabel('Величина ошибки')
    axs[1].set_title('Исследование дисперсии остатков')
    plt.show()
    return fig, axs

#10 Поиск опорных векторов
def find_support_vectors(svr, X, y):
    y_pred = svr.predict(X)
    epsilon = svr[-1].epsilon
    off_margin = np.abs(y - y_pred) >= epsilon
    return np.argwhere(off_margin.values).T[0]

#11 Поиск выбросов (Z-значения)
def outliers_search(data, z_score):
    mean = np.mean(data)
    std = np.std(data)
    return np.where(data > z_score, True, False)

#12 Расчёт метрик классификации
def clf_metrics(y_true, y_pred, y_pred_prob = None, roc_auc_calc = False):
    metrics = {'Recall' : recall_score(y_true, y_pred),
               'F1': f1_score(y_true, y_pred),
               'Accuracy': accuracy_score(y_true, y_pred),
              'Precision': precision_score(y_true, y_pred)}
    if roc_auc_calc == True:
        metrics.update({'Roc-Auc': roc_auc_score(y_true, y_pred_prob)})
        return pd.DataFrame(metrics, index=[i for i in range(len(metrics))]).T.drop([i for i in range(1, 5)], axis = 1)
    else:
        return pd.DataFrame(metrics, index=[i for i in range(len(metrics))]).T.drop([i for i in range(1, 4)], axis = 1)
      
#13 Поиск пропусков
def na_count(data, feat, retr_dat = False):
  na_n = data[feat].isna().sum()
  print(f'Число пропусков в признаке {feat}: ', na_n)
  if retr_dat:
    return na_n
  else:
    pass