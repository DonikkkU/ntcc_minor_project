# %%
import pandas as pd
import os
import sys
import tqdm
import matplotlib.pyplot as plt
import joblib
import datetime
%matplotlib inline
import seaborn as sns
import numpy as np
import gc
import catboost as cb
import lightgbm as lgb
from datetime import datetime
from lightgbm import early_stopping
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from data_partition_loader import read_parquet_dataset_from_local, prepare_transactions_dataset



os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# pd.set_option('display.max_columns', None)

# %%
import warnings
warnings.filterwarnings("ignore")

# %%
cuda_path = 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4'
os.environ['CUDA_HOME'] = cuda_path
os.environ['PATH'] = cuda_path + ';' + os.environ['PATH']
os.environ['PATH'] = os.path.join(cuda_path, 'bin') + ';' + os.environ['PATH']
os.environ['PATH'] = os.path.join(cuda_path, 'libnvvp') + ';' + os.environ['PATH']

# Указываем, какой GPU использовать
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Проверка в PyTorch
import torch
print("CUDA is available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("CuDNN version:", torch.backends.cudnn.version())

# %% [markdown]
# ## Data Loading (target)
# 

# %%
df_target = pd.read_csv('../data/train_target.csv')
df_target.head()

# %%
train_features_path = "../data/train_data/train_data_baseline_final"

# %%
train_features_path = "../data/train_data/train_data_baseline_final"

# %%
train_data_path = '../data/train_data'

# %% [markdown]
# ## Data Aggregation and Feature Extraction

# %% [markdown]
# ### Класс CountAggregator 
# предназначен для обработки больших наборов данных по частям, выполняя агрегирование на основе количества после применения one-hot кодирования к категориальным признакам
# One-Hot Кодирование:
# 
# Извлекает столбцы признаков, исключая id и rn
# Применяет one-hot кодирование к этим столбцам признаков
# Добавляет полученные dummy переменные к исходному data_frame
# Агрегирование на основе количества:
# 
# Группирует данные по id и суммирует one-hot кодированные столбц
# Преобразует суммированные признаки в тип данных float32
# Циклически обрабатывает набор данных по частям, обрабатывая указанное количество частей за раз
# Читает части данных с помощью функции read_parquet_dataset_from_local
# Применяет агрегирование на основе количества с помощью метода __extract_count_aggregations
# Конкатенирует обработанные части
# Заполняет отсутствующие значения нулями
# Настраивает столбцы признаков в зависимости от режима.
# В режиме "fit_transform" сохраняет имена кодированных признаков
# В режиме "transform" убеждается, что все ожидаемые признаки присутствуют
# Преобразует столбцы признаков в тип данных uint8 для экономии памяти
# Возвращает обработанный DataFrame

# %%
class CountAggregator(object):
    
    def __init__(self):
        self.encoded_features = None
        
    def __extract_count_aggregations(self, data_frame: pd.DataFrame, mode: str) -> pd.DataFrame:
        # one-hot-encoding
        feature_columns = list(data_frame.columns.values)
        feature_columns.remove("id")
        feature_columns.remove("rn")

        dummies = pd.get_dummies(data_frame[feature_columns], columns=feature_columns)

        for col in dummies.columns:
            data_frame[col] = dummies[col]

        # count aggregation
        features = data_frame.groupby("id")[dummies.columns].sum().reset_index(drop=False)
        features[dummies.columns] = features[dummies.columns].astype(np.float32)
        return features
        
    def __transform_data(self, path_to_dataset: str, num_parts_to_preprocess_at_once: int = 1, num_parts_total: int=50,
                         mode: str = "fit_transform", save_to_path=None, verbose: bool=False):
        assert mode in ["fit_transform", "transform"], f"Unrecognized mode: {mode}! Please use one of the following modes: \"fit_transform\", \"transform\""
        preprocessed_frames = []
        for step in tqdm.tqdm_notebook(range(0, num_parts_total, num_parts_to_preprocess_at_once), 
                                        desc="Transforming sequential credit data"):
            data_frame = read_parquet_dataset_from_local(path_to_dataset, start_from=step, 
                                                         num_parts_to_read=num_parts_to_preprocess_at_once, 
                                                         verbose=verbose)
            features = self.__extract_count_aggregations(data_frame, mode=mode)
            if save_to_path:
                features.to_parquet(os.path.join(save_to_path, f"processed_chunk_{step}.pq"))
            preprocessed_frames.append(features)
        
        features = pd.concat(preprocessed_frames)
        features.fillna(0, inplace=True)
        dummy_features = list(features.columns.values)
        dummy_features.remove("id")
        if mode == "fit_transform":
            self.encoded_features = dummy_features
        else:
            assert self.encoded_features is not None, "Transformer not fitted"
            for col in self.encoded_features:
                if col not in dummy_features:
                    features[col] = 0
        # Использование менне потребительных типов данных
        features[dummy_features] = features[dummy_features].astype(np.uint8)
        return features[["id"] + self.encoded_features]
    
    def fit_transform(self, path_to_dataset: str, num_parts_to_preprocess_at_once: int = 1, num_parts_total: int = 50,
                      save_to_path=None, verbose: bool=False):
        return self.__transform_data(path_to_dataset=path_to_dataset,
                                     num_parts_to_preprocess_at_once=num_parts_to_preprocess_at_once,
                                     num_parts_total=num_parts_total, mode="fit_transform",
                                     save_to_path=save_to_path, verbose=verbose)
    
    def transform(self, path_to_dataset: str, num_parts_to_preprocess_at_once: int = 1, num_parts_total: int=50,
                  save_to_path=None, verbose: bool=False):
        return self.__transform_data(path_to_dataset=path_to_dataset,
                                     num_parts_to_preprocess_at_once=num_parts_to_preprocess_at_once,
                                     num_parts_total=num_parts_total, mode="transform",
                                     save_to_path=save_to_path, verbose=verbose)



# %%
%%time
aggregator = CountAggregator()
train_data = aggregator.fit_transform(train_data_path, num_parts_to_preprocess_at_once=4, num_parts_total=12, 
                                      save_to_path=train_features_path, verbose=True)

# %%
train_data.shape

# %%
train_data.head(5)

# %% [markdown]
# ### Modeling LightGBM на кросс-валидации

# %%
df_final = df_target.merge(train_data, on='id')
df_final.shape

# %%
df_final.isna().sum().sum()

# %%
features = list(df_final.columns.values)
features.remove("id"), features.remove("flag")
len(features)

# %%
target = df_final['flag'].values

# %%
cv = KFold(n_splits=5, random_state=42, shuffle=True)
out_of_fold = np.zeros(len(df_final))
preds = np.zeros(len(df_final))
early_stopping_callback = early_stopping(stopping_rounds=50, verbose=50)
models = []

params = {
    'learning_rate': 0.05,
    'max_depth': 5,
    'reg_lambda': 1,
    'objective': 'binary',
    'num_leaves': 64,
    'n_jobs': -1,
    'n_estimators': 2000,
     'metric': 'auc',
    'scale_pos_weight': len(df_final[df_final.flag == 0]) / len(df_final[df_final.flag == 1]) 
}

assert np.isfinite(df_final.values).all(), "Data contains infinite values."

for fold, (train_index, val_index) in enumerate(cv.split(df_final, target), 1):
    print(f"Training in fold {fold} started")
    lgb_model = lgb.LGBMClassifier(**params)
    train, val = df_final.iloc[train_index], df_final.iloc[val_index]

    lgb_model.fit(train[features], train.flag.values, eval_set=[(val[features], val.flag.values)],
                  callbacks=[early_stopping_callback])
    
    out_of_fold[val_index] = lgb_model.predict_proba(val[features])[:, 1]
    preds[train_index] += lgb_model.predict_proba(train[features])[:, 1] / (cv.n_splits-1)
    models.append(lgb_model)
    print(f"Training with fold {fold} completed")

print("Out-of-fold predictions mean:", np.mean(out_of_fold))

# %%
print('roc-auc:', roc_auc_score(target, preds))

# %%
print('CV roc-auc:', roc_auc_score(target, out_of_fold))

# %%
accuracy_roc_auc   =  roc_auc_score(target, out_of_fold)
accuracy_roc_auc

# %% [markdown]
# ## Finals

# %%
best_iter = lgb_model.best_iteration_
val_auc = lgb_model.best_score_['valid_0']['auc']
if val_auc > best_score:
    best_model = lgb_model.booster_
    best_score = val_auc

# %%
joblib.dump({
            'model': best_model,
            'metadata': {
                         'name': 'сredit_score_classifier',
                         'author': 'Donik_system',
                         'version': 1,
                         'date': datetime.now(),
                         'type': type(best_model),
                         'accuracy_roc_auc': best_score
                        }
        }, 'best_lgb_model.pkl')

# %%
model = joblib.load('best_lgb_model.pkl')
model['metadata']

# %%



