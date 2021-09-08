from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def metrics_wrapper(task):
    d = {"classification": [{'path': 'sklearn.metrics.accuracy_score',
                             'name': 'Accuracy'},
                            {'path': 'sklearn.metrics.balanced_accuracy_score',
                             'name': 'Balanced accuracy'},
                            {'path': 'sklearn.metrics.f1_score',
                             'name': 'F1 score'},
                            {'path': 'sklearn.metrics.roc_auc_score',
                             'name': 'AUC'},
                            {'path': 'sklearn.metrics.confusion_matrix',
                             'name': 'Confusion matrix'}
                            ],
         "regression": [{'path': 'sklearn.metrics.mean_absolute_error',
                         'name': 'Mean Absolute Error'},
                        {'path': 'sklearn.metrics.mean_squared_error',
                         'name': 'Mean Squared Error'},
                        {'path': 'sklearn.metrics.r2_score',
                         'name': 'R2 score'},
                        ]
         }
    return d[task]

def model_wrapper(model_option, task):
    d = {"classification": {"XGBoost": XGBClassifier(),
                            "CatBoost": CatBoostClassifier(),
                            "Random Forest": RandomForestClassifier()
                            },
         "regression": {"XGBoost": XGBRegressor(),
                        "CatBoost": CatBoostRegressor(),
                        "Random Forest": RandomForestRegressor()
                        }
        }
    return d[task][model_option]
