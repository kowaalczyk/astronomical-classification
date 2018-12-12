from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adadelta
import keras.backend as K
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plasticc.models.utils import multi_weighted_logloss as mwl_common


def multi_weighted_logloss(y_true, y_pred):  # NOTE: weights must be sorted, y_true and y_pred - one-hot encoded
    class_weights = K.variable([1., 2., 1., 1., 1., 1., 1., 2., 1., 1., 1., 1., 1., 1.])  # from kaggle
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_pred = K.clip(x=y_pred, min_value=1e-15, max_value=1 - 1e-15)
    # Transform to log
    y_p_log = K.log(y_pred)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set
    # we gave a special process for that class
    y_log_ones = K.sum(y_true * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = K.sum(y_true, axis=0) + K.epsilon()  # prevent nans in division later
    # Weight average and divide by the number of positives
    y_w = y_log_ones * class_weights / nb_pos
    loss = - K.sum(y_w) / K.sum(class_weights)
    return loss


def build_classifier(
        input_dim: int, 
        num_classes: int=14, 
        num_hidden_layers: int=3, 
        hidden_layer_dim: int=128, 
        dropout_pct: float=0.5,
        optimizer_lr: float=0.25,
):
    model = Sequential()
    for i in range(num_hidden_layers):
        if i == 0:
            model.add(Dense(hidden_layer_dim, activation='tanh', input_dim=input_dim))
        else:
            model.add(Dense(hidden_layer_dim, activation='tanh'))
        model.add(Dropout(dropout_pct))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(
        loss=multi_weighted_logloss,
        optimizer=Adadelta(lr=0.25, rho=0.95, epsilon=None, decay=0.0),
        metrics=['categorical_accuracy', multi_weighted_logloss]
    )
    return model


def mlp_modeling_cross_validation(
        params: dict,
        fit_params,
        X_features,
        y,
        classes,  # List of class names
        class_weights,  # Dict class -> weight:int
        nr_fold=5,
        random_state=1,
):
    params['input_dim'] = len(X_features.columns)
    
    # Compute sample weights
    w = y.value_counts()
    weights = {i: np.sum(w) / w[i] for i in w.index}
    
    y_ohe = pd.get_dummies(y)
    
    folds = StratifiedKFold(
        n_splits=nr_fold,
        shuffle=True,
        random_state=random_state
    )
    
    clfs = []
    oof_preds = np.zeros((len(X_features), np.unique(y).shape[0]))
    for fold_, (trn_, val_) in enumerate(folds.split(y, y)):
        trn_x, trn_y = X_features.iloc[trn_], y.iloc[trn_]
        trn_y_ohe = y_ohe.iloc[trn_]
        val_x, val_y = X_features.iloc[val_], y.iloc[val_]
        val_y_ohe = y_ohe.iloc[val_]
        norm_sample_weights = trn_y.map(weights).values  #  / trn_y.map(weights).values.sum()
        
        clf = build_classifier(**params)        
        history = clf.fit(
            trn_x.values.astype(np.float32), 
            trn_y_ohe.values.astype(np.bool), 
            validation_data=(
                val_x.values.astype(np.float32),
                val_y_ohe.values.astype(np.bool)
            ),
            shuffle=True, 
            sample_weight=norm_sample_weights.astype(np.float32),
            **fit_params
        )
        clfs.append(clf)
        oof_preds[val_, :] = clf.predict(val_x.values.astype(np.float32))
        # summarize history for accuracy
        plt.plot(history.history['categorical_accuracy'])
        plt.plot(history.history['val_categorical_accuracy'])
        plt.title(f'fold {fold_ + 1} categorical_accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['multi_weighted_logloss'])
        plt.plot(history.history['val_multi_weighted_logloss'])
        plt.title(f'fold {fold_ + 1} multi_weighted_logloss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        # print ensembling loss info
        print('no {}-fold loss: {}'.format(
            fold_ + 1,
            mwl_common(
                val_y, 
                oof_preds[val_, :],
                classes, 
                class_weights
        )))
    # calculate score in the same way as for LGBM
    score = mwl_common(
        y_true=y, 
        y_preds=oof_preds,
        classes=classes, 
        class_weights=class_weights
    )
    print('MULTI WEIGHTED LOG LOSS: {:.5f}'.format(score))
    return clfs, score
