import pandas as pd
import numpy as np
import yaml
import dill
import sys
import lime.lime_tabular
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, matthews_corrcoef, accuracy_score, make_scorer, recall_score

###################################################################################
training_test_ratio = 0.8
input_csv = 'matrix/matrix.tsv'
###################################################################################

# UTILITY FUNCTIONS
def sensibilidad (y_true,y_pred):
    return recall_score(y_true,y_pred)

def especificidad (y_true,y_pred):
    return recall_score(y_true,y_pred,pos_label=0)

def reporte (y_true, y_pred, partition=None):
    acc = accuracy_score (y_true,y_pred)
    sens = sensibilidad (y_true,y_pred)
    esp = especificidad (y_true,y_pred)
    roc_auc = roc_auc_score (y_true,y_pred)
    mcc = matthews_corrcoef (y_true,y_pred)
    print(f'Metrics {partition: <10} ---> Accuracy = %.2f , Sensibility = %.2f, Specificity = %.2f, ROC_AUC_SCORE = %.2f, MCC = %.2f'%(acc,sens,esp,roc_auc,mcc))
    return acc, sens, esp, roc_auc, mcc

def report_dict (y_true, y_pred):
    result = {}
    result ['accuracy'] = accuracy_score (y_true,y_pred)
    result ['sensitivity'] = sensibilidad (y_true,y_pred)
    result ['specificity'] = especificidad (y_true,y_pred)
    result ['AUC'] = roc_auc_score (y_true,y_pred)
    result ['MCC'] = matthews_corrcoef (y_true,y_pred)
    return result

def calc_decil_info (p, y, reduced=False):
    decils = [float(np.percentile(p, q*10)) for q in np.arange(1, 10)]
    p_ranges = [float(0.0)]+decils+[float(1.0)]

    ntot = len(y)        # number of objects
    npos = np.sum(y)     # number of positives

    decil_info =[]

    for i in range (0, 10):
        iinfo = {}

        yi = y[ (p>p_ranges[i]) & (p<=p_ranges[i+1])] 

        itot = len(yi)    # number of selected
        ipos = np.sum(yi) # number of positive selected

        rtot = ntot-itot  # number of non-selected
        rpos = npos-ipos  # number of positive non-selected

        if itot==0 or rtot==0 or rpos==0 or (rtot -rpos) == 0:
            continue

        iinfo['label'] = f'D{(i+1)}'
        iinfo['pmin'] = p_ranges[i]
        iinfo['pmax'] = p_ranges[i+1]

        iinfo['proportion'] = 0.0
        iinfo['RR'] = 0.0
        iinfo['OR'] = 0.0
        iinfo['NNH'] = 0.0

        if itot > 0:
            iinfo['proportion'] = ipos/itot
            if (rtot > 0):
                iinfo['RR'] = (ipos/itot) / (rpos /rtot)    
                iinfo['NNH'] = 1.0 / ( (ipos/itot) - (rpos/rtot) )    

        if (itot-ipos) > 0 and (rtot-rpos) > 0:
            iinfo['OR'] = (ipos/(itot-ipos)) / (rpos /(rtot-rpos))    

        if reduced:
            if p_ranges[i+1] < 0.5:
                continue

        decil_info.append (iinfo)

    return (decil_info)


def main ():

    # READ INPUT
    sys.stdout.write('loading input csv... ')
    sys.stdout.flush()
    df = pd.read_csv(input_csv, sep='\t')
    print (f'{df.shape[0]} visits and {df.shape[1]} variables')
    
    # print (f'MIN: {np.min(df["age"])}')

    # Assume the Y is the last column and define X and Y
    X2=df.iloc[:,:-1]
    Y2=df.iloc[:,-1]

    # Split in train and test set
    X_train,X_test,y_train,y_test=train_test_split(X2,Y2,train_size=training_test_ratio,random_state=46)
    sys.stdout.write(f'splitting in training and test ({training_test_ratio})... ')
    sys.stdout.flush()
    print (f'{X_train.shape[0]} objects in the training series and {X_test.shape[0]} in the test series')

    # GRID SEARCH
    sys.stdout.write(f'searching best parameters... ')
    sys.stdout.flush()
    grid_searches_rf = {'best_params': [], 'best_score': []}
    model_rf = RandomForestClassifier(random_state=46)
    params = {'n_estimators': [200, 250],
            'criterion': ['gini'],
            'min_samples_split': [10, 20],
            'max_depth': [None, 6, 8, 10],
            'class_weight': ['balanced'] }

    gs = GridSearchCV(model_rf, params, verbose=0, scoring=make_scorer(roc_auc_score), refit=False, return_train_score=True, n_jobs=4)
    gs.fit(X_train, y_train)
    grid_searches_rf['best_params'].append(gs.best_params_)
    grid_searches_rf['best_score'].append(gs.best_score_)

    print (f'best results obtained using: {gs.best_params_} score: {gs.best_score_}')

    # REBUILD BEST MODEL
    model_rf_def=RandomForestClassifier(**gs.best_params_, random_state=46).fit(X_train,y_train)

    # calculate predictions and probabilities
    pred_proba_train = model_rf_def.predict_proba(X_train)[:,1]
    pred_train       = model_rf_def.predict(X_train)
    pred_proba_test  = model_rf_def.predict_proba(X_test)[:,1]
    pred_test        = model_rf_def.predict(X_test)

    # show standard metrics
    metrics_train_rf = reporte(y_train,pred_train,partition='training')
    metrics_test_rf  = reporte(y_test,pred_test,partition='test')

    met_total=pd.DataFrame(np.c_[metrics_train_rf,metrics_test_rf],                   
                            columns=['Train_RF','Test_RF'],index=['Aquracy','Sensibility','Specificity','ROC_AUC','MCC'])
    round(met_total,2)

    # EXPLAINER
    sys.stdout.write(f'generating LIME explainer... ')
    sys.stdout.flush()

    explainer = lime.lime_tabular.LimeTabularExplainer(X_train.to_numpy(), 
                                                    feature_names=model_rf_def.feature_names_in_, 
                                                    class_names=['negative', 'positive'],
                                                    discretize_continuous=True)
    print ('done')

    # Compute decils and percentils
    sys.stdout.write(f'calculate decil info... ')
    sys.stdout.flush()
    decil_info = calc_decil_info(pred_proba_train, y_train, True)

    prop=[decil_info[i]['proportion'] for i in range(0,len(decil_info))]
    print ('done')

    for j, i in enumerate(decil_info):
        print (f'D{j+1} min:{i["pmin"]:.3f} max:{i["pmax"]:.3f} proportion:{i["proportion"]:.3f} OR:{i["OR"]:.2f} RR:{i["RR"]:.2f} NNH:{i["NNH"]:.1f}')
        prop.append(i["proportion"])

    percentils = [float(np.percentile(pred_proba_train, q)) for q in np.arange(1, 100)]

    # VAR IMPORTANCE
    sys.stdout.write(f'exporting var importance... ')
    sys.stdout.flush()
    names = list(X_train)
    important_features_dict = {}
    for idx, val in enumerate(model_rf_def.feature_importances_):
        important_features_dict[idx] = val

    important_features_list = sorted(important_features_dict,
                                    key=important_features_dict.get,
                                    reverse=True)

    var_importance = [names[i] for i in important_features_list]
    print ('done')

    # WRITE DICTIONARY FOR EXPORTING ESTIMATOR AND MODEL INFO
    predictors_file = './matrix/predictors.yaml'
    predictors_dict=None
    with open(predictors_file, 'r') as handle:
        predictors_dict = yaml.safe_load(handle)

    decil_info = calc_decil_info(pred_proba_train, y_train, True)
    percentils = [float(np.percentile(pred_proba_train, q)) for q in np.arange(1, 100)]
    metrics_train = report_dict(y_train,pred_train)
    metrics_test  = report_dict(y_test,pred_test)
    metrics_train['label'] = f'{len(y_train)} objects, {(training_test_ratio)*100:.0f}% of the original dataset, selected randomly'
    metrics_test['label'] = f'{len(y_test)} objects, {(1.0-training_test_ratio)*100:.0f}% of the original dataset, selected randomly'

    model_pkl = './models/model-rf.dill'
    sys.stdout.write(f'exporting estimator as {model_pkl}... ')
    sys.stdout.flush()
    output_dict = {}
    output_dict['model'] = model_rf_def
    output_dict['explainer'] = explainer
    output_dict['description'] = 'RF model'
    output_dict['decil_info'] = decil_info
    output_dict['percentils'] = percentils
    output_dict['metrics_fitting'] = metrics_train
    output_dict['metrics_prediction'] = metrics_test
    output_dict['predictors_dict'] = predictors_dict
    output_dict['var_importance'] = var_importance

    with open(model_pkl, 'wb') as handle:
        dill.dump(output_dict, handle)

    print ('normal termination')

if __name__ == '__main__':
    main()
