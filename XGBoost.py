import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import confusion_matrix, classification_report
import random
import numpy as np
import matplotlib.pyplot as plt
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.utils.validation import column_or_1d
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.utils import class_weight
from sklearn import preprocessing


def classification_report_with_accuracy_score(y_true, y_pred):

    print(classification_report(y_true, y_pred)) # print classification report
    return accuracy_score(y_true, y_pred) # return accuracy score

class MLTask:
    def __init__(self):

        self.rs = 9987
        random.seed(self.rs)
        np.random.seed(self.rs)
        self.data=None
        self.data_x,self.data_y, self.x_train, self.x_test, self.y_train, self.y_test= None,None,None,None,None,None


    def prepareData(self):
        self.data = pd.read_csv('MultiClassification_Glass.csv', float_precision='high')
        self.data_x = self.data.drop(columns=['Type'])
        self.data_y = self.data['Type']
        type(self.data)
        print(self.data.head())

        print(self.data.tail())

        print(self.data.shape)

        print(self.data.describe())

    def saveData(self):
        print(self.data.describe())
        description = self.data.describe()
        description.to_excel("description.xlsx")
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.data_x, self.data_y,
                                                            test_size=0.2,
                                                            random_state=self.rs,
                                                            shuffle=True,
                                                            stratify=self.data_y)

        self.x_train.to_csv("x_train.csv", index=False)
        self.x_test.to_csv("x_test.csv", index=False)
        print(self.y_train)
        self.y_train.to_csv("y_train.csv", index=False)
        self.y_test.to_csv("y_test.csv",index = False)

    def corrolation(self):
        selector = VarianceThreshold()
        print("variance information************************************")
        print(selector.fit_transform(self.data).shape)

        corr_df_features = self.data_x.corr()
        print("The total corrolation matrix with threshold***********************")
        (np.fill_diagonal(corr_df_features.values, 0))
        print(corr_df_features[abs(corr_df_features) > 0.4])

        corr_df = self.data.corr()
        np.fill_diagonal(corr_df.values, 0)  # Setting to diagonal to zero
        print("Corrolation ***********************")
        print(abs(corr_df['Type']).sort_values(ascending=False))


    def XGBoostTrain(self,xPath,yPath):
        #******************Set all the parameters and pre-process data**************************
        X = pd.read_csv(xPath, float_precision='high')
        y=pd.read_csv(yPath,float_precision='high',header=None)
        X_test = pd.read_csv("data/x_test.csv", float_precision='high')
        y_test=pd.read_csv("data/y_test.csv",float_precision='high',header=None)
        y = column_or_1d(y, warn=False)
        pickleXGBoost = 'XGBoostModel.sav'

        le = preprocessing.LabelEncoder()
        le.fit(y)
        y_test = le.transform(y_test)
        y = le.transform(y)

        '''
        params_grid = {
            'max_depth': [1,3,4,6],
            'colsample_bytree': [0.6, 0.8],
            'n_estimators': [20,50,100],
            'learning_rate': np.linspace(0.2, 1, 10)
        }
        '''
        params_grid = {
            'max_depth': [6],
            'n_estimators': [17],
            'learning_rate': [0.6919191919191919]#np.linspace(0.5, 1, 100)
        }

        params_fixed = {
            'objective': 'multi:softprob',
            'silent': 1,
        }

        from sklearn.utils import class_weight
        class_weights = list(class_weight.compute_class_weight('balanced',
                                                               np.unique(y),y))
        #custom weights worked best
        class_weights=[1,1,5,1,1,1]
        w_array = np.ones(y.shape[0], dtype='float')
        for i, val in enumerate(y):
            w_array[i] = class_weights[val]

        #***********************Learning starts here************************************
        bst_grid = GridSearchCV(
            estimator=XGBClassifier(**params_fixed,seed= self.rs,random_state=self.rs),
            param_grid=params_grid,
            cv=5,
            scoring='balanced_accuracy',
            refit = True
        )
        bst_grid.fit(X,y,sample_weight=w_array)


        #**************************Print the results****************************
        print("Best accuracy obtained: {0}".format(bst_grid.best_score_))
        print("Parameters:")
        for key, value in bst_grid.best_params_.items():
            print("\t{}: {}".format(key, value))
        print("Best estimator: ",bst_grid.best_estimator_)
        results = pd.DataFrame(bst_grid.cv_results_)
        print(results)
        results.to_excel("results.xlsx")
        gs_result=bst_grid.predict(X)
        print("=== Confusion Matrix ===")
        confusion_matrix_test = confusion_matrix(y,gs_result )
        print(confusion_matrix_test)
        print('\n')
        print("=== Classification Report  ===")
        classification_report_test = classification_report(y, gs_result)
        print(classification_report_test)

        #*********************Test Results**********************************
        gs_result=bst_grid.predict(X_test)
        print("=== Confusion Matrix TEST===")
        confusion_matrix_test = confusion_matrix(y_test,gs_result )
        print(confusion_matrix_test)
        print('\n')
        print("=== Classification Report TEST ===")
        classification_report_test = classification_report(y_test, gs_result)
        print(classification_report_test)

        #*************************Save the classification report************************
        CR = classification_report(y_test, gs_result, output_dict=True)
        df = pd.DataFrame(CR).transpose()
        df.to_excel("XGBoostCR.xlsx")

        #************************Save confusion matrix***************************
        import seaborn as sn
        df_cm = pd.DataFrame(confusion_matrix_test, index=range(6), columns=range(6))
        sn.set(font_scale=1.4)  # for label size
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
        # fix for mpl bug that cuts off top/bottom of seaborn viz
        b, t = plt.ylim()  # discover the values for bottom and top
        b += 0.5  # Add 0.5 to the bottom
        t -= 0.5  # Subtract 0.5 from the top
        plt.ylim(b, t)  # update the ylim(bottom, top) values
        plt.savefig('XGBoostCM.png', bbox_inches='tight')

        #*******************************Plot the importance of features ******************************
        from xgboost import plot_importance
        plot_importance(bst_grid.best_estimator_, importance_type='gain', xlabel='Gain')
        plt.savefig('XGBoostImportantF.png', bbox_inches='tight')

        #****************************save model in pickle file****************************************
        pickle.dump(bst_grid.best_estimator_, open(pickleXGBoost, 'wb'))

        
        '''
        print(y_test)
        print(gs_result)
        print(bst_grid.predict_proba(X_test))
        df = pd.DataFrame(bst_grid.predict_proba(X_test))
        df["Predicted class"]=gs_result
        df["True value"]= y_test
        df.to_excel("XGBoostPredict.xlsx")
        '''

    def XGBoostPredict(self, X,model):
        #Input X is a path for a csv with column values like the original data, model is a string with the location of the model
        #returns the class 0-5 and the probability vector for debug
        data = pd.read_csv(X, float_precision='high')

        pickle_in = open(model, "rb")
        XGBoost = pickle.load(pickle_in)
        predictedClass = XGBoost.predict(data).tolist()
        probabilityVector = XGBoost.predict_proba(data).tolist()
        return predictedClass, probabilityVector

if __name__ == "__main__":
    #StratifiedKFold(y, n_folds=5, shuffle=True,random_state=9987)

    ML = MLTask()

    ML.prepareData()
    ML.saveData()

    ML.corrolation()

    ML.XGBoostTrain("data/x_train.csv","data/y_train.csv")
    X="data/x_test.csv"
    X="testInput.csv"

   # print(ML.XGBoostPredict(X,"XGBoostModel.sav"))
