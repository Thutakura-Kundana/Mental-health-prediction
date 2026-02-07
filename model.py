from sklearn.model_selection import train_test_split
from sklearn.cluster import FeatureAgglomeration
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from scipy import signal 
from sklearn import svm
import numpy as np
import pandas as pd
import plotly.express as px
from IPython.display import display


class dataset_handling():
    def creating_dataframe(self):
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_colwidth', None)
        dataframe_hrv = pd.read_csv("dataframe_hrv.csv")
        dataframe_hrv = dataframe_hrv.reset_index(drop=True)
        return dataframe_hrv

    def fix_stress_labels(self,df='',label_column='stress'):
        df['stress'] = np.where(df['stress']>=0.5, 1, 0)
        return df

    def missing_values(self,df):
        df = df.reset_index()
        df = df.replace([np.inf, -np.inf], np.nan)
        df[~np.isfinite(df)] = np.nan
        df['HR'].fillna((df['HR'].mean()), inplace=True)
        df['HR'] = signal.medfilt(df['HR'],13) 
        df=df.fillna(df.mean())
        return df

    def train_and_test(self,dataframe_hrv):
        selected_x_columns = ['HR','interval in seconds','AVNN', 'RMSSD', 'pNN50', 'TP', 'ULF', 'VLF', 'LF', 'HF','LF_HF']
        X = dataframe_hrv[selected_x_columns]
        y = dataframe_hrv['stress']
        exported_pipeline = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
        FeatureAgglomeration(affinity="l1", linkage="complete"),
        KNeighborsClassifier(n_neighbors=18, p=1, weights="distance")
    )
        X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.80, test_size=0.20)
        exported_pipeline.fit(X_train, y_train)
        return selected_x_columns,exported_pipeline
    
    def plotFitBitReading(self,dfnewHRV='', predictor = "none",selected_x_columns=''):
        dfnewHRV = self.missing_values(dfnewHRV)
        dfnewPol = dfnewHRV[selected_x_columns].fillna(0)

        pred = predictor.predict_proba(dfnewPol)
        
        dfpred = pd.DataFrame(pred)

        dfpred.columns = [["FALSE","TRUE"]]
        dfpred['stress'] = np.where(dfpred["TRUE"] > 0.5, 1, np.nan)

        
        dfnewHRV["stress"] = dfpred["stress"]
        dfnewHRV.loc[dfnewHRV["steps"] > 0, 'stress'] = np.nan
        #mark is to mark the RR peaks as stress
        dfnewHRV.loc[dfnewHRV["stress"] == 1, 'stress'] = dfnewHRV['interval in seconds'] 
        dfnewHRV.loc[dfnewHRV["steps"] > 0, 'moving'] = dfnewHRV['interval in seconds'] 
        dfnewHRV["minutes"] = (dfnewHRV['newtime']/60)/1000
        
        fig = px.line(dfnewHRV, x="minutes", y=['interval in seconds',"stress", "moving"])

        fig.update_layout(
            autosize=True,
            #width=500,
            #height=450,
            legend_title_text='Variable',
            legend=dict(
                yanchor="top",
                y = -0.20,
                xanchor="right",
                x = 1,
                bordercolor="Black",
                borderwidth=2,
            )
        )
        return fig