import pandas as pn
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

class Ploter:
    
    def plot_correlation_matrix(self, data_derma):
        """
                Plot the correlation matrix between the data from a heatmeap chart
                
            @base_derma - dermatologic dataset
                
            @return - void, but plot correlation matrix
        """
        colunas = {
    'c1':'Eriteme','c2':'Escala','c3':'Bordas definidas','c4':'Coceira','c5':'fenômeno de Koebner','c6':'Pápulas poligonais',
    'c7':'Envolvimento da mucosa oral','c8':'Envolvimento do joelho e do cotovelo','c9':'Envolvimento do escalpo',
    'c10':'Incotinência de melaninca','c11':'Eosinófilos no infiltrado','c12':'Infriltrado PNL','c13':'Fibrose na derme papilar',
    'c14':'Exocitose','c15':'Acantose','c16':'Hiperceratose','c17':'Paracertose','c18':'Dilatação',
    'c19':'Dilatação em clava dos cones epiteliais','c20':'Dilatação em elava dos cones epiteliais',
    'c21':'Alongamento dos cones epiteliais da epiderme','c22':'Pústulas espongiformes',
    'c23':'Microabscesso de Munro','c24':'Hipergranulose focal','c25':'Ausência da camada granulosa',
    'c26':'Vacuolização e destruição da camada basal','c27':'Espongiose','c28':'Aspecto dente de serra das cristas interpapilares',
     'c30':'Tampões cárneos foliculares',
    'c29':'Tampões cárneos foliculares','c31':'Paraceratose perifolicular','c32':'Infiltrado inflamatório mononuclear',
    'c33':'Infiltrado em banda','c34':'Idade','c35':'Classes',
                }
        
        corr = data_derma.rename(columns=colunas).corr()
        plt.figure(figsize=(20,10))
        ax = sns.heatmap(corr)

    def plot_confusion_matrix(self, df_m_conf):
        """
                Plot model confusion matrix result
            
            @df_m_conf - confusion matrix with dataframe type
        
            @return - void, but plot confusion matrix        
        """
        
        classes={1:'Classes 1',2:'Classes 2',3:'Classes 3',4:'Classes 4',5:'Classes 5',6:'Classes 6'}
        df_m_conf.rename(columns=classes,index=classes,inplace=True)

        group_counts = ["{0:0.0f}".format(value) for value in df_m_conf.values.flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in df_m_conf.values.flatten()/np.sum(df_m_conf.values)]

        labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts,group_percentages)]

        labels = np.asarray(labels).reshape(6,6)
        plt.figure(figsize=(10,6))
        sns.heatmap(df_m_conf.astype(float), annot=labels, fmt='', cmap='Blues');
        plt.title('Confusion Matrix')
        plt.show()