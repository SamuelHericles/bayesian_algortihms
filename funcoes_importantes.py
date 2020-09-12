import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
import seaborn as sns

class carrega_base:
    caminho = 'https://raw.githubusercontent.com/SamuelHericles/Algoritmos_de_classificao_baysianos/master/01.Dados/dema_dados.csv'
    derma_dados = pd.read_csv(caminho)
    derma_dados.sort_values('c35',inplace=True)
    derma_dados.reset_index(drop=True,inplace=True)
    
class funcoes_main:
    def __init__(self):
        self.base = carrega_base.derma_dados
        
    def vetor_medio(self,base):
        vetor_medio = []
        for i,j in zip(base.columns[:-1],[i for i in range(base.shape[1])]):
            vetor_medio.append(base[str(i)].mean())
        df = pd.DataFrame(data=vetor_medio,index=base.columns[:-1])
        return df.T


    def vetor_variancias(self,base):
        vetor_medio = []
        for i,j in zip(base.columns[:-1],[i for i in range(base.shape[1])]):
            vetor_medio.append(base[str(i)].var())
        df = pd.DataFrame(data=vetor_medio,index=base.columns[:-1])
        return df.T


    def matriz_covariancia(self,base):
        df = pd.DataFrame(index=[base.columns[i] for i in range(base.shape[1])],columns=base.columns[:-1])
        for i,count_i in zip(base.columns[:-1],[i for i in range(base.shape[1])]):
            for j,count_j in zip(base.columns[:-1],[i for i in range(base.shape[1])]):
                df.iloc[count_i,count_j] = (sum(base[str(i)]*base[str(j)])- 
                                            (sum(base[str(i)])*sum(base[str(j)]))/base.shape[0])/base.shape[0] 
        return df.iloc[:-1]

    def norm_z_score(self,base_trenio, base_teste):
        media         = base_trenio[base_trenio.columns[:-1]].mean()
        desvio_padrao = base_trenio[base_trenio.columns[:-1]].std()
        base_trenio[base_trenio.columns[:-1]] = base_trenio[base_trenio.columns[:-1]]-media/desvio_padrao
        base_teste[base_teste.columns[:-1]]   = base_teste[base_teste.columns[:-1]]-media/desvio_padrao
        return base_trenio,base_teste

    def kfold_shuffle(self,k=5):
        X = pd.DataFrame({})
        y = pd.DataFrame({})

        index_teste_classe = sorted(np.random.choice(self.base.index.values,round((self.base.shape[0]/k))))
        X_classe = self.base.iloc[[i for i in self.base.index if i not in index_teste_classe],:]
        y_classe = self.base.iloc[index_teste_classe,:]

        X = X.append(X_classe,ignore_index=True)
        y = y.append(y_classe,ignore_index=True)

        return self.norm_z_score(X,y)

    def prob_a_priori(self,base):
        probs_p = []
        for i in base['c35'].unique():
            probs_p.append(base.query('c35=='+str(i))['c35'].shape[0]/base.shape[0])
        return probs_p

    def teste_elem(self,X,X_medio,mcov_inv,determinante,probabilidade):
        X = X.values.reshape((1,-1))
        X_medio = X_medio.values.reshape((-1,1))
        det_log  = np.log(abs(determinante))
        m_linha  = (X-pd.DataFrame(X_medio).T).values
        m_coluna = (X-pd.DataFrame(X_medio).T).T.values
        log_prob = np.log(probabilidade)
        resultado = det_log + (m_linha @ mcov_inv) @ (m_coluna) - 2*log_prob
        return resultado[0][0]

    def get_acc(self,y_pred,y_true):
        return round(sum(y_pred==y_true)/y_true.shape[0],4)

    def teste_elem_lda(self,X,X_medio,mcov_inv):
        m_linha   = (X-X_medio).values
        m_coluna  = (X-X_medio).T.values
        resultado = (m_linha @ mcov_inv) @ m_coluna
        return resultado[0][0]

    def limpar_covariancia(self,matriz_covariancia):
        for i in range(matriz_covariancia.shape[0]):
            for j in range(matriz_covariancia.shape[0]):
                if i!=j:
                    matriz_covariancia.iloc[i,j]=0
        return matriz_covariancia
    
    def correcao_lambda(self,matriz_de_covariancia):
         return matriz_de_covariancia + np.identity(matriz_de_covariancia.shape[0], dtype=float)*0.01
    
    def matriz_inversa(self,matriz_covariancia):
         return pd.DataFrame(np.linalg.inv(np.matrix(matriz_covariancia.values, dtype='float')))
        
    def determinante_matriz_covariancia(self,matriz_covariancia):
        return np.linalg.det(np.matrix(matriz_covariancia.values, dtype='float'))        
    
    def vetor_medio_p_classe(self,base):
        vt_medios = pd.DataFrame(index=base.columns[:-1])
        for i in sorted(base[base.columns[-1]].unique()):
            vt_medios['classe'+str(i)] = self.vetor_medio(base.query('c35=='+str(i))).T
        return vt_medios.T
    
    def matriz_correlacao(self,base,vt_medio):
        # Cria df para popular
        df = pd.DataFrame(index=[base.columns[i] for i in range(base.shape[1]-1)],columns=base.columns[:-1])
        
        # Calcula a correlação de um atributo com outro(pode ser ele mesmo) e popula o df
        for i,count_i in zip(base.columns[:-1],[i for i in range(base.shape[1])]):
            for j,count_j  in zip(base.columns[:-1],[i for i in range(base.shape[1])]):

                num = np.sum(np.dot((base[i]-vt_medio[i].values),
                                    (base[j]-vt_medio[j].values)))

                dem = np.sqrt(np.dot(np.sum(pow(base[i]-vt_medio[i].values,2)),
                                     np.sum(pow(base[j]-vt_medio[j].values,2))))

                df.iloc[count_i,count_j] = num/dem
        return df
    
    def calcula_projecao(self,base,u,u_classe):
        return pd.DataFrame(base.shape[0]*((u_classe - u).values*(u_classe - u).T.values))
    
    def treino_teste_70_30(self,base):
        X = pd.DataFrame({})
        y = pd.DataFrame({})

        index_teste_classe = sorted(np.random.choice(base.index.values,int(base.shape[0]*0.3)))
        X_classe = base.iloc[[i for i in base.index if i not in index_teste_classe],:]
        y_classe = base.iloc[index_teste_classe,:]

        X = X.append(X_classe,ignore_index=True)
        y = y.append(y_classe,ignore_index=True)

        return X,y
    
    def transformar_dados(self,W,base):
        colunas = {0:'c1',1:'c2',2:'c3',3:'c4',4:'c5'}
        
        Xtr_t = pd.DataFrame(W.values @  base[base.columns[:-1]].T.values).T
        Xtr_t['c35'] = base['c35']
        Xtr_t.rename(columns=colunas,inplace=True)
        
        return Xtr_t
    
    def plot_matriz_de_correlacao(self,base_derma):
        colunas = {
    'c1':'Eriteme','c2':'Escala','c3':'Bordas definidas','c4':'Coceira','c5':'fenômeno de Koebner','c6':'Pápulas poligonais',
    'c7':'Envolvimento da mucosa oral','c8':'Envolvimento do joelho e do cotovelo','c9':'Envolvimento do escalpo',
    'c10':'Incotinência de melaninca','c11':'Eosinófilos no infiltrado','c12':'Infriltrado PNL','c13':'Fibrose na derme papilar',
    'c14':'Exocitose','c15':'Acantose','c16':'Hiperceratose','c17':'Paracertose','c18':'Dilatação',
    'c19':'Dilatação em clava dos cones epiteliais','c21':'Alongamento dos cones epiteliais da epiderme','c22':'Pústulas espongiformes',
    'c23':'Microabscesso de Munro','c24':'Hipergranulose focal','c25':'Ausência da camada granulosa',
    'c26':'Vacuolização e destruição da camada basal','c27':'Espongiose','c28':'Aspecto dente de serra das cristas interpapilares',
    'c29':'Tampões cárneos foliculares','c31':'Paraceratose perifolicular','c32':'Infiltrado inflamatório mononuclear',
    'c33':'Infiltrado em banda','c34':'Idade','c35':'Classes',
                }
        corr = base_derma.rename(columns=colunas).corr()
        plt.figure(figsize=(20,10))
        ax = sns.heatmap(corr)
    