# Useful Imports
import pandas as pd
import numpy as np
import seaborn as sns

from .LoaderFile import Loader
   
class Cleaner:
    
    def __init__(self):
        """
                Function to load the funcoes_main class already loads the database
        """ 
        
        self.load_base = Loader()
        self.base = self.load_base.data_derma

    def average_vector(self,base):
        """        
                Calculates the average of each attribute and stores it in a vector, in this case, the vector is a single-row dataframe with 34 columns.
            @base - demartologic dataset 
            
            @return - average dataset
        """
        
        vetor_medio = []
        for i,j in zip(base.columns[:-1],[i for i in range(base.shape[1])]):
            vetor_medio.append(base[str(i)].mean())
        df = pd.DataFrame(data=vetor_medio,index=base.columns[:-1])
        return df.T


    def variance_vector(self,base):
        """
                Calculates the variance of each atritube and stores it in a vector, in this case the vector is a single-row dataframe with 34 columns.
            @base - demartologic dataset 
            
            @return - dataset variance vetor
        """

        vetor_medio = []
        for i,j in zip(base.columns[:-1],[i for i in range(base.shape[1])]):
            vetor_medio.append(base[str(i)].var())
        df = pd.DataFrame(data=vetor_medio,index=base.columns[:-1])
        return df.T

    def covariance_matrix(self, base):
        """
                Calculates the covariance of one attribute with another (or itself being the variance) and stores it in a 34x34 matrix dataframe.
            
            @base - demartologic dataset 
            
            @return - dataset covariance matrix 
        """
                        
        #    Creates a dataframe the size of the number of attributes in the base, the last attribute being the sample labels, therefore
        #is taken from the calculation, so base.columns [: - 1]
        df = pd.DataFrame(index=[base.columns[i] for i in range(base.shape[1])],columns=base.columns[:-1])
        
        #    There are two for but could be 4, but there is the zip function that makes the for go through two or more vectors, with that
        #I can go through the columns of the database and at the same time I do the covariance of the attributes I populate the matrix of
        #covariance.
        for i,count_i in zip(base.columns[:-1],[i for i in range(base.shape[1])]):
            for j,count_j in zip(base.columns[:-1],[i for i in range(base.shape[1])]):
                
                # Calculates covariance
                df.iloc[count_i,count_j] = (sum(base[str(i)]*base[str(j)])- 
                                            (sum(base[str(i)])*sum(base[str(j)]))/base.shape[0])/base.shape[0]
                
        # Returns one column less because it is the class column and it is like NaN
        return df.iloc[:-1]

    

    def norm_z_score(self,base_train, base_test):
        """
                    Z-score normalization, I calculate the average and standard deviation of an attribute and normalize its data.
             With the detail that the normalization of the test base is with the average and standard deviation of the sleigh data
            
            @base_train - train dataset
            @base_test  - test dataset
            
            @return - base_train and base_test normalized
            
        """
        
        # Mean
        mean         = base_train[base_train.columns[:-1]].mean()
        
        # Standard desviantion
        std = base_train[base_train.columns[:-1]].std()
        
        # Normalize the training base
        base_train[base_train.columns[:-1]] = base_train[base_train.columns[:-1]]-mean/std
        base_test[base_test.columns[:-1]]   = base_test[base_test.columns[:-1]]-mean/std
        
        # Returns both normalized bases
        return base_train, base_test

    

    def kfold_shuffle(self, k=5):        
        """
                    K-fold for training, here it was made to take 20% of the data at random and create the test base and
             what’s left of the training base. I didn't stratify because there are many samples in each class so there is a very high chance
             have samples of all classes in each base.
             
             @k - fold in Kfold method
             
             @return - dataset 'kfolded'        
        """
        
        
        # Create a training and test dataframe
        X = pd.DataFrame({})
        y = pd.DataFrame({})

        #     Take 20% of the indexes from the database at random and remove them to put in the test base
        #what's left put in the base sleigh
        index_teste_classe = sorted(np.random.choice(self.base.index.values,round((self.base.shape[0]/k))))
        X_classe = self.base.iloc[[i for i in self.base.index if i not in index_teste_classe],:]
        y_classe = self.base.iloc[index_teste_classe,:]

        # When done the stored in X and y, the indexes are the same as the old base, so I reset them.        
        X = X.append(X_classe,ignore_index=True)
        y = y.append(y_classe,ignore_index=True)

        # With that, as was requested I already normalize the data at the exit
        return self.norm_z_score(X,y)

    
    def prob_priori(self,base):
        """
                Calculate the a priori probability of each class in a database.
                
            @base - demartologic dataset 
            
            @return  - probabilities between classes
        """
        probs_p = []
        for i in base['c35'].unique():
            probs_p.append(base.query('c35=='+str(i))['c35'].shape[0]/base.shape[0])
        return probs_p

    
    def test_elem(self, y, X_mean, mcov_inv, determinant, probability):
        """        
                Here an element of the test base is tested, this function is for the QDA and Naive Bayes algorithms.            
            
            @y - samples labels
            @X_mean -  average train dataset
            @mcov_inv - inverse covariance matrix
            @determinat - determinat matrix
            @probability - labels probability
            
            @return - output label
        """              
        
        # Correct the base dimension, because some in python come as (34,) or (5,) so I correct for (34.1) or (5.1)
        y = y.values.reshape((1,-1))
        X_mean = X_mean.values.reshape((-1,1))
        
        # Calculate the natural logarithm of the determinant of the inverse covariance matrix
        det_log  = np.log(abs(determinant))
        
        # Create the line matrix or line vector of the test data with the average vector of each class of the training base
        m_row  = (y-pd.DataFrame(X_mean).T).values
       
        # I create the column matrix or column vector of the test data with the average vector of each class of the training base        
        m_column = (y-pd.DataFrame(X_mean).T).T.values
        
        # Calculate the logarithm of the probability a priori
        log_prob = np.log(probability)
        
        # I perform the test to project the test data, here the return is a scale that the smaller the exact label is in the
        # which the algorithm returns
        results = det_log + (m_row @ mcov_inv) @ (m_column) - 2*log_prob
        return results[0][0]


    def test_elem_lda(self, y, X_mean, mcov_inv):
        """
                Here the calculation of the lda algorithm is done, in which it tests the test data with the projection of the training data.            
            
            @y - samples labels
            @X_mean -  average train dataset
            @mcov_inv - inverse covariance matrix
            
            @return - output label
        """
                
        y = y.values.reshape((1,-1))
        X_mean = X_mean.values.reshape((-1,1))
        
        # Create the line matrix or line vector of the test data with the average vector of each class of the training base
        m_row   = (y-X_mean.T)
        
        # Create the column matrix or column vector of the test data with the average vector of each class of the training base
        m_column  = (y-X_mean.T).T
        
        # Calculate the projection of the data to return a scale in which the smallest one refers to the class that the algorithm returns
        resultado = (m_row @ mcov_inv) @ m_column
        return resultado[0][0]

    def get_acc(self, y_pred, y_true):
        """
                Calculates the accuracy of the results
            @y_pred - labels predicted
            @y_true - labels true
            
            @retrun - accuraccy score
        """
        
        return round(sum(y_pred==y_true)/y_true.shape[0],4)    
    
    def clean_covariance(self, covariance_matrix):
        """
                Clears the covariance of the covariance matrix to apply the naive bayes algorithm as it considers that there is no covariance between
            the data because they are independent events between them.    
            
            @covariance_matrix - dataset covariance matrix
            
            @return covariance matrix cleaned
        """
                
        for i in range(covariance_matrix.shape[0]):
            for j in range(covariance_matrix.shape[0]):
                
                # Zero data that is not diagonal, that is, there is only variance of attributes, not covariance                
                if i!=j:
                    covariance_matrix.iloc[i,j] = 0
                    
        return covariance_matrix
    
    def correct_lambda(self, covariance_matrix):
        """
                I do the lambda correction, I need it because I add a very small value
            the covariance matrix so that there is a determinant and thus an inverse matrix.    
            
            @covariance_matrix - dataset covariance matrix
            
            @return covariance matrix corrected                        
        """

        return covariance_matrix + np.identity(covariance_matrix.shape[0], dtype=float)*0.01
    
    def inverse_matrix(self, covariance_matrix):
        """
                Calculates the inverse of the covariance matrix    
            
            @covariance_matrix - dataset covariance matrix
            
            @return covariance matrix inverse                   
        """
        
        return pd.DataFrame(np.linalg.inv(np.matrix(covariance_matrix.values, dtype='float')))
                
    def covariance_matrix_determinant(self, covariance_matrix):
        """
                Calculates the determinant of the covariance matrix or the inverse of it.
            
            @covariance_matrix - dataset covariance matrix
            
            @return determinat covariance matrix                                               
        """            

        return np.linalg.det(np.matrix(covariance_matrix.values, dtype='float'))        
    
    def mean_vector_medio_per_feature(self, base):
        """
                Calculates the average vector of all classes    

            @base - demartologic dataset 
            
            @return - a vector with mean columns features
        """
        
        vt_medios = pd.DataFrame(index=base.columns[:-1])
        for i in sorted(base[base.columns[-1]].unique()):
            vt_medios['classe'+str(i)] = self.average_vector(base.query('c35=='+str(i))).T
        return vt_medios.T
    
    def corrleation_matrix(self, base, average_vector):

        """
                Calculate the correlation matrix of the attributes
            
            @base - demartologic dataset 
            
            @average_vector - a vector with mean columns features
            
            @return - correlation matrix
        """
        
        # Create df for popular        
        df = pd.DataFrame(index=[base.columns[i] for i in range(base.shape[1]-1)],columns=base.columns[:-1])
        
        # Calculates the correlation of one attribute with another (can be itself) and populates the df        
        for i,count_i in zip(base.columns[:-1],[i for i in range(base.shape[1])]):
            for j,count_j  in zip(base.columns[:-1],[i for i in range(base.shape[1])]):

                # Numerator that is the covariance between attributes                
                num = np.sum(np.dot((base[i]-average_vector[i].values),
                                    (base[j]-average_vector[j].values)))

                # Denominator I calculate the variance of one attribute times the variance of another                
                dem = np.sqrt(np.dot(np.sum(pow(base[i]-average_vector[i].values,2)),
                                     np.sum(pow(base[j]-average_vector[j].values,2))))

                df.iloc[count_i,count_j] = num/dem
        return df
    
    def calculate_projection(self, base, u, u_feature):
        """
                Project the average vector minus the average of the average vectors
        
            @base - demartologic dataset 
            
            @u - mean especific feature
            
            @u_feature - mean features
            
            @return - dataset projection
        """
        
        return pd.DataFrame(base.shape[0]*((u_feature - u).values*(u_feature - u).T.values))
   

    def split_train_test_70_30(self, base):
        """
                Split the data between 70% for training and 30% for testing for the CDA approach        

            @base - demartologic dataset 

            @return - train dataset and train dataset labels
        """
        
        X = pd.DataFrame({})
        y = pd.DataFrame({})

        index_teste_classe = sorted(np.random.choice(base.index.values,int(base.shape[0]*0.3)))
        X_classe = base.iloc[[i for i in base.index if i not in index_teste_classe],:]
        y_classe = base.iloc[index_teste_classe,:]

        X = X.append(X_classe,ignore_index=True)
        y = y.append(y_classe,ignore_index=True)

        return X,y
    
    def data_transform(self, W, base):
        """
                Transform the data from the CDA W matrix
            
            @W - features matrix projected
            @base - demartologic dataset 
            
            @return - train dataset transformed
        """
        
        colunas = {0:'c1',1:'c2',2:'c3',3:'c4',4:'c5'}
        
        Xtr_t = pd.DataFrame(W.values @  base[base.columns[:-1]].T.values).T
        Xtr_t['c35'] = base['c35']
        Xtr_t.rename(columns=colunas,inplace=True)
        
        return Xtr_t
    
    
    def confusion_matrix(self, y_true, y_pred):
        """
                Calculates the confusion matrix
            @y_pred - labels predicted
            @y_true - labels true        
            
            @return - confusion matrix
        """
        df_results = pd.DataFrame(data=[y_true.values,y_pred]).T
        df_results.rename(columns={0:'Expected',1:'Provided'},inplace=True)
        df_m_conf = pd.DataFrame(index=[i for i in y_true.unique()],columns=y_true.unique())
        for i in range(df_m_conf.shape[0]):
            for j in range(df_m_conf.shape[0]):
                df_m_conf.iloc[i,j] = df_results.query('Expected=='+str(i+1)+' and Provided=='+str(j+1)).sum()[0]
        return df_m_conf




    

    