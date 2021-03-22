#Useful import
import numpy as np
import pandas as pd

class Reporter:
    
    def precision(self, TP, FP):
        """
                Get precision at confusion matrix
                
            @TP - True Positive
            @FP - False Positive
            
            @return - precision metric
        """
        
        return TP/(FP+TP)
    
    def recall(self, TP, FN):
        """
                Get recal at confusion matrix
                
            @TP - True Positve
            @FN - False Negative
            
            @return - recall metric
        """
        
        return TP/(TP+FN)
    
    def F1(self, precision, recall):
        """
                Get f1-score at confusion matrix
            
            @precision - precision metric
            @recall - recall metric
            
            @return - f1-score metric
        """
        
        return (2*precision*recall)/(precision+recall)
    
    def accuracy(self, TP, TN, FN, FP):
        """
                Get accuracy score at confusion matrix
            
            @TP - True Positive 
            @TN - True Negative
            @FN - False Negative
            @FP - False Negative
        
            @reutrn - accuracy score
        """
        
        return (TP+TN)/(TP+TN+FN+FP)
    
    
    def print_results(self, df_m_conf):
        """
            Calculate classification metrics an plot results
            
            @df_m_conf - confusion matrix with dataframe type
            
            @return - mean result after receive n confusion matrixes        
        """
        
        
        metrics = []
        for classes in range(df_m_conf.shape[0]):
            TP = df_m_conf.iloc[classes,classes]
            values = [i for i in range(6) if i!=classes]
            FP = np.sum(df_m_conf.iloc[classes,values])
            TN = np.sum([df_m_conf.iloc[i,i] for i in range(6) if i!=classes])
            FN = np.sum(df_m_conf.iloc[values,classes])

            results = {
                'precision':self.precision(TP,FP),
                'recall'   :self.recall(TP,FN),
                'accuracy' :self.accuracy(TP,TN,FN,FP),
                'f1-score' :self.F1(self.precision(TP,FP),self.recall(TP,FN))
            }

            metrics.append(results)

        return pd.DataFrame(metrics).mean()