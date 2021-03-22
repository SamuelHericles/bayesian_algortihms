import pandas as pd

class Loader:
        
        def __init__(self):
            path = 'https://raw.githubusercontent.com/SamuelHericles/Algoritmos_de_classificao_baysianos/master/data/dema_dados.csv'
            self.data_derma = pd.read_csv(path)
            self.data_derma.sort_values('c35',inplace=True)
            self.data_derma.reset_index(drop=True,inplace=True)

    