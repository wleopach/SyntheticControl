from config import PATH
import pandas as pd
import pickle
from datetime import datetime
pedidos = pd.read_parquet(PATH + 'DF_DET_PED_N.parquet')
Control_Dia = datetime(2023, 5, 24)
pedidos['FECHA_DIA_T'] = pd.to_datetime(pedidos['FECHA_DIA_T'])
# Sort the DataFrame by the 'FECHA_DIA_T' column
pedidos = pedidos.sort_values(by='FECHA_DIA_T')
pedidos = pedidos.query("FECHA_DIA_T >= @Control_Dia")
group_by_seller = pedidos.groupby('VENDEDOR')
total_sold_by_seller = {gr: df.groupby('FECHA_DIA_T')['TOTAL_NETO_T'].sum() for gr, df in group_by_seller}
total_skus_by_seller = {gr: df.groupby('FECHA_DIA_T')['CODIGOREF'].nunique() for gr, df in group_by_seller}
ts_dict = {'sells': total_sold_by_seller, 'skus_count': total_skus_by_seller}
with open('Data/sc_time_series.pickle', 'wb') as file:
    pickle.dump(ts_dict, file)