import pandas as pd
import numpy as np

# Путь к оригинальному файлу
input_path = r'C:\Users\pasha\My_SMARTS_fragmentation\Fragmenting\mol_gnn_project\2_data\raw\BindingDB_All.tsv'
output_path = r'C:\Users\pasha\My_SMARTS_fragmentation\Fragmenting\mol_gnn_project\2_data\raw\BindingDB_cleaned_standardized.csv'

# 1. Загрузка оригинального файла
print("Loading original BindingDB file...")
df = pd.read_csv(input_path, sep='\t', low_memory=False)
print(f"Original shape: {df.shape}")

# 2. Оставляем только нужные столбцы
columns_needed = [
    'BindingDB Reactant_set_id',
    'Ligand SMILES',
    'Target Name',
    'Target Source Organism According to Curator or DataSource',
    'Ki (nM)',
    'IC50 (nM)',
    'EC50 (nM)',
    'Kd (nM)'
]
df = df[columns_needed]

# 3. Фильтрация строк: оставляем те, где есть хотя бы одно значение активности
activity_cols = ['Ki (nM)', 'IC50 (nM)', 'EC50 (nM)', 'Kd (nM)']
df = df.dropna(subset=activity_cols, how='all')
print(f"Shape after filtering: {df.shape}")

# # 4. Функция для очистки числовых значений с символами >, <
# def clean_numeric(value):
#     if pd.isna(value):
#         return np.nan
#     if isinstance(value, str):
#         value = value.strip()
#         if value.startswith('>') or value.startswith('<'):
#             value = value[1:]
#         try:
#             return float(value)
#         except:
#             return np.nan
#     return value
#
# # 5. Чистим значения в активности
# for col in activity_cols:
#     if col in df.columns:
#         df[col] = df[col].apply(clean_numeric)
#
# # 6. Функция для пересчёта в pKi, pIC50, pEC50, pKd
# def nM_to_pX(value_nM):
#     try:
#         molar = value_nM * 1e-9
#         if molar > 0:
#             return -np.log10(molar)
#         else:
#             return np.nan
#     except:
#         return np.nan
#
# # 7. Пересчитываем p-значения
# for col in activity_cols:
#     p_col = 'p' + col.replace(' (nM)', '')
#     df[p_col] = df[col].apply(nM_to_pX)

# 8. Сохраняем результат
print("Saving standardized data...")
df.to_csv(output_path, index=False)
print(f"Standardized data saved to {output_path}")
