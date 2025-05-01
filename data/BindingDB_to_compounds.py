import pandas as pd

# 1. Загружаем файл
input_path = r'C:\Users\pasha\My_SMARTS_fragmentation\Fragmenting\mol_gnn_project\2_data\raw\BindingDB_cleaned_standardized.csv'
print("Loading cleaned standardized file...")
df = pd.read_csv(input_path, low_memory=False)
print(f"Original shape: {df.shape}")

# 2. Переименование и выбор нужных столбцов
# Столбцы для финального датасета
columns_mapping = {
    'Ligand SMILES': 'SMILES',
    'Target Name': 'Target',
    'Target Source Organism According to Curator or DataSource': 'Organism',
    'pKi': 'pKi',
    'pIC50': 'pIC50',
    'pEC50': 'pEC50',
    'pKd': 'pKd'
}

# Фильтруем и переименовываем
df_final = df[list(columns_mapping.keys())].rename(columns=columns_mapping)

# 3. Добавляем новый ID (начиная с 1)
df_final.insert(0, 'ID', range(1, len(df_final) + 1))

print(f"Shape after processing: {df_final.shape}")

# 4. Сохраняем результат
output_path = r'C:\Users\pasha\My_SMARTS_fragmentation\Fragmenting\mol_gnn_project\2_data\raw\compounds.csv'
df_final.to_csv(output_path, index=False)
print(f"Final processed data saved to {output_path}")
