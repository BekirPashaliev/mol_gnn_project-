import pandas as pd
import numpy as np

# 1. Загружаем ранее очищенный файл
file_path = r'C:\Users\pasha\My_SMARTS_fragmentation\Fragmenting\mol_gnn_project\2_data\raw\BindingDB_cleaned.csv'
print("Loading cleaned file...")
df = pd.read_csv(file_path, low_memory=False)
print(f"Loaded shape: {df.shape}")

# 2. Функция для обработки текстовых значений (>100, <0.1 и т.д.)
def clean_numeric(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        value = value.strip()
        if value.startswith('>'):
            # Значения больше — можно проигнорировать ">"
            value = value[1:]
        elif value.startswith('<'):
            # Значения меньше — берём как есть (или можно пометить отдельно)
            value = value[1:]
        try:
            return float(value)
        except:
            return np.nan
    return value

# 3. Чистим числовые значения в оригинальных колонках
activity_cols = ['Ki (nM)', 'IC50 (nM)', 'EC50 (nM)', 'Kd (nM)']

for col in activity_cols:
    if col in df.columns:
        df[col] = df[col].apply(clean_numeric)

# 4. Функция для пересчёта в pKi, pIC50 и т.п.
def nM_to_pX(value_nM):
    try:
        molar = value_nM * 1e-9
        if molar > 0:
            return -np.log10(molar)
        else:
            return np.nan
    except:
        return np.nan

# 5. Пересчитываем p-значения
for col in activity_cols:
    p_col = 'p' + col.replace(' (nM)', '')
    df[p_col] = df[col].apply(nM_to_pX)

# 6. Удаляем отдельные p-значения вне диапазона [3, 10]
p_cols = ['pKi', 'pIC50', 'pEC50', 'pKd']
for col in p_cols:
    df.loc[(df[col] < 3) | (df[col] > 10), col] = np.nan

# 7. Удаляем строки, где все p-значения пустые
df = df.dropna(subset=p_cols, how='all')

print(f"Shape after cleaning p-values: {df.shape}")

# 9. Умная фильтрация по организму
organisms_to_keep = ['Homo sapiens', "Rattus norvegicus", "Mus musculus", "Rattus"]

# Список допустимых таргетов для выбранных организмов
valid_targets = df[df['Target Source Organism According to Curator or DataSource'].isin(organisms_to_keep)]['Target Name'].unique()

# Фильтрация с учетом NaN в организме, но валидного таргета
df = df[
    (df['Target Source Organism According to Curator or DataSource'].isin(organisms_to_keep)) |
    (
        (df['Target Source Organism According to Curator or DataSource'].isna()) &
        (df['Target Name'].isin(valid_targets))
    )
]

print(f"Shape after smart organism filtering: {df.shape}")
print(f"Unique organisms left: {df['Target Source Organism According to Curator or DataSource'].nunique()}")


# 10. Заполняем пропущенные организмы на основе таргетов
# Строим словарь: Target Name -> Organism (самый частый организм для этого таргета)
target_to_organism = (
    df[df['Target Source Organism According to Curator or DataSource'].notna()]
    .groupby('Target Name')['Target Source Organism According to Curator or DataSource']
    .agg(lambda x: x.value_counts().idxmax())
    .to_dict()
)

# Заполняем пропуски

def fill_organism(row):
    if pd.isna(row['Target Source Organism According to Curator or DataSource']):
        return target_to_organism.get(row['Target Name'], np.nan)
    else:
        return row['Target Source Organism According to Curator or DataSource']

df['Target Source Organism According to Curator or DataSource'] = df.apply(fill_organism, axis=1)

print("Организмы успешно заполнены для молекул с пустыми значениями.")


# 8. Фильтрация таргетов по количеству молекул (>1000)
target_counts = df['Target Name'].value_counts()
targets_to_keep = target_counts[target_counts > 1000].index
df = df[df['Target Name'].isin(targets_to_keep)]

print(f"Shape after filtering targets with >1000 molecules: {df.shape}")
print(f"Number of unique targets after filtering: {df['Target Name'].nunique()}")


# 11. Сохраняем обратно
output_path = r'C:\Users\pasha\My_SMARTS_fragmentation\Fragmenting\mol_gnn_project\2_data\raw\BindingDB_cleaned_standardized.csv'
df.to_csv(output_path, index=False)
print(f"Standardized and filtered data saved to {output_path}")