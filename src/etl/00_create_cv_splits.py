import pandas as pd

from sklearn.model_selection import KFold, StratifiedKFold


def create_double_cv(df, id_col, outer_splits, inner_splits, stratified=None, seed=88):
    df['outer'] = 888
    splitter = KFold if stratified is None else StratifiedKFold
    outer_spl = splitter(n_splits=outer_splits, shuffle=True, random_state=seed)
    outer_counter = 0
    for outer_train, outer_test in outer_spl.split(df) if stratified is None else outer_spl.split(df, df[stratified]):
        df.loc[outer_test, 'outer'] = outer_counter
        inner_spl = splitter(n_splits=inner_splits, shuffle=True, random_state=seed)
        inner_counter = 0
        df['inner{}'.format(outer_counter)] = 888
        inner_df = df[df['outer'] != outer_counter].reset_index(drop=True)
        # Determine which IDs should be assigned to inner train
        for inner_train, inner_valid in inner_spl.split(inner_df) if stratified is None else inner_spl.split(inner_df, inner_df[stratified]):
            inner_train_ids = inner_df.loc[inner_valid, id_col]
            df.loc[df[id_col].isin(inner_train_ids), 'inner{}'.format(outer_counter)] = inner_counter
            inner_counter += 1
        outer_counter += 1
    return df


df = pd.read_csv('../../data/stage_2_train_labels.csv')
df = df[['patientId', 'Target']].drop_duplicates().reset_index(drop=True)
cv_df = df[['patientId']].drop_duplicates().reset_index(drop=True)
cv_df = create_double_cv(cv_df, 'patientId', 10, 10)
df = df.merge(cv_df, on='patientId')
df['filename'] = [f'{f}.dcm' for f in df['patientId']]
df.to_csv('../../data/train_kfold.csv', index=False)


