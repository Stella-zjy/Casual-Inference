import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
# relative_path = "D:/Workspace/Casual-Inference"
relative_path = ".."


def get_income2_dataset():
    data = pd.read_csv(relative_path + "/data/income_data2/ak91.csv", index_col=[0])
    data = data.dropna(axis=0)
    data.rename(columns={'log_wage': 'wage', 'years_of_schooling': 'education',
                         'year_of_birth': 'yob', 'quarter_of_birth': 'qob', 'state_of_birth': 'sob'}, inplace=True)

    def map_wage(wage):
        #wage_mean = data['wage'].mean()
        #if wage < wage_mean:
        #if wage < 6.257376:
        if wage < 5.952494:
            return 0
        else:
            return 1

    def map_education(education):
        if education <= 12:
            return 0
        else:
            return 1

    def map_yob(yob):
        if yob <= 31:
            return 0
        elif yob <= 33:
            return 1
        elif yob <= 35:
            return 2
        elif yob <= 37:
            return 3
        elif yob <= 39:
            return 4
        else:
            return 5

    def map_sob(sob):
        if sob <= 10:
            return 0
        elif sob <= 20:
            return 1
        elif sob <= 30:
            return 2
        elif sob <= 40:
            return 3
        elif sob <= 50:
            return 4
        else:
            return 5

    data['wage'] = data['wage'].map(map_wage)
    data['education'] = data['education'].map(map_education)
    #data['yob'] = data['yob'].map(map_yob)
    #data['sob'] = data['sob'].map(map_sob)

    data.to_csv(relative_path + "/data/income_data2/modified_ak91.csv")
    return data


def get_stage1_input():
    ds = pd.read_csv(relative_path + "/data/income_data2/modified_ak91.csv")
    zx = ds[["qob", "yob", "sob"]]
    p = ds["education"]
    return zx, p


def get_stage2_input():
    data = pd.read_csv(relative_path + "/data/income_data2/sample_data.csv")
    p = data.iloc[:, 4:].apply(lambda x: x.mean(), axis=1)
    data.insert(loc=len(data.columns), column='education', value="1,0")
    data = pd.concat([data, p], axis=1).rename(columns={0: 'probability'})
    y = pd.read_csv(relative_path + "/data/income_data2/modified_ak91.csv")["wage"]
    data = pd.concat([data, y], axis=1)
    df_split_row = data.drop('education', axis=1).join(
        data['education'].str.split(',', expand=True).stack().reset_index(level=1, drop=True).rename('education')) \
        .reset_index(level=0)
    probability = df_split_row['probability']
    data = df_split_row.drop('probability', axis=1)
    for key in probability.keys():
        if key % 2 == 1:
            probability.iloc[key] = 1 - probability.iloc[key]
    xp = data[["yob", "sob", "education", "wage"]]
    xp["education"] = pd.to_numeric(xp["education"])
    return xp, probability


if __name__ == '__main__':
    print(get_income2_dataset())
    #get_stage2_input()
