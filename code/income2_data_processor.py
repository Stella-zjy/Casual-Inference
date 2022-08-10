import pandas as pd
import numpy as np

def get_income2_dataset():
    data = pd.read_csv('data/income_data2/ak91.csv',index_col=[0])
    data = data.dropna(axis=0)
    data.rename(columns={'log_wage':'wage','years_of_schooling':'education',
        'year_of_birth':'yob','quarter_of_birth':'qob','state_of_birth':'sob'}, inplace=True)


    def map_wage(wage):
        wage_mean = data['wage'].mean()
        if wage < wage_mean:
            return 0
        else:
            return 1

    def map_education(education):
        if education <= 12:
            return 0
        else:
            return 1

    def map_yob(yob):
        if yob <= 33:
            return 0
        elif yob <= 36:
            return 1
        else:
            return 2

    def map_sob(sob):
        if sob < 25:
            return 0
        elif sob < 39:
            return 1
        else:
            return 2

    data['wage'] = data['wage'].map(map_wage)
    data['education'] = data['education'].map(map_education)
    data['yob'] = data['yob'].map(map_yob)
    data['sob'] = data['sob'].map(map_sob)

    data.to_csv("data/income_data2/modified_ak91.csv")
    return data


if __name__ == '__main__':
    print(get_income2_dataset())
    
