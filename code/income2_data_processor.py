import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
# relative_path = "D:/Workspace/Casual-Inference"
relative_path = ".."


class ak91Data:
    @staticmethod
    def get_dataset():
        data = pd.read_csv(relative_path + "/data/income_data2/ak91.csv", index_col=[0])
        data = data.dropna(axis=0)
        data.rename(columns={'log_wage': 'wage', 'years_of_schooling': 'education',
                             'year_of_birth': 'yob', 'quarter_of_birth': 'qob', 'state_of_birth': 'sob'}, inplace=True)

        def map_wage(wage):
            # wage_mean = data['wage'].mean()
            # if wage < wage_mean:
            # if wage < 6.257376:
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
        # data['yob'] = data['yob'].map(map_yob)
        # data['sob'] = data['sob'].map(map_sob)

        data.to_csv(relative_path + "/data/income_data2/modified_ak91.csv")
        return data

    @staticmethod
    def get_stage1_input():
        ds = pd.read_csv(relative_path + "/data/income_data2/modified_ak91.csv")
        zx = ds[["qob", "yob", "sob"]]
        p = ds["education"]
        return zx, p

    @staticmethod
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

class caEducationalData:
    @staticmethod
    def get_dataset():
        data = pd.read_csv(relative_path + "/data/income_data2/ca-educational-attainment-personal-income-2008-2014.csv", index_col=[0])
        data = data.dropna(axis=0)
        data.rename(columns={'Age': 'age', 'Gender': 'gender',
                             'Educational Attainment': 'education', 'Personal Income': 'income', 'Population Count': 'count'}, inplace=True)

        def map_age(age):
            if age == "00 to 17":
                return 0
            if age == "18 to 64":
                return 1
            if age == "65 to 80+":
                return 2
            else:
                return 3

        def map_gender(gender):
            if gender == "Female":
                return 0
            if gender == "Male":
                return 1
            else:
                return 2

        def map_education(education):
            if education == "No high school diploma":
                return 0
            if education == "High school or equivalent":
                return 0
            if education == "Some college, less than 4-yr degree":
                return 1
            if education == "Bachelor's degree or higher":
                return 1
            else:
                return 2

        def map_income(income):
            if income == "No Income":
                return 0
            if income == "$5,000 to $9,999":
                return 0
            if income == "$10,000 to $14,999":
                return 0
            if income == "$15,000 to $24,999":
                return 0
            if income == "$25,000 to $34,999":
                return 1
            if income == "$35,000 to $49,999":
                return 1
            if income == "$50,000 to $74,999":
                return 1
            if income == "$75,000 and over":
                return 1
            else:
                return 2

        data['age'] = data['age'].map(map_age)
        data['gender'] = data['gender'].map(map_gender)
        data['education'] = data['education'].map(map_education)
        data['income'] = data['income'].map(map_income)
        data.to_csv(relative_path + "/data/income_data2/modified_ca.csv")
        return data

    @staticmethod
    def get_stage1_input():
        ds = pd.read_csv(relative_path + "/data/income_data2/modified_ca.csv")
        zx = ds[["count", "age", "gender"]]
        p = ds["education"]
        return zx, p

    @staticmethod
    def get_stage2_input():
        data = pd.read_csv(relative_path + "/data/income_data2/ca_sample_data.csv")
        p = data.iloc[:, 3:].apply(lambda x: x.mean(), axis=1)
        data.insert(loc=len(data.columns), column='education', value="1,0")
        data = pd.concat([data, p], axis=1).rename(columns={0: 'probability'})
        y = pd.read_csv(relative_path + "/data/income_data2/modified_ca.csv")["income"]
        data = pd.concat([data, y], axis=1)
        df_split_row = data.drop('education', axis=1).join(
            data['education'].str.split(',', expand=True).stack().reset_index(level=1, drop=True).rename('education')) \
            .reset_index(level=0)
        probability = df_split_row['probability']
        data = df_split_row.drop('probability', axis=1)
        for key in probability.keys():
            if key % 2 == 1:
                probability.iloc[key] = 1 - probability.iloc[key]
        xp = data[["count", "gender", "education", "income"]]
        xp["education"] = pd.to_numeric(xp["education"])
        return xp, probability

class IPUMSData:
    @staticmethod
    def get_dataset():
        data = pd.read_csv(relative_path + "/data/IPUMS_IncomeData/IPUMS_IncomeData.csv", index_col=[0])
        data = data.dropna(axis=0)
        data = data[['AGE', 'SEX', 'MARST', 'RACE', 'BPL', 'EDUC', 'OCC', 'IND', 'UHRSWORK', 'INCTOT']]
        data.rename(columns={'AGE': 'age', 'SEX': 'gender', 'MARST': 'marital_status', 'RACE': 'race',
                             'BPL': 'birth_place', 'EDUC': 'education', 'OCC': 'occupation', 'IND': 'industry',
                             'UHRSWORK': 'hours_per_week', 'INCTOT': 'income'}, inplace=True)
        #print(data.groupby('occupation').agg({'occupation': 'count'}))
        def map_marital_status(x):
            if x == 1:
                return 0
            if x == 6:
                return 1
            if x == 4:
                return 2
            else:
                return 3

        def map_race(x):
            if x == 1:
                return 0
            if x == 2:
                return 1
            if x == 7:
                return 2
            else:
                return 3

        def map_birthplace(x):
            if x <= 199:
                return 0
            if x <= 300:
                return 1
            if x <= 499:
                return 2
            if x <= 599:
                return 3
            else:
                return 4

        def map_education(x):
            if x <= 6:
                return 0
            else:
                return 1

        def map_income(x):
            if x < 50000:
                return 0
            return 1

        def map_industry(x):
            if x <= 60:
                return 0
            if x < 400:
                return 1
            if x < 500:
                return 2
            if x < 700:
                return 3
            if x < 800:
                return 4
            if x < 900:
                return 5
            return 6

        data['marital_status'] = data['marital_status'].map(map_marital_status)
        data['race'] = data['race'].map(map_race)
        data['birth_place'] = data['birth_place'].map(map_birthplace)
        data['education'] = data['education'].map(map_education)
        data['industry'] = data['industry'].map(map_industry)
        data['income'] = data['income'].map(map_income)
        data.to_csv(relative_path + "/data/IPUMS_IncomeData/modified_IPUMS_IncomeData.csv")
        return data

    @staticmethod
    def get_stage1_input():
        ds = pd.read_csv(relative_path + "/data/IPUMS_IncomeData/modified_IPUMS_IncomeData.csv")
        zx = ds[["age", "gender", "marital_status", "race", "birth_place", "industry", "hours_per_week"]]
        p = ds["education"]
        return zx, p

    @staticmethod
    def get_stage2_input():
        data = pd.read_csv(relative_path + "/data/IPUMS_IncomeData/sample_data.csv")
        p = data.iloc[:, 8:].apply(lambda x: x.mean(), axis=1)
        data.insert(loc=len(data.columns), column='education', value="1,0")
        data = pd.concat([data, p], axis=1).rename(columns={0: 'probability'})
        y = pd.read_csv(relative_path + "/data/IPUMS_IncomeData/modified_IPUMS_IncomeData.csv")["income"]
        data = pd.concat([data, y], axis=1)
        df_split_row = data.drop('education', axis=1).join(
            data['education'].str.split(',', expand=True).stack().reset_index(level=1, drop=True).rename('education')) \
            .reset_index(level=0)
        probability = df_split_row['probability']
        data = df_split_row.drop('probability', axis=1)
        for key in probability.keys():
            if key % 2 == 1:
                probability.iloc[key] = 1 - probability.iloc[key]
        xp = data[["age", "gender", "marital_status", "birth_place", "industry", "hours_per_week", "education", "income"]]
        xp["education"] = pd.to_numeric(xp["education"])
        return xp, probability

if __name__ == '__main__':
    # xp, pr = caEducationalData.get_stage2_input()
    # print(max(pr))
    # print(min(pr))
    # print(pr)
    print(caEducationalData.get_dataset())
