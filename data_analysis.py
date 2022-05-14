import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def data_collector(number_of_files=20000, phase='train'):

    df = pd.read_csv("data/" + phase + "/patient_0.psv", sep='|')
    df['id'] = 0  # TODO beware of test id
    for i in range(1, number_of_files):
        tmp = pd.read_csv("data/" + phase + "/patient_" + str(i) + ".psv", sep='|')
        tmp['id'] = i
        df = pd.concat([df, tmp])

    return df


def Nans_count(df: pd.DataFrame):
    percent_missing = df.isnull().sum() * 100 / len(df)
    nans = {k: v for k, v in sorted(percent_missing.to_dict().items(), key=lambda item: item[1], reverse=True)}

    return nans


def histograms(df: pd.DataFrame, presets: list):
    # histograms of different features
    for (col, xlab, color) in presets:
        hist = plt.figure()
        plt.hist(df[col], color=color)
        plt.title('Histogram of:' + xlab)
        plt.xlabel(xlab)
        plt.ylabel('Count in data')
        hist.savefig('Histogram of:' + col)  # TODO fuck your mom (somehow make it work)


def correlation(df: pd.DataFrame):
    # TODO make it normal to use
    corrs = {}
    for col in df:
        corrs[col] = df.corr()[col].abs().sort_values(ascending=False)[1:6]


    # heatmap = sns.heatmap(df.corr(), annot=True, cmap='BrBG')
    # heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 18}, pad=12)
    # plt.savefig('correlation.png', bbox_inches='tight')


def feature_distribution(data: pd.DataFrame):
    # Each row is now a patient (not an hour)
    df_per_person = data.copy().drop_duplicates(subset=['id'], keep='last').astype({'Age': 'int'})
    descb = data.describe()
    if not os.path.isfile('Statistical_stuff.csv'):
        descb.to_csv('Statistical_stuff.csv')

    # Sick vs Not sick
    # plot0 = plt.figure()
    # plt.hist(x=df_per_person['SepsisLabel'], bins=2, orientation='vertical')
    # plot0.show()
    num_sick = df_per_person.groupby('SepsisLabel').size()
    plot0 = plt.figure()
    plt.barh(df_per_person['SepsisLabel'].unique(), num_sick)
    plt.title('Distribution of ages in data')
    plt.xlabel('Sick or not')
    plt.ylabel('Count')
    if os.path.isfile('figure0.png'):
        os.remove('figure0.png')
    plot0.savefig('figure0')
    # count data by gender
    males_df = df_per_person[df_per_person['Gender'] == 1]
    females_df = df_per_person[df_per_person['Gender'] == 0]

    diff_gender = len(males_df/len(females_df))
    print("The difference between males and females is", diff_gender, "\n")

    #Ages distb
    bins = [17, 30, 40, 50, 60, 70, 80, 90, 120]  # TODO minimum in age?
    labels = ['17-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90+']
    df_per_person['Agerange'] = pd.cut(df_per_person['Age'], bins, labels=labels, include_lowest=True)
    ages_sum = df_per_person.groupby('Agerange').size()
    plot1 = plt.figure()
    plt.bar(labels, ages_sum.tolist(), color='blue')
    plt.title('Distribution of ages in data')
    plt.xlabel('Age groups')
    plt.ylabel('Count')
    if os.path.isfile('figure1.png'):
        os.remove('figure1.png')
    plot1.savefig('figure1')

    plot2 = plt.figure()
    males_sick_mean = males_df.groupby(['Age'])['SepsisLabel'].mean()
    females_sick_mean = females_df.groupby(['Age'])['SepsisLabel'].mean()
    plt.scatter(males_df['Age'].unique(), males_sick_mean, label='Male')
    plt.scatter(females_df['Age'].unique(), females_sick_mean, label='Females')
    plt.title('Sick patients given age and gender')
    plt.xlabel('age')
    plt.ylabel('Count')
    if os.path.isfile('figure2.png'):
        os.remove('figure2.png')
    plot2.savefig('figure2')

    # histograms
    presets = [('HospAdmTime', 'Hospital Admission Times', 'blue'),
               ('ICULOS', 'ICU Admission Times', 'red'),
               ('HR', 'Heart Beats Per Minute', 'seagreen')]  # TODO check Hosp distb on the whole data
    histograms(df_per_person, presets)

    correlation(data)
    # [[c for c in data.columns if c in
    #                       ['HR', 'Age', 'Temp', 'O2Sat', 'ICULOS', 'HospAdmTime', 'SepsisLabel']]]

    # TODO diff between sick genders
    # TODO os.if_file_exists then replace it and shit to all the graphs


def dora_the_data_explorer():

    head = pd.read_csv("data/train/patient_5.psv", sep='|', nrows=1)
    print("Those are features in our dataset:", list(head.columns))
    print("Number of features:", len(list(head.columns)))

    df = data_collector(1000, phase='train')
    nans = Nans_count(df)
    feature_distribution(df)
    print(nans)


def main():
    dora_the_data_explorer()


if __name__ == '__main__':
    main()