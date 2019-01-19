import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#Creates dataframe from given csv file
def create_dataframe():
    all_data = pd.read_csv("profiles.csv")
    return all_data

#normalized data
def normalize_data(dataframe):
    temp_df = dataframe.dropna(subset = ['body_code','drugs_code','smokes_code','status_code','drinks_code'])
    temp_df = (temp_df - temp_df.min()) / (temp_df.max() - temp_df.min())
    return temp_df.sample(n=5000)

#Training and validation sets
def learn(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y.ravel(), test_size=0.2, random_state=50)
    accuracy = []

    for k in range(1,150):
        neighbor = KNeighborsClassifier(n_neighbors = k)
        neighbor.fit(x_train, y_train)
        accuracy.append(neighbor.score(x_test, y_test))

    return accuracy

#Graph data in question
def scatter_data(perc, labels, tle):

    plt.pie(perc)
    plt.legend(labels,loc="upper left",bbox_to_anchor=(-0.3,1.),fontsize=8)
    plt.title(tle)
    plt.show()

def array_data(df,colI, colII, colNo, k):
    lst = []
    for i in range(0,k):
        lst.append(len(df[(df[colI]==colNo) & (df[colII]==i)]))

    return lst

def main():
    df = create_dataframe()

    #Data point maps
    body_mapping = {"rather not say": 0, "used up": 1, "thin": 2, "skinny": 3, "curvy": 4, "athletic": 5, "fit": 6, "average": 7, "full figured": 8, "jacked": 9, "overweight": 10}
    status_mapping = {"unknown": 0, "single": 1, "available": 2, "seeing someone": 3, "married": 4}
    smoke_mapping = {"no": 0, "trying to quit": 1, "when drinking": 2, "sometimes": 3, "yes": 4}
    drink_mapping = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}
    drug_mapping = {"never": 0, "sometimes": 1, "often": 2}

    df["body_code"] = df.body_type.map(body_mapping)
    df["smokes_code"] = df.smokes.map(smoke_mapping)
    df["drinks_code"] = df.drinks.map(drink_mapping)
    df["drugs_code"] = df.drugs.map(drug_mapping)
    df["status_code"] = df.status.map(status_mapping)

    stat = ['Unknown','Single','Available','Seeing Someone','Married']
    pie_data_I = array_data(df,"drinks_code","status_code",5,5)
    pie_data_II = array_data(df, "drinks_code","status_code",0,5)
    pie_data_III = array_data(df, "smokes_code","status_code",0,5)
    pie_data_IV = array_data(df, "smokes_code","status_code",4,5)

    scatter_data(pie_data_I, stat, "Relationship Status of Heavy Drinkers")
    scatter_data(pie_data_II, stat, "Relationship Status of Non-drinkers")
    scatter_data(pie_data_III, stat, "Relationship Status of Non-smokers")
    scatter_data(pie_data_IV, stat, "Relationship Status of Regular Smokers")

    df["status_code"].replace([1,2], 0, inplace=True)
    df["status_code"].replace([3,4], 1, inplace=True)
    
    feat_data = normalize_data(df[['body_code', 'drugs_code', 'drinks_code', 'smokes_code', 'status_code']])

    accuracies = learn(np.asarray(feat_data[['body_code','drugs_code','drinks_code','smokes_code']]), np.asarray(feat_data[['status_code']]))
    k_list = range(1,150)

    plt.plot(k_list, accuracies)

    plt.xlabel("k")
    plt.ylabel("Status Accuracy")
    plt.title("Status Classification Accuracy")
    plt.show()

if __name__ == "__main__":
    main()
