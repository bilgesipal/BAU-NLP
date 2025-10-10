from sklearn.datasets import fetch_20newsgroups
import pickle


if __name__ == "__main__":
    newsgroups_train = fetch_20newsgroups(subset='train')
    cats = ['talk.politics.misc', 'sci.electronics']
    newsgroups_train = fetch_20newsgroups(subset='train', categories=cats)
    df_data=newsgroups_train.data
    print(len(df_data))
    with open('df_data.pkl', "wb") as f:
        pickle.dump(df_data, f)
    label = df_target = newsgroups_train.target[:len(df_data)]
    with open('label.pkl', "wb") as f:
        pickle.dump(label, f)