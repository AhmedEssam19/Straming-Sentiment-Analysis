import pandas as pd
import glob


def main():
    with open('keyword.txt') as file:
        keyword = file.read().strip()

    full_df = pd.DataFrame(columns=['word', 'polarity'])
    for file_name in glob.glob(f'parc_{keyword}/*'):
        if '.parquet' not in file_name:
            continue

        df = pd.read_parquet(file_name)
        full_df = full_df.append(df)

    tweets_num = full_df.shape[0]
    polarity_percentage = full_df['polarity'].value_counts() / tweets_num * 100
    print(f"There are {tweets_num} tweets streamed.")
    for polarity in polarity_percentage.index:
        print('{:.2f}% of the tweets are {}'.format(polarity_percentage[polarity], polarity))


if __name__ == "__main__":
    main()
