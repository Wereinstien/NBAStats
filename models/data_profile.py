import pandas as pd
from pandas_profiling import ProfileReport


SEASONS_STATS_CSV = 'data/Seasons_Stats.csv'
SEASONS_STATS_1982_ONWARDS_CSV = 'data/Seasons_Stats_1982_Onwards.csv'
REPORT_HTML = 'data/report.html'
PROFILE_SAMPLE_SIZE = 2400


def pandas_options():
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)


def create_dataframe():
    df = pd.read_csv(SEASONS_STATS_CSV)

    # full dataframe info
    df.head(10)
    df.describe()
    df.info()

    # only want 1982 onward (RowId - 6449)
    df = df.loc[df[df.columns[0]] >= 6449]
    df.to_csv(SEASONS_STATS_1982_ONWARDS_CSV, index=False)

    return df


def create_profile(df):
    prof = ProfileReport(df)
    prof.to_file(output_file=REPORT_HTML)


def main():
    # df = create_dataframe()
    df = pd.read_csv(SEASONS_STATS_1982_ONWARDS_CSV)
    create_profile(df.sample(n=PROFILE_SAMPLE_SIZE))


if __name__ == '__main__':
    pandas_options()
    main()
