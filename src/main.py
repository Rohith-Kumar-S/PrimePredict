from data.dataloader import DataLoader
from preproccessing.datapreprocessing import DataPreprocessor
from features.featureengineering import FeatureEngineering


def main():
    data = DataLoader()
    preprocessed_data = DataPreprocessor(
        data.purchases, data.categories, data.products
    ).output()
    df =  FeatureEngineering(
        preprocessed_data,
        data.holidays,
        data.holidays_past_2021,
        data.inflation,
        data.amazon_events,
    ).output()
    
    print(df)


if __name__ == "__main__":
    main()
