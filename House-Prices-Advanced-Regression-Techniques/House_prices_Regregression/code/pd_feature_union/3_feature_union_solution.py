import pandas as pd
from sklearn.datasets import load_iris
from sklearn.pipeline import make_pipeline

from pandas_feature_union import PandasFeatureUnion
from pandas_transform import PandasTransform


def main():
    raw_data = load_iris()
    data = pd.DataFrame(raw_data["data"], columns=raw_data["feature_names"])
    data.loc[:, "class"] = raw_data["target"]

    pipeline = PandasFeatureUnion([
        ("1", make_pipeline(
            PandasTransform(lambda X: X.loc[:, ["sepal length (cm)"]]),
            # other transformations
        )),
        ("2", make_pipeline(
            PandasTransform(lambda X: X.loc[:, ["sepal width (cm)"]]),
            # other transformations
        ))
    ])

    X = pipeline.fit_transform(data)
    print(X["sepal length (cm)"].mean())
    print(X["sepal width (cm)"].mean())


if __name__ == '__main__':
    main()
