def scan(dataset):
    """ Computes statistical properties that are used for the
    later dataset generation. These properties are provided by
    the user in the frontend of the final integration into the
    platform. Therefore, this method simulates the behavior of
    the user.

    Args:
        :params dataset: [DataFrame] original dataset (usually only accessible to the user)

    Returns:
        :return feature_names: [List:String] feature names
        :return [List:String] feature names
        :return [List:String] feature names
        :return [List:String] feature names
        :return [List:String] feature names
        feature_names, min_, max_, unique, mean, std, skew

    """
    feature_names = [col for col in dataset.columns]
    stat_props = dataset.agg(["min", "max", "mean", "skew", "std"])
    min_ = list(stat_props.iloc[0])
    max_ = list(stat_props.iloc[1])
    mean = list(stat_props.iloc[2])
    skew = list(stat_props.iloc[3])
    std = list(stat_props.iloc[4])
    unique = [dataset[col].unique() for col in dataset.columns]
    return feature_names, min_, max_, unique, mean, std, skew






