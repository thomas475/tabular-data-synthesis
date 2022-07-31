import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import pairwise_distances
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon


# returns the fifth percentile of distance to closest record (dcr) and nearest neighbour distance ratio (nndr) between two datasets.
# taken from https://github.com/Team-TUD/CTAB-GAN/blob/main/model/eval/evaluation.py
def calculate_dcr_nndr(real_dataset, synthetic_dataset):
    # Scaling real and synthetic data samples
    scalerR = StandardScaler()
    scalerR.fit(real_dataset)
    scalerF = StandardScaler()
    scalerF.fit(synthetic_dataset)
    df_real_scaled = scalerR.transform(real_dataset)
    df_fake_scaled = scalerF.transform(synthetic_dataset)

    # Computing pair-wise distances between real and synthetic
    dist_rf = pairwise_distances(df_real_scaled, Y=df_fake_scaled, metric='minkowski')

    # Computing first and second smallest nearest neighbour distances between real and synthetic
    smallest_two_indexes_rf = [dist_rf[i].argsort()[:2] for i in range(len(dist_rf))]
    smallest_two_rf = [dist_rf[i][smallest_two_indexes_rf[i]] for i in range(len(dist_rf))]

    # Computing 5th percentiles for DCR and NNDR between real and synthetic datasets
    min_dist_rf = np.array([i[0] for i in smallest_two_rf])
    fifth_perc_rf = np.percentile(min_dist_rf, 5)
    nn_ratio_rf = np.array([i[0] / i[1] for i in smallest_two_rf])
    nn_fifth_perc_rf = np.percentile(nn_ratio_rf, 5)

    return fifth_perc_rf, nn_fifth_perc_rf

# returns the jason-shennon distance for categorical columns and wasserstein distance for continous columns
# taken from https://github.com/Team-TUD/CTAB-GAN/blob/main/model/eval/evaluation.py
def calculate_jsd_wd(real, fake, cat_cols):
    # Lists to store the results of statistical similarities for categorical and numeric columns respectively
    cat_stat = []
    num_stat = []

    for column in real.columns:

        if column in cat_cols:
            real_column = real[column].astype(int)
            fake_column = fake[column].astype(int)

            # Computing the real and synthetic probabibility mass distributions (pmf) for each categorical column
            real_pmf = (real_column.value_counts() / real_column.value_counts().sum())
            fake_pmf = (fake_column.value_counts() / fake_column.value_counts().sum())

            real_categories = real_column.value_counts().keys().tolist()
            fake_categories = fake_column.value_counts().keys().tolist()

            categories = real_categories.copy()
            categories.extend(category for category in fake_categories if category not in real_categories)

            # Ensuring the pmfs of real and synthetic data have the categories within a column in the same order
            sorted_categories = sorted(categories)

            real_pmf_ordered = []
            fake_pmf_ordered = []

            for i in sorted_categories:
                # If a category of a column is not found in the real dataset, pmf of zero is assigned
                if i in real_pmf:
                    real_pmf_ordered.append(real_pmf[i])
                else:
                    real_pmf_ordered.append(0)

                # If a category of a column is not generated in the synthetic dataset, pmf of zero is assigned
                if i in fake_pmf:
                    fake_pmf_ordered.append(fake_pmf[i])
                else:
                    fake_pmf_ordered.append(0)

            # Computing the statistical similarity between real and synthetic pmfs
            cat_stat.append(jensenshannon(real_pmf_ordered, fake_pmf_ordered, 2.0))

        else:
            # Scaling the real and synthetic numerical column values between 0 and 1 to obtained normalized statistical similarity
            scaler = MinMaxScaler()
            scaler.fit(real[column].values.reshape(-1, 1))
            l1 = scaler.transform(real[column].values.reshape(-1, 1)).flatten()
            l2 = scaler.transform(fake[column].values.reshape(-1, 1)).flatten()

            # Computing the statistical similarity between scaled real and synthetic numerical distributions
            num_stat.append(wasserstein_distance(l1, l2))

    return np.mean(num_stat), np.mean(cat_stat)
