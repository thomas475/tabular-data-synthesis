from sklearn import datasets
from sklearn import tree
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
import category_encoders as ce
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier

from framework.samplers import *


def racog_test():
    iris = datasets.load_iris()
    X = iris['data']
    y = iris['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    X_train_encoded = ce.OneHotEncoder().fit_transform(X_train, random_state=42)
    X_test_encoded = ce.OneHotEncoder().fit_transform(X_test, random_state=42)

    print('starting sampling ...')

    train_encoded = X_train_encoded.copy()
    temp = []
    for i in range(len(y_train)):
        temp.append('true')
    train_encoded['target'] = temp

    racog_sampler = RACOG()
    train_encoded_augmented = racog_sampler.resample(dataset=train_encoded, num_instances=150, class_attr='target')

    X_train_encoded_augmented = train_encoded_augmented.drop(columns=['target'])

    print('finished sampling ...')

    complex_depth = 15
    simple_depth = 3

    complex_tree = tree.DecisionTreeClassifier(max_depth=complex_depth, random_state=42)
    simple_tree = tree.DecisionTreeClassifier(max_depth=simple_depth, random_state=42)
    augmented_simple_tree = tree.DecisionTreeClassifier(max_depth=simple_depth, random_state=42)

    print('starting training ...')

    X_train_encoded_augmented.columns = X_train_encoded.columns

    complex_tree.fit(X_train_encoded, y_train)
    y_train_augmented = complex_tree.predict(X_train_encoded_augmented)

    simple_tree.fit(X_train_encoded, y_train)
    augmented_simple_tree.fit(pd.concat([X_train_encoded, X_train_encoded_augmented], ignore_index=True),
                              np.append(y_train, y_train_augmented))

    print('finishing training ...')

    print('complex:', complex_tree.score(X_test_encoded, y_test))
    print('simple:', simple_tree.score(X_test_encoded, y_test))
    print('augmented:', augmented_simple_tree.score(X_test_encoded, y_test))


def racog_test_2():
    iris = datasets.load_iris()
    dataset = pd.DataFrame(iris['data'])
    dataset.columns = iris['feature_names']

    racog = ProportionalRACOGSampler(1)
    augmented_dataset = racog.fit_transform(dataset)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(augmented_dataset)


def proportional_smote():
    iris = datasets.load_iris()
    X = pd.DataFrame(iris['data'])
    X.columns = iris['feature_names']
    y = iris['target']

    smote = ProportionalSMOTESampler(1)
    augmented_dataset = smote.fit_transform(X, y)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(augmented_dataset)


def unlabeled_smote():
    iris = datasets.load_iris()
    X = pd.DataFrame(iris['data'])
    X.columns = iris['feature_names']
    y = iris['target']

    smote = UnlabeledSMOTESampler(1)
    augmented_dataset = smote.fit_transform(X)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(augmented_dataset)


def proportional_racog():
    iris = datasets.load_iris()
    X = pd.DataFrame(iris['data'])
    X.columns = iris['feature_names']
    y = iris['target']

    racog = ProportionalRACOGSampler(1)
    augmented_dataset = racog.fit_transform(X, y)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(augmented_dataset)


def unlabeled_racog():
    iris = datasets.load_iris()
    X = pd.DataFrame(iris['data'])
    X.columns = iris['feature_names']
    y = iris['target']

    racog = UnlabeledRACOGSampler(1)
    augmented_dataset = racog.fit_transform(X)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(augmented_dataset)


def vanilla_gan_test_1():
    iris = datasets.load_iris()
    X = pd.DataFrame(iris['data'])
    X.columns = iris['feature_names']
    y = iris['target']

    gan = ProportionalVanillaGANSampler(1, epochs=50, batch_size=25)
    augmented_dataset = gan.fit_transform(X, y)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(augmented_dataset)


def vanilla_gan_test_2():
    iris = datasets.load_iris()
    X = pd.DataFrame(iris['data'])
    X.columns = iris['feature_names']
    y = iris['target']

    gan = UnlabeledVanillaGANSampler(1, epochs=50, batch_size=25)
    augmented_dataset = gan.fit_transform(X, y)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(augmented_dataset)


def load_iris():
    iris = datasets.load_iris()
    X = pd.DataFrame(iris['data'])
    X.columns = iris['feature_names']
    y = iris['target']

    return X, y


def pipeline_test():
    complex_tree = DecisionTreeClassifier(max_depth=15)
    simple_tree = DecisionTreeClassifier(max_depth=5)
    augmented_simple_tree = DecisionTreeClassifier(max_depth=5)

    sample_multiplication_factor = 1

    samplers = [
        ProportionalSMOTESampler,
        UnlabeledSMOTESampler,
        ProportionalRACOGSampler,
        UnlabeledRACOGSampler,
        UnlabeledVanillaGANSampler
    ]

    iris = datasets.load_iris()
    X = pd.DataFrame(iris['data'])
    X.columns = iris['feature_names']
    y = iris['target']

    for sampler in samplers:
        pipeline = Pipeline(steps=[
            ('encoder', ce.OneHotEncoder()),
            ('scaler', RobustScaler()),
            ('sampler', sampler(sample_multiplication_factor))
        ])

        print(pipeline.fit_transform(X=X.copy(), y=y.copy()))

def cgan_test_1():
    iris = datasets.load_iris()
    X = pd.DataFrame(iris['data'])
    X.columns = iris['feature_names']
    y = iris['target']

    gan = ProportionalConditionalGANSampler(1, epochs=50, batch_size=25)
    augmented_dataset = gan.fit_transform(X, y)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(augmented_dataset)

def cgan_test_2():
    iris = datasets.load_iris()
    X = pd.DataFrame(iris['data'])
    X.columns = iris['feature_names']
    y = iris['target']

    gan = UnlabeledConditionalGANSampler(1, epochs=50, batch_size=25)
    augmented_dataset = gan.fit_transform(X, y)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(augmented_dataset)

def dragan_test_1():
    iris = datasets.load_iris()
    X = pd.DataFrame(iris['data'])
    X.columns = iris['feature_names']
    y = iris['target']

    gan = UnlabeledDRAGANSampler(1, discriminator_updates_per_step=3, epochs=50, batch_size=25)
    augmented_dataset = gan.fit_transform(X, y)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(augmented_dataset)

def dragan_test_2():
    iris = datasets.load_iris()
    X = pd.DataFrame(iris['data'])
    X.columns = iris['feature_names']
    y = iris['target']

    gan = ProportionalDRAGANSampler(1, discriminator_updates_per_step=3, epochs=50, batch_size=25)
    augmented_dataset = gan.fit_transform(X, y)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(augmented_dataset)


# proportional_smote()
# unlabeled_smote()
# proportional_racog()
# unlabeled_racog()
# unlabeled_vanilla_gan()
vanilla_gan_test_1()
vanilla_gan_test_2()
# pipeline_test()
# cgan_test_1()
# cgan_test_2()
# dragan_test_1()
# dragan_test_2()