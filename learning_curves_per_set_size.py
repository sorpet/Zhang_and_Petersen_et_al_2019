"""Run tests with arguments as: 
python3 learning_curves_per_set_size.py run_number set_ind"""

import sys
import argparse
import random
import pickle
sys.path.append('../AutomatedRecommendationTool')

from AutomaticRecommendationTool.art import *
import AutomaticRecommendationTool.plot as plot

parser = argparse.ArgumentParser(description='Run CV for learning curves plots.')
parser.add_argument('run_number', metavar='run_number', type=int,
                    help='Run number')
parser.add_argument('set_ind', metavar='set_ind', type=int, help='Index for a set size')
args = parser.parse_args()

run_number = args.run_number
set_ind = args.set_ind

edd_study_slug = 'petersen-et-al-2019'
edd_server = 'edd-test.jbei.org'

data_file = '../sdpetersen/strains_for_ART_one_hot.csv'
data_file_unique = '../sdpetersen/strains_for_ART_one_hot_unique_des.csv'

df = utils.load_study(data_file=data_file)
df_unique = utils.load_study(data_file=data_file_unique)

measurement = ['dgfp_dt']

noisy_line_name = utils.find_noisy_data(df, measurement, percentile=99, plot_flag=False)
df_filtered = utils.filter_noisy_data(df_unique, noisy_line_name, measurement, plot_flag=False)


def take_value(source, dest_obj, ind1, ind2=None, ind3=None):
    # Copy values from a dataframe

    temp = dest_obj.copy()
    if ind3 is None:
        if ind2 is None:
            temp[ind1] = source
        else:
            temp[ind1,ind2] = source
    else:
        temp[ind1,ind2,ind3] = source
    return temp

colnames = ['p1_1.0', 'p1_2.0', 'p1_3.0', 'p1_4.0', 'p1_5.0', 'p1_6.0',
            'p2_7.0', 'p2_8.0', 'p2_9.0', 'p2_10.0', 'p2_11.0', 'p2_12.0',
            'p3_13.0', 'p3_14.0', 'p3_15.0', 'p3_16.0', 'p3_17.0', 'p3_18.0',
            'p4_19.0', 'p4_20.0', 'p4_21.0', 'p4_22.0', 'p4_23.0', 'p4_24.0',
            'p5_25.0', 'p5_26.0', 'p5_27.0', 'p5_28.0', 'p5_29.0', 'p5_30.0']

art_params = {}
art_params['input_var'] = colnames
art_params['response_var'] = measurement
art_params['verbose'] = 0
art_params['seed'] = 1234
art_params['output_directory'] = '../sdpetersen/results_onehot/learning_curve_10foldCV_unique_designs'
art_params['build_model'] = False
art_params['recommend'] = False

num_runs = 1
num_folds = 10
set_sizes = np.array([10,  28,  46,  64,  82, 100, 118, 136, 154, 172, 190, 208, 227, 246])  # in terms of line names
num_cases = len(set_sizes)

zeros = np.zeros((num_runs, num_folds, num_cases))
metrics = pd.DataFrame.from_dict({'Metric': ['MAE', 'MRAE(%)', '$R^2$'],
                                  'Train': [zeros, zeros, zeros],
                                  'Test': [zeros, zeros, zeros]}).set_index('Metric')

line_names_all = df_filtered['Line Name'].str.split('-').str[0].unique().tolist()

i = set_ind
set_size = set_sizes[i]

for run in range(num_runs):

    line_names = random.sample(line_names_all, set_size)

    test_partitions = utils.partition(line_names, num_of_partitions=num_folds)
    train_partitions = [set(line_names) - set(test_partition) for test_partition in test_partitions]

    for j in range(num_folds):
        print(f'Run: {run+1}\nFold: {j+1}\nset_size: {set_size}')

        # Make sure test is not the same as train
        train_partition = train_partitions[j]
        test_partition = test_partitions[j]

        line_names_train = set([line + '-r' + str(i + 1) for line in train_partition for i in range(15)])
        line_names_test = set([line + '-r' + str(i + 1) for line in test_partition for i in range(15)])

        df_train = df_filtered[df_filtered['Line Name'].isin(line_names_train)]
        df_test = df_filtered[df_filtered['Line Name'].isin(line_names_test)]

        # Read train and test data separately:
        art_train = RecommendationEngine(df_train, **art_params)
        art_test = RecommendationEngine(df_test, **art_params)

        # build the predictive model:
        art_train.build_model()

        # Evaluate models on train data:
        art_train.evaluate_models()
        metrics['Train']['MAE'] = take_value(art_train.model_df[0]['MAE']['Ensemble Model'],
                                             metrics['Train']['MAE'], run, j, i)
        metrics['Train']['MRAE(%)'] = take_value(art_train.model_df[0]['MRAE(%)']['Ensemble Model'],
                                                 metrics['Train']['MRAE(%)'], run, j, i)
        metrics['Train']['$R^2$'] = take_value(art_train.model_df[0]['$R^2$']['Ensemble Model'],
                                               metrics['Train']['$R^2$'], run, j, i)

        # Evaluate models on test data:
        X_test = art_test.df['Input Variables'].values
        y_test = art_test.df['Response Variables'].values
        art_train.evaluate_models(X_test, y_test)
        metrics['Test']['MAE'] = take_value(art_train.model_df[0]['MAE']['Ensemble Model'], metrics['Test']['MAE'],
                                            run, j, i)
        metrics['Test']['MRAE(%)'] = take_value(art_train.model_df[0]['MRAE(%)']['Ensemble Model'],
                                                metrics['Test']['MRAE(%)'], run, j, i)
        metrics['Test']['$R^2$'] = take_value(art_train.model_df[0]['$R^2$']['Ensemble Model'],
                                              metrics['Test']['$R^2$'], run, j, i)

    metrics_filename = f'{art_train.outDir}/metrics_train_test_run{str(run_number)}_size{set_sizes[i]}'
    metrics.to_csv(metrics_filename + '.csv')
    with open(metrics_filename + '.pkl', 'wb') as output:  # Overwrites any existing file
        pickle.dump(metrics, output, -1)

