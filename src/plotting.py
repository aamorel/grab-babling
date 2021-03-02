import pandas as pd
import numpy as np
import utils
import matplotlib.pyplot as plt
import joypy
import glob
import json
import seaborn as sns
import re

DEBUG = False


def plot_analysis():
    ns_rand = '../kuka_grasps/1BD/'
    ns_multi_no_qual = '../kuka_grasps/3BD/'
    rand = '../kuka_grasps/random/'
    map_elites = '../kuka_grasps/MAP_ELITES/'
    # folders = [ns_rand, ns_multi_no_qual]
    # labels = ['1 BD',
    #           '3 BD']

    folders = [ns_rand, ns_multi_no_qual, rand, map_elites]
    labels = ['1 BD',
              '3 BD',
              'random',
              'map_elites']

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 15))
    data_cov = []
    data_uni = []
    data_count = []
    for f in folders:
        runs = glob.glob(f + '*/run_details.json')
        list_cov = []
        list_uni = []
        count = 0
        for run in runs:
            with open(run) as json_file:
                details = json.load(json_file)
            if 'diversity coverage' in details:
                count += 1
                list_cov.append(details['diversity coverage'])
                list_uni.append(details['diversity uniformity'])
        data_cov.append(np.array(list_cov))
        data_uni.append(np.array(list_uni))
        data_count.append(count / len(runs))
    data_cov = np.transpose(np.array(data_cov))
    data_uni = np.transpose(np.array(data_uni))
    ax[0].boxplot(data_cov, labels=labels, showmeans=True, meanline=True)
    ax[0].set_title('Coverage')
    ax[0].tick_params(labelrotation=45)

    ax[1].boxplot(data_uni, labels=labels, showmeans=True, meanline=True)
    ax[1].set_title('Uniformity')
    ax[1].tick_params(labelrotation=45)

    ax[2].bar(list(range(len(data_count))), data_count, tick_label=labels)
    ax[2].set_title('Successful run frequency')
    ax[2].tick_params(labelrotation=45)

    plt.show()


def plot_launch(details, data):
    """Plotting

    Args:
        mean_hist (list): history of mean population fitness
        min_hist (list): history of min population fitness
        max_hist (list): history of max population fitness
        arch_size_hist (list): history of archive size
        coverage_hist (list): history of coverage of archive
        uniformity_hist (list): history of uniformity of archive
        mean_age_hist (list): history of mean age of population
        max_age_hist (list): history of max age of population
        run_name (String): path of the run folder to save the figure
        algo_type (String): name of the algo
        full_cov_hist (list): history of coverage of all generated individuals
        full_uni_hist (list): history of uniformity of all generated individuals
        pop_cov_hist (list): history of coverage of all generated individuals
        pop_uni_hist (list): history of uniformity of all generated individuals
        novelty_distrib (list): history of distributions of novelty across pop + offsprings
        qualities (list): history of qualities if measured


   """
    coverage_hist = data['archive coverage']
    uniformity_hist = data['archive uniformity']
    full_cov_hist = data['coverage']
    full_uni_hist = data['uniformity']
    mean_hist = data['mean fitness']
    min_hist = data['min fitness']
    max_hist = data['max fitness']
    arch_size_hist = data['archive size']
    mean_age_hist = data['mean age']
    max_age_hist = data['max_age_hist']
    pop_cov_hist = data['population coverage']
    pop_uni_hist = data['population uniformity']
    novelty_distrib = data['novelty distribution']
    algo_type = details['algo type']
    qualities = data['qualities']
    multi_qual = details['multi quality']

    mean_hist = np.array(mean_hist)
    min_hist = np.array(min_hist)

    # plot evolution
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(15, 10))
    ax[0][0].set(title='Evolution of fitness in population', ylabel='Fitness')
    ax[0][0].plot(mean_hist, label='Mean')
    ax[0][0].plot(min_hist, label='Min')
    ax[0][0].plot(max_hist, label='Max')
    ax[0][0].legend()

    # plot evolution
    ax[1][0].set(title='Evolution of age in population', ylabel='Age')
    ax[1][0].plot(mean_age_hist, label='Mean')
    ax[1][0].plot(max_age_hist, label='Max')
    ax[1][0].legend()

    # plot evolution
    ax[0][1].set(title='Evolution of archive size', ylabel='Archive size')
    ax[0][1].plot(arch_size_hist)

    # plot evolution
    ax[1][1].set(title='Evolution of selected metrics in archive')
    if algo_type == 'ns_rand_multi_bd':
        coverage_hist = np.array(coverage_hist)
        uniformity_hist = np.array(uniformity_hist)
        for i in range(np.size(coverage_hist, 1)):
            ax[1][1].plot(coverage_hist[:, i], color=utils.color_list[i], label='Coverage ' + str(i))
            ax[1][1].plot(uniformity_hist[:, i], ls='--', color=utils.color_list[i], label='Uniformity ' + str(i))

    else:
        ax[1][1].plot(coverage_hist, color='blue', label='Coverage')
        ax[1][1].plot(uniformity_hist, ls='--', color='blue', label='Uniformity')
    ax[1][1].legend()

    # plot evolution
    ax[2][1].set(title='Evolution of selected metrics in historic of all individuals', xlabel='Generations')
    if algo_type == 'ns_rand_multi_bd':
        full_cov_hist = np.array(full_cov_hist)
        full_uni_hist = np.array(full_uni_hist)
        for i in range(np.size(full_cov_hist, 1)):
            ax[2][1].plot(full_cov_hist[:, i], color=utils.color_list[i], label='Coverage ' + str(i))
            ax[2][1].plot(full_uni_hist[:, i], ls='--', color=utils.color_list[i], label='Uniformity ' + str(i))

    else:
        ax[2][1].plot(full_cov_hist, color='blue', label='Coverage')
        ax[2][1].plot(full_uni_hist, ls='--', color='blue', label='Uniformity')
    ax[2][1].legend()

    # plot evolution
    ax[2][0].set(title='Evolution of selected metrics in population', xlabel='Generations')
    if algo_type == 'ns_rand_multi_bd':
        pop_cov_hist = np.array(pop_cov_hist)
        pop_uni_hist = np.array(pop_uni_hist)
        for i in range(np.size(pop_cov_hist, 1)):
            ax[2][0].plot(pop_cov_hist[:, i], color=utils.color_list[i], label='Coverage ' + str(i))
            ax[2][0].plot(pop_uni_hist[:, i], ls='--', color=utils.color_list[i], label='Uniformity ' + str(i))

    else:
        ax[2][0].plot(pop_cov_hist, color='blue', label='Coverage')
        ax[2][0].plot(pop_uni_hist, ls='--', color='blue', label='Uniformity')
    ax[2][0].legend()

    fig_2 = 0
    if algo_type == 'ns_rand_multi_bd' or algo_type == 'map_elites':
        pass  # TODO: deal with multi_bd for novelty distrib plot
    else:
        if len(novelty_distrib) < 100:
            novelty_distrib = np.array(novelty_distrib)
            df = novelty_distrib.reshape((novelty_distrib.shape[0], novelty_distrib.shape[1]))
            df = df.transpose()
            df = pd.DataFrame(df, columns=list(range(df.shape[1])))
            fig_2, ax_2 = joypy.joyplot(df, ylabels=False, grid='y',
                                        title='Evolution of novelty distributions with respect to generations',
                                        legend=False, kind='counts', bins=30, ylim='max',
                                        figsize=(15, 15), color='red', linecolor='black')
    
    fig_3 = 0
    if multi_qual is not None:
        qualities = np.array(qualities)
        count = 0
        for i, qual in enumerate(multi_qual[0]):
            if qual is not None:
                count += 1
        fig_3, ax_3 = plt.subplots(nrows=count)
        count = 0
        for i, qual in enumerate(multi_qual[0]):
            if qual is not None:
                ax_3[count].plot(qualities[:, i], color=utils.color_list[i])
                ax_3[count].set(title='Evolution of mean ' + qual + ' in offsprings', xlabel='Generations')
                count += 1

    return fig, fig_2, fig_3


def collect_launchs(conditions, number, folder):

    exp_path = folder + '/*details.json'
    all_jsons = glob.glob(exp_path)
    count = 0
    valid_launches = []
    condition_keys = conditions.keys()
    for launch in all_jsons:
        cond = True
        with open(launch) as json_file:
            data = json.load(json_file)
        for cond_key in condition_keys:
            if data[cond_key] != conditions[cond_key]:
                if DEBUG:
                    print('required: ', conditions[cond_key])
                    print('data: ', data[cond_key])
                cond = False
                break
        if cond:
            count += 1
            valid_launches.append(launch)
        if count == number:
            break
    
    if len(valid_launches) != number:
        raise Exception('Not enough launches match your criteria')

    return valid_launches


def add_coverage_uniformity(data, df, legend):

    for launch in data:
        data_json_file = re.sub('details', 'data', launch)
        with open(data_json_file) as json_file:
            json_data = json.load(json_file)
        data = np.array([json_data['coverage'], json_data['uniformity']])
        data = np.transpose(data)
        df_temp = pd.DataFrame(data, columns=['coverage', 'uniformity'])
        df_temp['legend'] = data.shape[0] * [legend]
        df = df.append(df_temp)
    
    return df


def plot_end_cov_box(df, gen, colors, savepath):
    df_end = df[df['generation'] == gen - 1]
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.boxplot(x='legend', y='coverage', data=df_end, ax=ax, palette=colors)
    ax.set_facecolor("#ffebb8")
    fig.suptitle('Final coverages')
    if savepath is not None:
        fig.savefig(savepath + '_final')


def plot_experiment(temoin_dict, variation_key, variation_colors,
                    variation_possibilities, title, n_required, folder, savepath=None):
    fig, ax = plt.subplots(2, 1, figsize=(20, 15))
    df = pd.DataFrame(columns=['coverage', 'uniformity', 'legend'])

    # temoin
    details = collect_launchs(temoin_dict, n_required[0], folder)
    df = add_coverage_uniformity(details, df, temoin_dict[variation_key])

    # variations
    for i, var in enumerate(variation_possibilities):
        temoin_dict[variation_key] = var
        details = collect_launchs(temoin_dict, n_required[i + 1], folder)
        df = add_coverage_uniformity(details, df, temoin_dict[variation_key])

    df['generation'] = df.index
    variation_colors.insert(0, 'grey')
    sns.lineplot(data=df, x='generation', y='coverage', hue='legend', ax=ax[0], palette=variation_colors)
    sns.lineplot(data=df, x='generation', y='uniformity', hue='legend', ax=ax[1], palette=variation_colors)

    ax[0].set_facecolor("#ffebb8")
    ax[0].legend(loc=4)
    ax[1].set_facecolor("#ffebb8")
    ax[1].legend(loc=3)
    fig.suptitle(title)

    if savepath is not None:
        fig.savefig(savepath)
    

def plot_archive_management(env, arch_size, pop, gen, nb_cells, n_required, folder, savepath=None):
    size = 15

    plt.rc('font', size=size, weight='bold')          # controls default text sizes
    plt.rc('axes', titlesize=size, titleweight='bold')     # fontsize of the axes title
    plt.rc('axes', labelsize=size, labelweight='bold')    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=size)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=size)    # fontsize of the tick labels
    plt.rc('legend', fontsize=size)    # legend fontsize
    plt.rc('figure', titlesize=size, titleweight='bold')  # fontsize of the figure title

    if savepath:
        param_dict = {'pop_size': pop, 'gen_nb': gen, 'env': env, 
                      'n_cells': nb_cells, 'arch_size': arch_size,
                      'n_rep': n_required}
        with open(savepath + '_params.json', 'w') as fp:
            json.dump(param_dict, fp)
    
    fig, ax = plt.subplots(2, 1, figsize=(20, 15))
    df = pd.DataFrame(columns=['coverage', 'uniformity', 'legend'])

    # define experimental conditions
    temoin_dict = {
        'algo type': 'ns_rand',
        'evaluation function': env,
        'nb of generations': gen,
        'pop size': pop,
        'nb of cells': nb_cells,
        'altered novelty': False,
        'archive limit size': None
    }

    # temoin
    data = collect_launchs(temoin_dict, n_required, folder)
    df = add_coverage_uniformity(data, df, 'no limit')

    variation_possibilities = ['random', 'least_novel', 'oldest', 'grid', 'grid_density', 'gmm', 'newest',
                               'least_novel_iter', 'most_novel']
    variation_colors = utils.color_list[:9]

    # variations
    temoin_dict['archive limit size'] = arch_size
    variation_key = 'archive limit strat'
    for i, var in enumerate(variation_possibilities):
        temoin_dict[variation_key] = var
        data = collect_launchs(temoin_dict, n_required, folder)
        df = add_coverage_uniformity(data, df, temoin_dict[variation_key])

    df['generation'] = df.index
    variation_colors.insert(0, 'grey')
    sns.lineplot(data=df, x='generation', y='coverage', hue='legend', ax=ax[0], palette=variation_colors)
    sns.lineplot(data=df, x='generation', y='uniformity', hue='legend', ax=ax[1], palette=variation_colors)

    ax[0].set_facecolor("#ffebb8")
    ax[0].legend(loc=4)
    ax[1].set_facecolor("#ffebb8")
    ax[1].legend(loc=3)
    fig.suptitle('Archive management strategies in ' + env)

    if savepath is not None:
        fig.savefig(savepath + '_evo')

    plot_end_cov_box(df, gen, variation_colors, savepath)

    plt.show()


def plot_archive_importance(env, pop, gen, nb_cells, n_required, folder, savepath=None):
    size = 15

    plt.rc('font', size=size, weight='bold')          # controls default text sizes
    plt.rc('axes', titlesize=size, titleweight='bold')     # fontsize of the axes title
    plt.rc('axes', labelsize=size, labelweight='bold')    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=size)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=size)    # fontsize of the tick labels
    plt.rc('legend', fontsize=size)    # legend fontsize
    plt.rc('figure', titlesize=size, titleweight='bold')  # fontsize of the figure title

    if savepath:
        param_dict = {'pop_size': pop, 'gen_nb': gen, 'env': env, 'n_cells': nb_cells, 'n_rep': n_required}
        with open(savepath + '_params.json', 'w') as fp:
            json.dump(param_dict, fp)

    # define experimental conditions
    temoin_dict = {
        'algo type': 'ns_rand',
        'evaluation function': env,
        'nb of generations': gen,
        'pop size': pop,
        'nb of cells': nb_cells,
        'altered novelty': False,
        'archive limit size': None
    }
    variation_key = 'algo type'
    variation_possibilities = ['ns_no_archive', 'random_search']
    variation_colors = ['green', 'orange']
    n_required = [n_required, n_required, n_required]
    title = 'Archive importance in ' + env
    folder = 'results'

    fig, ax = plt.subplots(2, 1, figsize=(20, 15))
    df = pd.DataFrame(columns=['coverage', 'uniformity', 'legend'])

    # temoin
    details = collect_launchs(temoin_dict, n_required[0], folder)
    df = add_coverage_uniformity(details, df, temoin_dict[variation_key])

    # variations
    for i, var in enumerate(variation_possibilities):
        temoin_dict[variation_key] = var
        details = collect_launchs(temoin_dict, n_required[i + 1], folder)
        df = add_coverage_uniformity(details, df, temoin_dict[variation_key])

    df['generation'] = df.index
    variation_colors.insert(0, 'grey')
    sns.lineplot(data=df, x='generation', y='coverage', hue='legend', ax=ax[0], palette=variation_colors)
    sns.lineplot(data=df, x='generation', y='uniformity', hue='legend', ax=ax[1], palette=variation_colors)

    ax[0].set_facecolor("#ffebb8")
    ax[0].legend(loc=4)
    ax[1].set_facecolor("#ffebb8")
    ax[1].legend(loc=3)
    fig.suptitle(title)

    if savepath is not None:
        fig.savefig(savepath)
    
    plot_end_cov_box(df, temoin_dict['nb of generations'], variation_colors, savepath)

    plt.show()


def prepare_and_plot_exp():
    size = 15

    plt.rc('font', size=size, weight='bold')          # controls default text sizes
    plt.rc('axes', titlesize=size, titleweight='bold')     # fontsize of the axes title
    plt.rc('axes', labelsize=size, labelweight='bold')    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=size)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=size)    # fontsize of the tick labels
    plt.rc('legend', fontsize=size)    # legend fontsize
    plt.rc('figure', titlesize=size, titleweight='bold')  # fontsize of the figure title

    # define experimental conditions
    temoin_dict = {
        'algo type': 'ns_rand',
        'evaluation function': 'evaluate_maze',
        'nb of generations': 200,
        'pop size': 10,
        'nb of cells': 100,
        'altered novelty': False,
        'archive limit size': None
    }
    variation_key = 'algo type'
    variation_possibilities = ['ns_no_archive', 'random_search']
    variation_colors = ['green', 'orange']
    n_required = [5, 5, 5]
    title = 'Archive importance in deceptive maze'
    exp_folder = 'results'
    
    plot_experiment(temoin_dict, variation_key, variation_colors,
                    variation_possibilities, title, n_required, exp_folder)


def print_details(folder):
    exp_path = folder + '/*details.json'
    all_jsons = glob.glob(exp_path)
    with open(all_jsons[0]) as json_file:
        data = json.load(json_file)
    data['bd bounds'] = None
    df = pd.DataFrame(list(data.items())).transpose()
    df = df.rename(columns=df.iloc[0])
    df = df.drop(0)
    all_jsons.pop(0)
    for launch in all_jsons:
        with open(launch) as json_file:
            data = json.load(json_file)
        data['bd bounds'] = None
        df_temp = pd.DataFrame(list(data.items())).transpose()
        df_temp = df_temp.rename(columns=df_temp.iloc[0])
        df_temp = df_temp.drop(0)
        df = df.append(df_temp)
    for col in df.columns:
        uniques = df[col].unique()
        print('')
        print('Column: ', str(col))
        for unique in uniques:
            print(unique, ': ', (df[col] == unique).sum())


if __name__ == "__main__":
    # folder = 'results'
    # print_details(folder)
    # plot_archive_importance('evaluate_maze', 10, 200, 100, 5, folder)

    plot_analysis()
