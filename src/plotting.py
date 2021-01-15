import pandas as pd
import numpy as np
import utils
import matplotlib.pyplot as plt
import joypy
import glob
import json
import seaborn as sns


def plot_launch(details):
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


   """
    coverage_hist = details['archive coverage']
    uniformity_hist = details['archive uniformity']
    full_cov_hist = details['coverage']
    full_uni_hist = details['uniformity']
    mean_hist = details['mean fitness']
    min_hist = details['min fitness']
    max_hist = details['max fitness']
    arch_size_hist = details['archive size']
    mean_age_hist = details['mean age']
    max_age_hist = details['max_age_hist']
    pop_cov_hist = details['population coverage']
    pop_uni_hist = details['population uniformity']
    novelty_distrib = details['novelty distribution']
    algo_type = details['algo type']

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
    if algo_type == 'ns_rand_multi_bd':
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

    return fig, fig_2


def collect_launchs(conditions, number, folder):

    exp_path = folder + '/*.json'
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
                cond = False
                break
        if cond:
            count += 1
            valid_launches.append(data)
    
    if len(valid_launches) != number:
        raise Exception('Not enough launches match your criteria')

    return valid_launches


def add_coverage_uniformity(data, df, legend):

    for launch in data:
        data = np.array([launch['coverage'], launch['uniformity']])
        data = np.transpose(data)
        df_temp = pd.DataFrame(data, columns=['coverage', 'uniformity'])
        df_temp['legend'] = data.shape[0] * [legend]
        df = df.append(df_temp)
    
    return df


def plot_experiment(temoin_dict, variation_key, variation_colors, 
                    variation_possibilities, title, n_required, folder):
    fig, ax = plt.subplots(2, 1, figsize=(20, 15))
    df = pd.DataFrame(columns=['coverage', 'uniformity', 'legend'])

    # temoin
    data = collect_launchs(temoin_dict, n_required[0], folder)
    df = add_coverage_uniformity(data, df, temoin_dict[variation_key])

    # variations
    for i, var in enumerate(variation_possibilities):
        temoin_dict[variation_key] = var
        data = collect_launchs(temoin_dict, n_required[i + 1], folder)
        df = add_coverage_uniformity(data, df, temoin_dict[variation_key])

    df['generation'] = df.index
    variation_colors.insert(0, 'grey')
    sns.lineplot(data=df, x='generation', y='coverage', hue='legend', ax=ax[0], palette=variation_colors)
    sns.lineplot(data=df, x='generation', y='uniformity', hue='legend', ax=ax[1], palette=variation_colors)

    ax[0].set_facecolor("#ffebb8")
    ax[0].legend(loc=4)
    ax[1].set_facecolor("#ffebb8")
    ax[1].legend(loc=3)
    fig.suptitle(title)
    plt.show()


if __name__ == "__main__":
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
        'nb of generations': 5,
        'pop size': 10,
        'nb of cells': 100,
        'altered novelty': False,
        'archive limit size': None
    }
    variation_key = 'algo type'
    variation_possibilities = ['ns_no_archive', 'random_search']
    variation_colors = ['green', 'orange']
    n_required = [2, 2, 2]
    title = 'Archive importance in deceptive maze'
    exp_folder = 'results'
    
    plot_experiment(temoin_dict, variation_key, variation_colors,
                    variation_possibilities, title, n_required, exp_folder)
