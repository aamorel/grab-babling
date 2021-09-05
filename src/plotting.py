import pandas as pd
import numpy as np
import utils
import matplotlib.pyplot as plt
#import joypy
import glob
import json
import seaborn as sns
from matplotlib import cm
import re
import os

DEBUG = False
JOYPLOT = False


def half_sphere_projection(radius=1, num=10):
    linspace = np.linspace(-radius, radius, num=num)
    points = np.array(np.meshgrid(linspace, linspace)).T.reshape(-1, 2)
    points = points[np.linalg.norm(points, axis=1) <= radius]
    temp = np.sqrt(np.maximum(radius * radius - points[:, 0] * points[:, 0] - points[:, 1] * points[:, 1], 0))
    return np.hstack([points, temp[:, None]])


def plot_heat_circle(num, positions, values, fig=None, ax=None):
    map_m = np.zeros((num, num))
    map_m[:] = np.nan
    radius = np.linalg.norm(positions, axis=1).mean()
    indexes = np.round((positions[:, :2] / radius + 1) * (num - 1) / 2, 0).astype(int).T
    map_m[tuple(np.split(indexes, 2))] = values

    if ax is None or fig is None:
        fig, ax = plt.subplots(1, 1)
    heatmap = ax.imshow(np.flip(map_m.transpose(), 0), cmap='viridis', extent=[-radius, radius, -radius, radius])
    fig.colorbar(heatmap)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Success with obstacle")
    fig.savefig("halfSphereHeatMap.pdf")
    plt.show()

    return map_m


def plot_analysis():
    plt.rc('axes', titlesize=19, titleweight='bold')     # fontsize of the axes title
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.rc('axes', labelsize=16, labelweight='bold')    # fontsize of the x and y labels

    # exp = 'kuka_1000_gen_cube'
    # # labels = ['random', 'map_elites', 'concat_3BD', 'concat_4BD', '3BD', '4BD']
    # # colors = ['#686868', '#0000FF', '#008000', '#D94D1A', '#426A8C', '#73020C']
    # labels = ['random', 'map_elites', 'concat_4BD', '4BD']
    # colors = ['#686868', '#0000FF', '#D94D1A', '#73020C']
    # thresh = -0.08
    # detail_key_verif = {'robot': 'kuka',
    #                     'object': 'cube', 'nb of generations': 1000}

    exp = 'baxter_300_gen_cylinder/600'
    labels = ['concat_3BD', 'concat_4BD', '3BD', '4BD']
    colors = ['#008000', '#D94D1A', '#426A8C', '#73020C']
    # labels = ['concat_4BD', '4BD']
    # colors = ['#D94D1A', '#73020C']
    thresh = -0.16
    detail_key_verif = {}

    color_dicts = []
    for color in colors:
        color_dicts.append({'color': color})

    line_dicts = []
    for color in colors:
        line_dicts.append({'color': '#000000'})

    box_dicts = []
    for color in colors:
        box_dicts.append({'color': '#000000', 'facecolor': color})

    f_dicts = []
    for color in colors:
        f_dicts.append({'markerfacecolor': color, 'marker': 's'})

    folders = []
    for alg in labels:
        folders.append(os.path.join('..', 'experiments_analysis', exp, alg))

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 15))
    data_cov = []
    data_uni = []
    data_first = []
    data_count = []
    for f in folders:
        print(f)
        runs = glob.glob(f + '/*/run_details.json')
        list_cov = []
        list_uni = []
        list_first = []
        count = 0
        for run in runs:
            with open(run) as json_file:
                details = json.load(json_file)
            for key in detail_key_verif:
                assert(details[key] == detail_key_verif[key])
            if 'diversity coverage' in details:
                count += 1
                if details['algo type'] == 'map_elites':
                    list_cov.append(details['diversity coverage'] * 5)
                else:
                    list_cov.append(details['diversity coverage'])
                list_uni.append(details['diversity uniformity'])
                data_path = run[:-12] + 'data.json'
                with open(data_path) as json_data:
                    data = json.load(json_data)
                if 'first saved ind gen' in data:
                    list_first.append(data['first saved ind gen'])
                else:
                    fitness_hist = np.array(data['max fitness'])
                    grasping_in_pop = fitness_hist > thresh
                    for i, val in enumerate(grasping_in_pop):
                        if val:
                            list_first.append(i)
                            break

        data_cov.append(np.array(list_cov))
        data_uni.append(np.array(list_uni))
        data_first.append(np.array(list_first))
        data_count.append(count / len(runs))
    data_cov = np.array(data_cov)
    data_uni = np.transpose(np.array(data_uni))
    data_first = np.array(data_first)
    for i in range(len(data_cov)):
        ax[0].boxplot(data_cov[i], positions=[i], labels=[labels[i]], showfliers=False,
                      showmeans=True, meanline=True, patch_artist=True,
                      widths=[0.6], boxprops=box_dicts[i], medianprops=line_dicts[i],
                      whiskerprops=color_dicts[i], capprops=color_dicts[i],
                      flierprops=f_dicts[i], meanprops=line_dicts[i])
    ax[0].set_title('Diversity')
    ax[0].set_ylabel('BD3 coverage')
    ax[0].tick_params(labelrotation=45)
    # Add major gridlines in the y-axis
    ax[0].grid(color='grey', axis='y', linestyle='-', linewidth=0.75, alpha=0.5)

    ax[1].bar(list(range(len(data_count))), data_count, tick_label=labels, color=colors)
    ax[1].set_title('Successful run frequency')
    ax[1].grid(color='grey', axis='y', linestyle='-', linewidth=0.75, alpha=0.5)
    ax[1].tick_params(labelrotation=45)

    for i in range(len(data_first)):
        ax[2].boxplot(data_first[i], positions=[i], labels=[labels[i]], showfliers=False,
                      showmeans=True, meanline=True, patch_artist=True,
                      widths=[0.6], boxprops=box_dicts[i], medianprops=line_dicts[i],
                      whiskerprops=color_dicts[i], capprops=color_dicts[i],
                      flierprops=f_dicts[i], meanprops=line_dicts[i])
    ax[2].set_title('First grasping individual')
    ax[2].set_ylabel('Number of generations')
    ax[2].tick_params(labelrotation=45)
    # Add major gridlines in the y-axis
    ax[2].grid(color='grey', axis='y', linestyle='-', linewidth=0.75, alpha=0.5)

    # ax[3].boxplot(data_uni, labels=labels, showmeans=True, meanline=True)
    # ax[3].set_title('Uniformity')
    # ax[3].tick_params(labelrotation=45)
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
        if len(novelty_distrib) < 100 and JOYPLOT:
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
        """
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
        """
        fig_3, ax_3 = plt.subplots(nrows=len(qualities))
        ax_3 = np.array(ax_3).reshape(len(qualities))
        for i, (quality_name, quality_mean) in enumerate(qualities.items()):
            ax_3[i].plot(quality_mean)#, color=utils.color_list[i])
            ax_3[i].set(title='Evolution of mean ' + quality_name + ' in offsprings', xlabel='Generations')

    fig_4 = 0
    if algo_type == 'ns_rand_multi_bd':
        fig_4, ax_4 = plt.subplots()
        rates = np.array(data['eligibility rates'])
        for i in range(np.size(rates, 1)):
            ax_4.plot(rates[:, i], color=utils.color_list[i], label='Eligibility rate ' + str(i))
        ax_4.set(title='Evolution of behavioral descriptors eligibility rates in population')
    return fig, fig_2, fig_3, fig_4


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

    # radius = 0.23
    # num = 20
    # points = half_sphere_projection(radius=radius, num=num)
    # values = np.load('results/obstacle_results.npy')
    # values = np.mean(values, 1)

    # fig_2 = plt.figure()
    # ax = fig_2.gca(projection='3d')

    # ax.scatter(points[:, 0], points[:, 1], values)
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('value')

    # plot_heat_circle(num=num, positions=points, values=values)

    # with open('../long_run_kuka_cup/run_data.json') as f:
    #     data = json.load(f)
    # with open('../long_run_kuka_cup/run_details.json') as f:
    #     details = json.load(f)

    # plot_launch(details, data)
    # plt.show()
