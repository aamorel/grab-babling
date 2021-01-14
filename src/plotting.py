import pandas
import numpy as np
import utils
import matplotlib.pyplot as plt
import joypy
import glob
import json


def gen_plot(mean_hist, min_hist, max_hist, arch_size_hist, coverage_hist, uniformity_hist,
             mean_age_hist, max_age_hist, run_name, algo_type, full_cov_hist, full_uni_hist,
             pop_cov_hist, pop_uni_hist, novelty_distrib):
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

    if run_name is not None:
        fig.savefig(run_name + 'novelty_search_plots.png')

    fig_2 = 0
    if algo_type == 'ns_rand_multi_bd':
        pass  # TODO: deal with multi_bd for novelty distrib plot
    else:
        if len(novelty_distrib) < 100:
            novelty_distrib = np.array(novelty_distrib)
            df = novelty_distrib.reshape((novelty_distrib.shape[0], novelty_distrib.shape[1]))
            df = df.transpose()
            df = pandas.DataFrame(df, columns=list(range(df.shape[1])))
            fig_2, ax_2 = joypy.joyplot(df, ylabels=False, grid='y',
                                        title='Evolution of novelty distributions with respect to generations',
                                        legend=False, kind='counts', bins=30, ylim='max',
                                        figsize=(15, 15), color='red', linecolor='black')

    return fig, fig_2


def collect_launchs(conditions, number):

    all_jsons = glob.glob('results/*.json')
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


if __name__ == "__main__":
    
    data = collect_launchs({'algo type': 'ns_rand'}, 2)
    print('hello')

    # MEDIUM_SIZE = 10
    # BIGGER_SIZE = 25

    # if SIMPLE_RUN:
    #     size = MEDIUM_SIZE
    # else:
    #     size = BIGGER_SIZE

    # plt.rc('font', size=size, weight='bold')          # controls default text sizes
    # plt.rc('axes', titlesize=size, titleweight='bold')     # fontsize of the axes title
    # plt.rc('axes', labelsize=size, labelweight='bold')    # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=size)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=size)    # fontsize of the tick labels
    # plt.rc('legend', fontsize=size)    # legend fontsize
    # plt.rc('figure', titlesize=size, titleweight='bold')  # fontsize of the figure title


    #     else:
    #         # ############################### ANALYSE IMPORTANCE OF ARCHIVE #####################################
    #         fig, ax = plt.subplots(2, 1, figsize=(20, 15))
    #         # adding a run for classic ns
    #         coverages = []
    #         uniformities = []
    #         for i in range(N_EXP):
    #             pop, archive, hof, info = noveltysearch.novelty_algo(EVALUATE_INDIVIDUAL, INITIAL_GENOTYPE_SIZE,
    #                                                                  BD_BOUNDS,
    #                                                                  mini=MINI, archive_limit_size=None,
    #                                                                  plot=PLOT, algo_type='ns_rand', nb_gen=GEN,
    #                                                                  parallelize=PARALLELIZE, bound_genotype=1,
    #                                                                  measures=True, pop_size=POP_SIZE,
    #                                                                  nb_cells=NB_CELLS)
    #             cov = np.array(info['coverage'])
    #             uni = np.array(info['uniformity'])
    #             coverages.append(cov)
    #             uniformities.append(uni)

    #         mean_cov = np.mean(coverages, 0)
    #         std_cov = [np.percentile(coverages, 25, 0), np.percentile(coverages, 75, 0)]
    #         sig_cov = [mean_cov - np.std(coverages, 0), mean_cov + np.std(coverages, 0)]
    #         mean_uni = np.mean(uniformities, 0)
    #         std_uni = [np.percentile(uniformities, 25, 0), np.percentile(uniformities, 75, 0)]
    #         sig_uni = [mean_uni - np.std(uniformities, 0), mean_uni + np.std(uniformities, 0)]

    #         ax[0].plot(mean_cov, label='classic ns', lw=2, color='grey')
    #         ax[0].fill_between(list(range(GEN)), sig_cov[0], sig_cov[1], facecolor='grey', alpha=0.5)
    #         ax[1].plot(mean_uni, label='classic ns', lw=2, color='grey')
    #         ax[1].fill_between(list(range(GEN)), sig_uni[0], sig_uni[1], facecolor='grey', alpha=0.5)

    #         # adding a run for no archive ns
    #         coverages = []
    #         uniformities = []
    #         for i in range(N_EXP):
    #             pop, archive, hof, info = noveltysearch.novelty_algo(EVALUATE_INDIVIDUAL, INITIAL_GENOTYPE_SIZE,
    #                                                                  BD_BOUNDS,
    #                                                                  mini=MINI, archive_limit_size=None, nb_gen=GEN,
    #                                                                  plot=PLOT, algo_type='ns_no_archive',
    #                                                                  parallelize=PARALLELIZE, bound_genotype=1,
    #                                                                  measures=True, pop_size=POP_SIZE,
    #                                                                  nb_cells=NB_CELLS, analyze_archive=False)
    #             cov = np.array(info['coverage'])
    #             uni = np.array(info['uniformity'])
    #             coverages.append(cov)
    #             uniformities.append(uni)

    #         mean_cov = np.mean(coverages, 0)
    #         std_cov = [np.percentile(coverages, 25, 0), np.percentile(coverages, 75, 0)]
    #         sig_cov = [mean_cov - np.std(coverages, 0), mean_cov + np.std(coverages, 0)]
    #         mean_uni = np.mean(uniformities, 0)
    #         std_uni = [np.percentile(uniformities, 25, 0), np.percentile(uniformities, 75, 0)]
    #         sig_uni = [mean_uni - np.std(uniformities, 0), mean_uni + np.std(uniformities, 0)]

    #         ax[0].plot(mean_cov, label='no archive', lw=2, color='green')
    #         ax[0].fill_between(list(range(GEN)), sig_cov[0], sig_cov[1], facecolor='green', alpha=0.5)
    #         ax[1].plot(mean_uni, label='no archive', lw=2, color='green')
    #         ax[1].fill_between(list(range(GEN)), sig_uni[0], sig_uni[1], facecolor='green', alpha=0.5)

    #         # adding a run for random search
    #         coverages = []
    #         uniformities = []
    #         for i in range(N_EXP):
    #             pop, archive, hof, info = noveltysearch.novelty_algo(EVALUATE_INDIVIDUAL, INITIAL_GENOTYPE_SIZE,
    #                                                                  BD_BOUNDS,
    #                                                                  mini=MINI, archive_limit_size=None, nb_gen=GEN,
    #                                                                  plot=PLOT, algo_type='random_search',
    #                                                                  parallelize=PARALLELIZE, bound_genotype=1,
    #                                                                  measures=True, pop_size=POP_SIZE,
    #                                                                  nb_cells=NB_CELLS, analyze_archive=False)
    #             cov = np.array(info['coverage'])
    #             uni = np.array(info['uniformity'])
    #             coverages.append(cov)
    #             uniformities.append(uni)

    #         mean_cov = np.mean(coverages, 0)
    #         std_cov = [np.percentile(coverages, 25, 0), np.percentile(coverages, 75, 0)]
    #         sig_cov = [mean_cov - np.std(coverages, 0), mean_cov + np.std(coverages, 0)]
    #         mean_uni = np.mean(uniformities, 0)
    #         std_uni = [np.percentile(uniformities, 25, 0), np.percentile(uniformities, 75, 0)]
    #         sig_uni = [mean_uni - np.std(uniformities, 0), mean_uni + np.std(uniformities, 0)]

    #         ax[0].plot(mean_cov, label='random search', lw=2, color='orange')
    #         ax[0].fill_between(list(range(GEN)), sig_cov[0], sig_cov[1], facecolor='orange', alpha=0.5)
    #         ax[1].plot(mean_uni, label='random search', lw=2, color='orange')
    #         ax[1].fill_between(list(range(GEN)), sig_uni[0], sig_uni[1], facecolor='orange', alpha=0.5)

    #         # generating the plot
    #         ax[0].set_xlabel("Generations", labelpad=15, color="#333533")
    #         ax[1].set_xlabel("Generations", labelpad=15, color="#333533")
    #         ax[0].set_ylabel("Mean coverage", labelpad=15, color="#333533")
    #         ax[0].set_facecolor("#ffebb8")
    #         ax[0].legend(loc=4)
    #         ax[1].set_facecolor("#ffebb8")
    #         ax[1].set_ylabel("Mean uniformity", labelpad=15, color="#333533")
    #         ax[1].legend(loc=2)

    #         fig.savefig('archive_importance_maze.png')
    #         if PLOT:
    #             plt.show()
    # else:
    #     if not NOVELTY_ANALYSIS:
    #         # ################################## ARCHIVE MANAGEMENT ANALYSIS ####################################
    #         possible_strats = ['random', 'least_novel', 'oldest', 'grid', 'grid_density', 'gmm', 'newest',
    #                            'least_novel_iter']
    #         colors = ['blue', 'red', 'yellow', 'green', 'pink', 'brown', 'purple', '#92F680']
    #         fig, ax = plt.subplots(3, 1, figsize=(20, 15))
    #         fig_2, ax_2 = plt.subplots(3, 1, figsize=(20, 15))

    #         for s, archive_strat in enumerate(possible_strats):
    #             coverages = []
    #             arch_coverages = []
    #             uniformities = []
    #             arch_uniformities = []
    #             rk_sim = []
    #             for i in range(N_EXP):
    #                 print('experience', i, 'of strat', archive_strat)
    #                 pop, archive, hof, info = noveltysearch.novelty_algo(EVALUATE_INDIVIDUAL, INITIAL_GENOTYPE_SIZE,
    #                                                                      BD_BOUNDS,
    #                                                                      mini=MINI, archive_limit_size=ARCHIVE_LIMIT,
    #                                                                      archive_limit_strat=archive_strat,
    #                                                                      plot=PLOT, algo_type=ALGO, nb_gen=GEN,
    #                                                                      parallelize=PARALLELIZE, bound_genotype=1,
    #                                                                      measures=True, pop_size=POP_SIZE,
    #                                                                      nb_cells=NB_CELLS, analyze_archive=True)
    #                 cov = np.array(info['coverage'])
    #                 uni = np.array(info['uniformity'])
    #                 arch_cov = np.array(info['archive coverage'])
    #                 arch_uni = np.array(info['archive uniformity'])
    #                 coverages.append(cov)
    #                 uniformities.append(uni)
    #                 arch_uniformities.append(arch_uni)
    #                 arch_coverages.append(arch_cov)

    #                 ranking_similarities = np.array(info['ranking similarities'])
    #                 rk_sim.append(ranking_similarities)
    #             mean_cov = np.mean(coverages, 0)
    #             std_cov = [np.percentile(coverages, 25, 0), np.percentile(coverages, 75, 0)]
    #             sig_cov = [mean_cov - np.std(coverages, 0), mean_cov + np.std(coverages, 0)]
    #             mean_uni = np.mean(uniformities, 0)
    #             std_uni = [np.percentile(uniformities, 25, 0), np.percentile(uniformities, 75, 0)]
    #             sig_uni = [mean_uni - np.std(uniformities, 0), mean_uni + np.std(uniformities, 0)]
    #             mean_arch_cov = np.mean(arch_coverages, 0)
    #             sig_arch_cov = [mean_arch_cov - np.std(arch_coverages, 0), mean_arch_cov + np.std(arch_coverages, 0)]
    #             std_arch_cov = [np.percentile(arch_coverages, 25, 0), np.percentile(arch_coverages, 75, 0)]
    #             mean_arch_uni = np.mean(arch_uniformities, 0)
    #             std_arch_uni = [np.percentile(arch_uniformities, 25, 0), np.percentile(arch_uniformities, 75, 0)]
    #             sig_arch_uni = [mean_arch_uni - np.std(arch_uniformities, 0),
    #                             mean_arch_uni + np.std(arch_uniformities, 0)]
    #             mean_rk_sim = np.mean(rk_sim, 0)

    #             ax[0].plot(mean_cov, label=archive_strat, lw=2, color=colors[s])
    #             ax[0].fill_between(list(range(GEN)), sig_cov[0], sig_cov[1], facecolor=colors[s], alpha=0.5)
    #             ax[1].plot(mean_uni, label=archive_strat, lw=2, color=colors[s])
    #             ax[1].fill_between(list(range(GEN)), sig_uni[0], sig_uni[1], facecolor=colors[s], alpha=0.5)
    #             ax[2].plot(mean_rk_sim, label=archive_strat, lw=2, color=colors[s])
    #             ax_2[0].plot(mean_arch_cov, label=archive_strat, lw=2, color=colors[s])
    #             ax_2[0].fill_between(list(range(GEN)), sig_arch_cov[0], sig_arch_cov[1], facecolor=colors[s], alpha=0.5)
    #             ax_2[1].plot(mean_arch_uni, label=archive_strat, lw=2, color=colors[s])
    #             ax_2[1].fill_between(list(range(GEN)), sig_arch_uni[0], sig_arch_uni[1], facecolor=colors[s], alpha=0.5)
    #             ax_2[2].plot(mean_rk_sim, label=archive_strat, lw=2, color=colors[s])
    #             # ax[2].fill_between(list(range(len(mean_rk_sim))), std_rk[0], std_rk[1],
    #             #                    facecolor=colors[s], alpha=0.5)
            
    #         # adding a run for classic ns
    #         coverages = []
    #         arch_coverages = []
    #         uniformities = []
    #         arch_uniformities = []
    #         rk_sim = []
    #         for i in range(N_EXP):
    #             pop, archive, hof, info = noveltysearch.novelty_algo(EVALUATE_INDIVIDUAL, INITIAL_GENOTYPE_SIZE,
    #                                                                  BD_BOUNDS,
    #                                                                  mini=MINI, archive_limit_size=None,
    #                                                                  plot=PLOT, algo_type=ALGO, nb_gen=GEN,
    #                                                                  parallelize=PARALLELIZE, bound_genotype=1,
    #                                                                  measures=True, pop_size=POP_SIZE,
    #                                                                  nb_cells=NB_CELLS, analyze_archive=False)
    #             cov = np.array(info['coverage'])
    #             uni = np.array(info['uniformity'])
    #             coverages.append(cov)
    #             uniformities.append(uni)
    #             arch_cov = np.array(info['archive coverage'])
    #             arch_uni = np.array(info['archive uniformity'])
    #             arch_coverages.append(arch_cov)
    #             arch_uniformities.append(arch_uni)

    #         mean_cov = np.mean(coverages, 0)
    #         std_cov = [np.percentile(coverages, 25, 0), np.percentile(coverages, 75, 0)]
    #         sig_cov = [mean_cov - np.std(coverages, 0), mean_cov + np.std(coverages, 0)]
    #         mean_uni = np.mean(uniformities, 0)
    #         std_uni = [np.percentile(uniformities, 25, 0), np.percentile(uniformities, 75, 0)]
    #         sig_uni = [mean_uni - np.std(uniformities, 0), mean_uni + np.std(uniformities, 0)]
    #         mean_arch_cov = np.mean(arch_coverages, 0)
    #         sig_arch_cov = [mean_arch_cov - np.std(arch_coverages, 0), mean_arch_cov + np.std(arch_coverages, 0)]
    #         std_arch_cov = [np.percentile(arch_coverages, 25, 0), np.percentile(arch_coverages, 75, 0)]
    #         mean_arch_uni = np.mean(arch_uniformities, 0)
    #         std_arch_uni = [np.percentile(arch_uniformities, 25, 0), np.percentile(arch_uniformities, 75, 0)]
    #         sig_arch_uni = [mean_arch_uni - np.std(arch_uniformities, 0),
    #                         mean_arch_uni + np.std(arch_uniformities, 0)]
    #         mean_rk_sim = np.mean(rk_sim, 0)

    #         ax[0].plot(mean_cov, label='no archive limit', lw=2, color='gray')
    #         ax[0].fill_between(list(range(GEN)), sig_cov[0], sig_cov[1], facecolor='gray', alpha=0.5)
    #         ax[1].plot(mean_uni, label='no archive limit', lw=2, color='gray')
    #         ax[1].fill_between(list(range(GEN)), sig_uni[0], sig_uni[1], facecolor='gray', alpha=0.5)
    #         ax_2[0].plot(mean_arch_cov, label='no archive limit', lw=2, color='gray')
    #         ax_2[0].fill_between(list(range(GEN)), sig_arch_cov[0], sig_arch_cov[1], facecolor='gray', alpha=0.5)
    #         ax_2[1].plot(mean_arch_uni, label='no archive limit', lw=2, color='gray')
    #         ax_2[1].fill_between(list(range(GEN)), sig_arch_uni[0], sig_arch_uni[1], facecolor='gray', alpha=0.5)

    #         # adding a run for random search
    #         coverages = []
    #         arch_coverages = []
    #         uniformities = []
    #         arch_uniformities = []
    #         rk_sim = []
    #         for i in range(N_EXP):
    #             pop, archive, hof, info = noveltysearch.novelty_algo(EVALUATE_INDIVIDUAL, INITIAL_GENOTYPE_SIZE,
    #                                                                  BD_BOUNDS,
    #                                                                  mini=MINI, archive_limit_size=None,
    #                                                                  plot=PLOT, algo_type='random_search', nb_gen=GEN,
    #                                                                  parallelize=PARALLELIZE, bound_genotype=1,
    #                                                                  measures=True, pop_size=POP_SIZE,
    #                                                                  nb_cells=NB_CELLS, analyze_archive=False)
    #             cov = np.array(info['coverage'])
    #             uni = np.array(info['uniformity'])
    #             coverages.append(cov)
    #             uniformities.append(uni)
    #             arch_cov = np.array(info['archive coverage'])
    #             arch_uni = np.array(info['archive uniformity'])
    #             arch_coverages.append(arch_cov)
    #             arch_uniformities.append(arch_uni)

    #         mean_cov = np.mean(coverages, 0)
    #         std_cov = [np.percentile(coverages, 25, 0), np.percentile(coverages, 75, 0)]
    #         sig_cov = [mean_cov - np.std(coverages, 0), mean_cov + np.std(coverages, 0)]
    #         mean_uni = np.mean(uniformities, 0)
    #         std_uni = [np.percentile(uniformities, 25, 0), np.percentile(uniformities, 75, 0)]
    #         sig_uni = [mean_uni - np.std(uniformities, 0), mean_uni + np.std(uniformities, 0)]
    #         mean_arch_cov = np.mean(arch_coverages, 0)
    #         sig_arch_cov = [mean_arch_cov - np.std(arch_coverages, 0), mean_arch_cov + np.std(arch_coverages, 0)]
    #         std_arch_cov = [np.percentile(arch_coverages, 25, 0), np.percentile(arch_coverages, 75, 0)]
    #         mean_arch_uni = np.mean(arch_uniformities, 0)
    #         std_arch_uni = [np.percentile(arch_uniformities, 25, 0), np.percentile(arch_uniformities, 75, 0)]
    #         sig_arch_uni = [mean_arch_uni - np.std(arch_uniformities, 0),
    #                         mean_arch_uni + np.std(arch_uniformities, 0)]
    #         mean_rk_sim = np.mean(rk_sim, 0)

    #         ax[0].plot(mean_cov, label='random search', lw=2, color='orange')
    #         ax[0].fill_between(list(range(GEN)), sig_cov[0], sig_cov[1], facecolor='orange', alpha=0.5)
    #         ax[1].plot(mean_uni, label='random search', lw=2, color='orange')
    #         ax[1].fill_between(list(range(GEN)), sig_uni[0], sig_uni[1], facecolor='orange', alpha=0.5)
    #         ax_2[0].plot(mean_arch_cov, label='random search', lw=2, color='orange')
    #         ax_2[0].fill_between(list(range(GEN)), sig_arch_cov[0], sig_arch_cov[1], facecolor='orange', alpha=0.5)
    #         ax_2[1].plot(mean_arch_uni, label='random search', lw=2, color='orange')
    #         ax_2[1].fill_between(list(range(GEN)), sig_arch_uni[0], sig_arch_uni[1], facecolor='orange', alpha=0.5)

    #         # adding a run for fitness ea
    #         coverages = []
    #         arch_coverages = []
    #         uniformities = []
    #         arch_uniformities = []
    #         rk_sim = []
    #         for i in range(N_EXP):
    #             pop, archive, hof, info = noveltysearch.novelty_algo(EVALUATE_INDIVIDUAL, INITIAL_GENOTYPE_SIZE,
    #                                                                  BD_BOUNDS,
    #                                                                  mini=MINI, archive_limit_size=None,
    #                                                                  plot=PLOT, algo_type='classic_ea', nb_gen=GEN,
    #                                                                  parallelize=PARALLELIZE, bound_genotype=1,
    #                                                                  measures=True, pop_size=POP_SIZE,
    #                                                                  nb_cells=NB_CELLS, analyze_archive=False)
    #             cov = np.array(info['coverage'])
    #             uni = np.array(info['uniformity'])
    #             coverages.append(cov)
    #             uniformities.append(uni)
    #             arch_cov = np.array(info['archive coverage'])
    #             arch_uni = np.array(info['archive uniformity'])
    #             arch_coverages.append(arch_cov)
    #             arch_uniformities.append(arch_uni)

    #         mean_cov = np.mean(coverages, 0)
    #         std_cov = [np.percentile(coverages, 25, 0), np.percentile(coverages, 75, 0)]
    #         sig_cov = [mean_cov - np.std(coverages, 0), mean_cov + np.std(coverages, 0)]
    #         mean_uni = np.mean(uniformities, 0)
    #         std_uni = [np.percentile(uniformities, 25, 0), np.percentile(uniformities, 75, 0)]
    #         sig_uni = [mean_uni - np.std(uniformities, 0), mean_uni + np.std(uniformities, 0)]
    #         mean_arch_cov = np.mean(arch_coverages, 0)
    #         sig_arch_cov = [mean_arch_cov - np.std(arch_coverages, 0), mean_arch_cov + np.std(arch_coverages, 0)]
    #         std_arch_cov = [np.percentile(arch_coverages, 25, 0), np.percentile(arch_coverages, 75, 0)]
    #         mean_arch_uni = np.mean(arch_uniformities, 0)
    #         std_arch_uni = [np.percentile(arch_uniformities, 25, 0), np.percentile(arch_uniformities, 75, 0)]
    #         sig_arch_uni = [mean_arch_uni - np.std(arch_uniformities, 0),
    #                         mean_arch_uni + np.std(arch_uniformities, 0)]
    #         mean_rk_sim = np.mean(rk_sim, 0)

    #         ax[0].plot(mean_cov, label='fitness ea', lw=2, color='cyan')
    #         ax[0].fill_between(list(range(GEN)), sig_cov[0], sig_cov[1], facecolor='cyan', alpha=0.5)
    #         ax[1].plot(mean_uni, label='fitness ea', lw=2, color='cyan')
    #         ax[1].fill_between(list(range(GEN)), sig_uni[0], sig_uni[1], facecolor='cyan', alpha=0.5)
    #         ax_2[0].plot(mean_arch_cov, label='fitness ea', lw=2, color='cyan')
    #         ax_2[0].fill_between(list(range(GEN)), sig_arch_cov[0], sig_arch_cov[1], facecolor='cyan', alpha=0.5)
    #         ax_2[1].plot(mean_arch_uni, label='fitness ea', lw=2, color='cyan')
    #         ax_2[1].fill_between(list(range(GEN)), sig_arch_uni[0], sig_arch_uni[1], facecolor='cyan', alpha=0.5)

    #         # generating the plot
    #         ax[0].set_xlabel("Generations", labelpad=15, color="#333533")
    #         ax[1].set_xlabel("Generations", labelpad=15, color="#333533")
    #         ax[2].set_xlabel("Iterations of reduction of archive", labelpad=15, color="#333533")
    #         ax[0].set_ylabel("Mean coverage", labelpad=15, color="#333533")
    #         ax[0].set_facecolor("#ffebb8")
    #         ax[0].legend(loc=4)
    #         ax[1].set_facecolor("#ffebb8")
    #         ax[1].set_ylabel("Mean uniformity", labelpad=15, color="#333533")
    #         ax[1].legend(loc=2)
    #         ax[2].set_ylabel("Mean Kendall Tau similarity", labelpad=15, color="#333533")
    #         ax[2].set_facecolor("#ffebb8")
    #         ax[2].legend(loc=4)

    #         ax_2[0].set_xlabel("Generations", labelpad=15, color="#333533")
    #         ax_2[1].set_xlabel("Generations", labelpad=15, color="#333533")
    #         ax_2[2].set_xlabel("Iterations of reduction of archive", labelpad=15, color="#333533")
    #         ax_2[0].set_ylabel("Mean archive coverage", labelpad=15, color="#333533")
    #         ax_2[0].set_facecolor("#ffebb8")
    #         ax_2[0].legend(loc=4)
    #         ax_2[1].set_facecolor("#ffebb8")
    #         ax_2[1].set_ylabel("Mean archive uniformity", labelpad=15, color="#333533")
    #         ax_2[1].legend(loc=2)
    #         ax_2[2].set_ylabel("Mean Kendall Tau similarity", labelpad=15, color="#333533")
    #         ax_2[2].set_facecolor("#ffebb8")
    #         ax_2[2].legend(loc=4)

    #         fig.savefig('full_analysis_maze.png')
    #         fig_2.savefig('archive_analysis_maze.png')
    #         if PLOT:
    #             plt.show()
    #     else:
    #         # ################################# ANALYSIS OF ALTERATION OF NOVELTY ######################################

    #         # looping through all degrees
    #         possible_degrees = [0.1, 0.5, 1, 5, 10, 20, 100]
    #         colors = ['blue', 'red', 'yellow', 'green', 'pink', 'brown', 'purple']
    #         fig, ax = plt.subplots(3, 1, figsize=(20, 15))
    #         fig_2, ax_2 = plt.subplots(3, 1, figsize=(20, 15))

    #         for s, archive_strat in enumerate(possible_degrees):
    #             coverages = []
    #             arch_coverages = []
    #             uniformities = []
    #             arch_uniformities = []
    #             rk_sim = []
    #             for i in range(N_EXP):
    #                 print('experience', i, 'of strat', archive_strat)
    #                 pop, archive, hof, info = noveltysearch.novelty_algo(EVALUATE_INDIVIDUAL, INITIAL_GENOTYPE_SIZE,
    #                                                                      BD_BOUNDS, altered_novelty=True,
    #                                                                      alteration_degree=archive_strat,
    #                                                                      mini=MINI,
    #                                                                      plot=PLOT, algo_type=ALGO, nb_gen=GEN,
    #                                                                      parallelize=PARALLELIZE, bound_genotype=1,
    #                                                                      measures=True, pop_size=POP_SIZE,
    #                                                                      nb_cells=NB_CELLS)
    #                 cov = np.array(info['coverage'])
    #                 uni = np.array(info['uniformity'])
    #                 arch_cov = np.array(info['archive coverage'])
    #                 arch_uni = np.array(info['archive uniformity'])
    #                 coverages.append(cov)
    #                 uniformities.append(uni)
    #                 arch_uniformities.append(arch_uni)
    #                 arch_coverages.append(arch_cov)

    #                 ranking_similarities = np.array(info['ranking similarities novelty'])
    #                 rk_sim.append(ranking_similarities)
    #             mean_cov = np.mean(coverages, 0)
    #             std_cov = [np.percentile(coverages, 25, 0), np.percentile(coverages, 75, 0)]
    #             sig_cov = [mean_cov - np.std(coverages, 0), mean_cov + np.std(coverages, 0)]
    #             mean_uni = np.mean(uniformities, 0)
    #             std_uni = [np.percentile(uniformities, 25, 0), np.percentile(uniformities, 75, 0)]
    #             sig_uni = [mean_uni - np.std(uniformities, 0), mean_uni + np.std(uniformities, 0)]
    #             mean_arch_cov = np.mean(arch_coverages, 0)
    #             sig_arch_cov = [mean_arch_cov - np.std(arch_coverages, 0), mean_arch_cov + np.std(arch_coverages, 0)]
    #             std_arch_cov = [np.percentile(arch_coverages, 25, 0), np.percentile(arch_coverages, 75, 0)]
    #             mean_arch_uni = np.mean(arch_uniformities, 0)
    #             std_arch_uni = [np.percentile(arch_uniformities, 25, 0), np.percentile(arch_uniformities, 75, 0)]
    #             sig_arch_uni = [mean_arch_uni - np.std(arch_uniformities, 0),
    #                             mean_arch_uni + np.std(arch_uniformities, 0)]
    #             mean_rk_sim = np.mean(rk_sim, 0)

    #             ax[0].plot(mean_cov, label=archive_strat, lw=2, color=colors[s])
    #             ax[0].fill_between(list(range(GEN)), sig_cov[0], sig_cov[1], facecolor=colors[s], alpha=0.5)
    #             ax[1].plot(mean_uni, label=archive_strat, lw=2, color=colors[s])
    #             ax[1].fill_between(list(range(GEN)), sig_uni[0], sig_uni[1], facecolor=colors[s], alpha=0.5)
    #             ax[2].plot(mean_rk_sim, label=archive_strat, lw=2, color=colors[s])
    #             ax_2[0].plot(mean_arch_cov, label=archive_strat, lw=2, color=colors[s])
    #             ax_2[0].fill_between(list(range(GEN)), sig_arch_cov[0], sig_arch_cov[1], facecolor=colors[s], alpha=0.5)
    #             ax_2[1].plot(mean_arch_uni, label=archive_strat, lw=2, color=colors[s])
    #             ax_2[1].fill_between(list(range(GEN)), sig_arch_uni[0], sig_arch_uni[1], facecolor=colors[s], alpha=0.5)
    #             ax_2[2].plot(mean_rk_sim, label=archive_strat, lw=2, color=colors[s])
    #             # ax[2].fill_between(list(range(len(mean_rk_sim))), std_rk[0], std_rk[1],
    #             #                    facecolor=colors[s], alpha=0.5)
            
    #         # adding a run for classic ns
    #         coverages = []
    #         arch_coverages = []
    #         uniformities = []
    #         arch_uniformities = []
    #         rk_sim = []
    #         for i in range(N_EXP):
    #             pop, archive, hof, info = noveltysearch.novelty_algo(EVALUATE_INDIVIDUAL, INITIAL_GENOTYPE_SIZE,
    #                                                                  BD_BOUNDS, altered_novelty=False,
    #                                                                  mini=MINI,
    #                                                                  plot=PLOT, algo_type=ALGO, nb_gen=GEN,
    #                                                                  parallelize=PARALLELIZE, bound_genotype=1,
    #                                                                  measures=True, pop_size=POP_SIZE,
    #                                                                  nb_cells=NB_CELLS)
    #             cov = np.array(info['coverage'])
    #             uni = np.array(info['uniformity'])
    #             coverages.append(cov)
    #             uniformities.append(uni)
    #             arch_cov = np.array(info['archive coverage'])
    #             arch_uni = np.array(info['archive uniformity'])
    #             arch_coverages.append(arch_cov)
    #             arch_uniformities.append(arch_uni)

    #         mean_cov = np.mean(coverages, 0)
    #         std_cov = [np.percentile(coverages, 25, 0), np.percentile(coverages, 75, 0)]
    #         sig_cov = [mean_cov - np.std(coverages, 0), mean_cov + np.std(coverages, 0)]
    #         mean_uni = np.mean(uniformities, 0)
    #         std_uni = [np.percentile(uniformities, 25, 0), np.percentile(uniformities, 75, 0)]
    #         sig_uni = [mean_uni - np.std(uniformities, 0), mean_uni + np.std(uniformities, 0)]
    #         mean_arch_cov = np.mean(arch_coverages, 0)
    #         sig_arch_cov = [mean_arch_cov - np.std(arch_coverages, 0), mean_arch_cov + np.std(arch_coverages, 0)]
    #         std_arch_cov = [np.percentile(arch_coverages, 25, 0), np.percentile(arch_coverages, 75, 0)]
    #         mean_arch_uni = np.mean(arch_uniformities, 0)
    #         std_arch_uni = [np.percentile(arch_uniformities, 25, 0), np.percentile(arch_uniformities, 75, 0)]
    #         sig_arch_uni = [mean_arch_uni - np.std(arch_uniformities, 0),
    #                         mean_arch_uni + np.std(arch_uniformities, 0)]
    #         mean_rk_sim = np.mean(rk_sim, 0)

    #         ax[0].plot(mean_cov, label='no alteration', lw=2, color='gray')
    #         ax[0].fill_between(list(range(GEN)), sig_cov[0], sig_cov[1], facecolor='gray', alpha=0.5)
    #         ax[1].plot(mean_uni, label='no alteration', lw=2, color='gray')
    #         ax[1].fill_between(list(range(GEN)), sig_uni[0], sig_uni[1], facecolor='gray', alpha=0.5)
    #         ax_2[0].plot(mean_arch_cov, label='no alteration', lw=2, color='gray')
    #         ax_2[0].fill_between(list(range(GEN)), sig_arch_cov[0], sig_arch_cov[1], facecolor='gray', alpha=0.5)
    #         ax_2[1].plot(mean_arch_uni, label='no alteration', lw=2, color='gray')
    #         ax_2[1].fill_between(list(range(GEN)), sig_arch_uni[0], sig_arch_uni[1], facecolor='gray', alpha=0.5)

    #         # adding a run for random search
    #         coverages = []
    #         arch_coverages = []
    #         uniformities = []
    #         arch_uniformities = []
    #         rk_sim = []
    #         for i in range(N_EXP):
    #             pop, archive, hof, info = noveltysearch.novelty_algo(EVALUATE_INDIVIDUAL, INITIAL_GENOTYPE_SIZE,
    #                                                                  BD_BOUNDS, altered_novelty=False,
    #                                                                  mini=MINI,
    #                                                                  plot=PLOT, algo_type='random_search', nb_gen=GEN,
    #                                                                  parallelize=PARALLELIZE, bound_genotype=1,
    #                                                                  measures=True, pop_size=POP_SIZE,
    #                                                                  nb_cells=NB_CELLS)
    #             cov = np.array(info['coverage'])
    #             uni = np.array(info['uniformity'])
    #             coverages.append(cov)
    #             uniformities.append(uni)
    #             arch_cov = np.array(info['archive coverage'])
    #             arch_uni = np.array(info['archive uniformity'])
    #             arch_coverages.append(arch_cov)
    #             arch_uniformities.append(arch_uni)

    #         mean_cov = np.mean(coverages, 0)
    #         std_cov = [np.percentile(coverages, 25, 0), np.percentile(coverages, 75, 0)]
    #         sig_cov = [mean_cov - np.std(coverages, 0), mean_cov + np.std(coverages, 0)]
    #         mean_uni = np.mean(uniformities, 0)
    #         std_uni = [np.percentile(uniformities, 25, 0), np.percentile(uniformities, 75, 0)]
    #         sig_uni = [mean_uni - np.std(uniformities, 0), mean_uni + np.std(uniformities, 0)]
    #         mean_arch_cov = np.mean(arch_coverages, 0)
    #         sig_arch_cov = [mean_arch_cov - np.std(arch_coverages, 0), mean_arch_cov + np.std(arch_coverages, 0)]
    #         std_arch_cov = [np.percentile(arch_coverages, 25, 0), np.percentile(arch_coverages, 75, 0)]
    #         mean_arch_uni = np.mean(arch_uniformities, 0)
    #         std_arch_uni = [np.percentile(arch_uniformities, 25, 0), np.percentile(arch_uniformities, 75, 0)]
    #         sig_arch_uni = [mean_arch_uni - np.std(arch_uniformities, 0),
    #                         mean_arch_uni + np.std(arch_uniformities, 0)]
    #         mean_rk_sim = np.mean(rk_sim, 0)

    #         ax[0].plot(mean_cov, label='random search', lw=2, color='orange')
    #         ax[0].fill_between(list(range(GEN)), sig_cov[0], sig_cov[1], facecolor='orange', alpha=0.5)
    #         ax[1].plot(mean_uni, label='random search', lw=2, color='orange')
    #         ax[1].fill_between(list(range(GEN)), sig_uni[0], sig_uni[1], facecolor='orange', alpha=0.5)
    #         ax_2[0].plot(mean_arch_cov, label='random search', lw=2, color='orange')
    #         ax_2[0].fill_between(list(range(GEN)), sig_arch_cov[0], sig_arch_cov[1], facecolor='orange', alpha=0.5)
    #         ax_2[1].plot(mean_arch_uni, label='random search', lw=2, color='orange')
    #         ax_2[1].fill_between(list(range(GEN)), sig_arch_uni[0], sig_arch_uni[1], facecolor='orange', alpha=0.5)

    #         # adding a run for fitness ea
    #         coverages = []
    #         arch_coverages = []
    #         uniformities = []
    #         arch_uniformities = []
    #         rk_sim = []
    #         for i in range(N_EXP):
    #             pop, archive, hof, info = noveltysearch.novelty_algo(EVALUATE_INDIVIDUAL, INITIAL_GENOTYPE_SIZE,
    #                                                                  BD_BOUNDS, altered_novelty=False,
    #                                                                  mini=MINI,
    #                                                                  plot=PLOT, algo_type='classic_ea', nb_gen=GEN,
    #                                                                  parallelize=PARALLELIZE, bound_genotype=1,
    #                                                                  measures=True, pop_size=POP_SIZE,
    #                                                                  nb_cells=NB_CELLS)
    #             cov = np.array(info['coverage'])
    #             uni = np.array(info['uniformity'])
    #             coverages.append(cov)
    #             uniformities.append(uni)
    #             arch_cov = np.array(info['archive coverage'])
    #             arch_uni = np.array(info['archive uniformity'])
    #             arch_coverages.append(arch_cov)
    #             arch_uniformities.append(arch_uni)

    #         mean_cov = np.mean(coverages, 0)
    #         std_cov = [np.percentile(coverages, 25, 0), np.percentile(coverages, 75, 0)]
    #         sig_cov = [mean_cov - np.std(coverages, 0), mean_cov + np.std(coverages, 0)]
    #         mean_uni = np.mean(uniformities, 0)
    #         std_uni = [np.percentile(uniformities, 25, 0), np.percentile(uniformities, 75, 0)]
    #         sig_uni = [mean_uni - np.std(uniformities, 0), mean_uni + np.std(uniformities, 0)]
    #         mean_arch_cov = np.mean(arch_coverages, 0)
    #         sig_arch_cov = [mean_arch_cov - np.std(arch_coverages, 0), mean_arch_cov + np.std(arch_coverages, 0)]
    #         std_arch_cov = [np.percentile(arch_coverages, 25, 0), np.percentile(arch_coverages, 75, 0)]
    #         mean_arch_uni = np.mean(arch_uniformities, 0)
    #         std_arch_uni = [np.percentile(arch_uniformities, 25, 0), np.percentile(arch_uniformities, 75, 0)]
    #         sig_arch_uni = [mean_arch_uni - np.std(arch_uniformities, 0),
    #                         mean_arch_uni + np.std(arch_uniformities, 0)]
    #         mean_rk_sim = np.mean(rk_sim, 0)

    #         ax[0].plot(mean_cov, label='fitness ea', lw=2, color='cyan')
    #         ax[0].fill_between(list(range(GEN)), sig_cov[0], sig_cov[1], facecolor='cyan', alpha=0.5)
    #         ax[1].plot(mean_uni, label='fitness ea', lw=2, color='cyan')
    #         ax[1].fill_between(list(range(GEN)), sig_uni[0], sig_uni[1], facecolor='cyan', alpha=0.5)
    #         ax_2[0].plot(mean_arch_cov, label='fitness ea', lw=2, color='cyan')
    #         ax_2[0].fill_between(list(range(GEN)), sig_arch_cov[0], sig_arch_cov[1], facecolor='cyan', alpha=0.5)
    #         ax_2[1].plot(mean_arch_uni, label='fitness ea', lw=2, color='cyan')
    #         ax_2[1].fill_between(list(range(GEN)), sig_arch_uni[0], sig_arch_uni[1], facecolor='cyan', alpha=0.5)
            
    #         ax[0].set_xlabel("Generations", labelpad=15, color="#333533")
    #         ax[1].set_xlabel("Generations", labelpad=15, color="#333533")
    #         ax[2].set_xlabel("Iterations of novelty computation", labelpad=15, color="#333533")
    #         ax[0].set_ylabel("Mean coverage", labelpad=15, color="#333533")
    #         ax[0].set_facecolor("#ffebb8")
    #         ax[0].legend(loc=4)
    #         ax[1].set_facecolor("#ffebb8")
    #         ax[1].set_ylabel("Mean uniformity", labelpad=15, color="#333533")
    #         ax[1].legend(loc=2)
    #         ax[2].set_ylabel("Mean Kendall Tau similarity", labelpad=15, color="#333533")
    #         ax[2].set_facecolor("#ffebb8")
    #         ax[2].legend(loc=4)

    #         # generating the plot
    #         ax_2[0].set_xlabel("Generations", labelpad=15, color="#333533")
    #         ax_2[1].set_xlabel("Generations", labelpad=15, color="#333533")
    #         ax_2[2].set_xlabel("Iterations of novelty computation", labelpad=15, color="#333533")
    #         ax_2[0].set_ylabel("Mean archive coverage", labelpad=15, color="#333533")
    #         ax_2[0].set_facecolor("#ffebb8")
    #         ax_2[0].legend(loc=4)
    #         ax_2[1].set_facecolor("#ffebb8")
    #         ax_2[1].set_ylabel("Mean archive uniformity", labelpad=15, color="#333533")
    #         ax_2[1].legend(loc=2)
    #         ax_2[2].set_ylabel("Mean Kendall Tau similarity", labelpad=15, color="#333533")
    #         ax_2[2].set_facecolor("#ffebb8")
    #         ax_2[2].legend(loc=4)

    #         fig.savefig('full_analysis_novelty.png')
    #         fig_2.savefig('archive_analysis_novelty.png')
    #         if PLOT:
    #             plt.show()