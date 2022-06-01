import numpy as np
import matplotlib.pyplot as plt
import keras_utils as ku
import os
import numpy as np
import configparser as cp
import itertools

metric_functions = {
    'Maximum': (lambda err: np.max(err, axis=1)),
    'Median': (lambda err: np.median(err, axis=1)),
    'Root Mean Squared': (lambda err: np.sqrt(np.mean(np.square(err), axis=1))),
    'Mean': (lambda err: np.mean(err, axis=1))
}

def generate_statistics(dataset_files, metrics):
    values = {name:[] for name in metrics}
    for dataset in dataset_files:
        config_errors = np.loadtxt(dataset)
        for m in metrics:
            values[m].append(metric_functions[m](config_errors))
    return values

def resolution_separate_directories(path_to_configs):
    '''
    Given a path containing config files, it seperates the files into 
    three resolutions.

    Parameters
    ----------
    path_to_configs : str; required; path to configs folder
        
    Returns
    ----------
    highres_directory: str; location containing config files
        for high resolution dataset
    midres_directory: str; location containing config files
        for medium resolution dataset
    lowres_directory: str; location containing config files
        for low resolution dataset
    
    '''
    list_of_directories = os.scandir(path_to_configs)
    directories_in_string = []
    for folder in list_of_directories:
        directories_in_string.append(folder.name)
    for folder_names in directories_in_string:
        if folder_names.startswith('16'):
            highres_directory = path_to_configs + folder_names + "/"
        elif folder_names.startswith('15'):
            midres_directory = path_to_configs + folder_names + "/"
        elif folder_names.startswith('14'):
            lowres_directory = path_to_configs + folder_names + "/"
    return highres_directory, midres_directory, lowres_directory

def latentspace_loss(conf_path):
    conf = cp.ConfigParser()
    conf.read(conf_path)
    resolution = conf['data']['intensity']
    config_names = [filename for filename in os.listdir(os.path.dirname(conf_path)) if filename.startswith(str(resolution))]
    if config_names:
        configs = []
        for i in config_names:
            conf = cp.ConfigParser()
            conf_= os.path.dirname(conf_path)+'/'+i
            conf.read(conf_)
            configs.append(conf)
        error = []    
        for j in configs:
            history = ku.loadHist(j['output']['historyFile'])
            error.append(history['loss'][-1])
    else:
        print('No config files found...')
        error = []

    return error

def knn_error_stats(latent_space, configs_path):
    path_to_project = os.path.normpath(os.getcwd())
    conf = cp.ConfigParser()
    conf.read(configs_path+str(latent_space)+'_1n0u-1n0vc.ini')
    path_to_errors = conf['output']['errorPath']
    path_to_save_figures = path_to_project + '/' + path_to_errors
    x = [x for x in range(2, 21)]
    error_stats = generate_statistics((path_to_project + '/' + path_to_errors + str(k) + ' Nearest_Neighbors/rf_test_errors.txt'
                                                for k in x), 
                                               ['Root Mean Squared', 'Median'])
    return error_stats, path_to_save_figures
    
    
def read_files(conf_path, latent_space, n_neighbors):
    conf = cp.ConfigParser()
    conf.read(conf_path)
    error_path = conf['output']['errorPath']
    test_error = np.loadtxt(error_path+str(n_neighbors)+" Nearest_Neighbors/rf_test_errors.txt")
    testing_errors = np.transpose(np.array(test_error))
    psi_test_error = np.loadtxt(error_path+str(n_neighbors)+" Nearest_Neighbors/psi_test_errors.txt")
    psi_testing_errors = np.transpose(np.array(psi_test_error))
    test_accuracies = np.loadtxt(error_path+str(n_neighbors)+" Nearest_Neighbors/rf_test_accuracies_conf.txt")
    test_type_accuracies = np.loadtxt(error_path+str(n_neighbors)+" Nearest_Neighbors/rf_test_accuracies_type.txt")
    return testing_errors[:,0], psi_testing_errors[:,0],test_accuracies, test_type_accuracies


def plot_distribution_2d(conf_path, subset=False):
    '''
    Given a txt file, it renders the 2D plot of a combination of 2 or 3 angles 
    present in the file. This helps visualize the uniformity of the dataset. 

    Parameters
    ----------
    path: str; required; path to the file containing the angles
    subset: boolean; reads from original data if False and subset file if True
        
    Returns
    ----------
    renders 1 or 3 plots depending on no. of angles
    '''
    conf = cp.ConfigParser()
    conf.read(conf_path)
    conf_data = conf['data']
    if subset == False:
        data_path = conf_data['euleranglefilepath']
        marker = '.'
    else:
        data_path = conf_data['subsetfilepath']
        marker = 'bo'
    intensity = conf_data['intensity'].split(';')
    if '1e16' in intensity:
        resolution = 'high'
    else:
        if '1e15' in intensity:
            resolution = "mid"
        else:
            resolution = "low"
    # loading text
    euler_lst = np.transpose(np.loadtxt(data_path, dtype=None, usecols=(0, 1, 2)))
    # set ranges for the three angles in orientation
    list_of_limits = [list(range(-180, 181, 40)), list(range(0, 181, 20)), list(range(0, 361, 40))]
    angle_combinations = list(itertools.combinations(np.array(range(0, len(euler_lst))), 2))
    angle_labels = ['Φ (Azimuth)', 'Θ (Altitude)', 'Ψ (Psi)']
    for ac in angle_combinations:
        plt.plot(euler_lst[ac[0]], euler_lst[ac[1]], marker)
        plt.xlabel(angle_labels[ac[0]])
        plt.ylabel(angle_labels[ac[1]])
        plt.title(f"Orientation - {angle_labels[ac[0]]}, {angle_labels[ac[1]]} - {resolution} resolution dataset")
        plt.xticks(list_of_limits[ac[0]])
        plt.yticks(list_of_limits[ac[1]])
        plt.show()


def plot_distribution_3d(conf_path, subset=False):
    conf = cp.ConfigParser()
    conf.read(conf_path)
    conf_data = conf['data']
    if subset == False:
        data_path = conf_data['euleranglefilepath']
    else:
        data_path = conf_data['subsetfilepath']
    intensity = conf_data['intensity'].split(';')
    if '1e16' in intensity:
        resolution = 'high'
    else:
        if '1e15' in intensity:
            resolution = "mid"
        else:
            resolution = "low"
    # loading text
    euler_lst = np.transpose(np.loadtxt(data_path, dtype=None, usecols=(0, 1, 2)))
    # set ranges for the three angles in orientation
    list_of_limits = [list(range(-180, 181, 40)), list(range(0, 181, 20)), list(range(0, 361, 40))]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(euler_lst[0], euler_lst[1], euler_lst[2])
    ax.set_title(f"Distribution of Φ (Azimuth), Θ (Altitude), and Ψ (Psi) angles - {resolution} resolution dataset")
    ax.set_xlabel("Φ (Azimuth)")
    ax.set_ylabel("Θ (Altitude)")
    ax.set_zlabel("Ψ (Psi)")
    fig.set_size_inches(10, 10)
    ax.set_xticks(list_of_limits[0])
    ax.set_yticks(list_of_limits[1])
    ax.set_zticks(list_of_limits[2])
    plt.show()
    

def latentspace_plot(configs_path):
    '''
    Given the path to configs directory, it renders the plot of 
    latentspace dimension on the x-axis and mean squared error on the y-axis.

    Parameters
    ----------
    configs_path: str; required; path to the configs directory
        
    Returns
    ----------
    renders three plot in a single graph of latent space vs 
    mean squared error for each latent space
    '''
    #latentspace_size = range(5,55,5)
    high, mid, low = resolution_separate_directories(configs_path)
    error_16 = latentspace_loss(high, latentspace_size)
    error_15 = latentspace_loss(mid, latentspace_size)
    error_14 = latentspace_loss(low, latentspace_size)
    
    plt.plot(latentspace_size, error_16, 'o-', label = '1e16')
    plt.plot(latentspace_size, error_15, 'o-', label = '1e15')
    plt.plot(latentspace_size, error_14, 'o-', label = '1e14')
    plt.xlabel('Latent Space Dimension')
    plt.ylabel('Mean Squared Error')
    #plt.ylim(0,0.00060)
    plt.title('Image Reconstruction Error')
    plt.legend()

    
def plot_knn_error(stats, comparison, intensity, ylim_top, path_to_figures):
    path_to_project = os.path.normpath(os.getcwd())
    x = [x for x in range(2, 21)]
    if comparison:
        plt.figure(figsize=(6, 5))
    else:
        plt.figure(figsize=(6,6))
    for (color, metric) in (('C0', 'Root Mean Squared'),
                            #('C1', 'Mean'),
                            ('C1', 'Median'),
                            #('C2', 'Maximum')
                            ):
        label = metric
        data = stats[metric]
        
        plt.boxplot(data, whis=[5, 95], sym='', positions=x,
                boxprops={'color':color}, whiskerprops={'color':color}, capprops={'color':color}, medianprops={'color':color})
        median = np.median(data, axis=1)
        plt.plot(x, median, color+'o-', label=label)
        #print(np.argmin(median))
        plt.axvline(x=x[np.argmin(median)], color=color, zorder=1, ls=':', label='_nolegend_')

    plt.ylabel("Error (Degrees)", fontsize=14)
    plt.xlabel("K", fontsize=14)
    plt.ylim(0, ylim_top)

    plt.legend()
    if comparison:
        plt.title('Error for K-Nearest Neighbors - '+intensity+' - '+protein, fontsize=14)
    else:
        plt.title('Error for K-Nearest Neighbors - '+intensity, fontsize=14)

    os.makedirs(path_to_figures+'visual/', exist_ok=True)
    plt.savefig(path_to_figures+'visual/KNN error vs K_box-error'+'-compare.png',
                dpi=300, transparent='True', bbox_inches='tight', pad_inches=0.05)


def knn_k_selection_depricated(configs_path, latentspace_size):
    '''
    This function plots the median and roots mean squared of 
    angle errors in degrees on the y-axis and the 'k' value of 
    the k-nearest neighbor for KNN regression on the x-axis.   

    Parameters
    ----------
    configs_path: str; required; path to the configs directory
    latentspace_size: int; required; the selected latentspace 
        dimension
        
    Returns
    ----------
    renders 3 plots (one for each resolution) of median and root 
        mean squared errors in degrees against each 'k' value
    '''
    high, mid, low = resolution_separate_directories(configs_path)
    error_stat_high, path_to_figs = knn_error_stats(latentspace_size, high)
    error_stat_mid, _ = knn_error_stats(latentspace_size, mid)
    error_stat_low, _ = knn_error_stats(latentspace_size, low)
    plot_knn_error(error_stat_high, 0, 'High', 180, path_to_figs)
    plot_knn_error(error_stat_mid, 0, 'Medium', 180, path_to_figs)
    plot_knn_error(error_stat_low, 0, 'Low', 180, path_to_figs)

def knn_k_selection(conf_path, k_max):
    all_configs = []
    # return all files as a list
    for file in os.listdir(os.path.dirname(conf_path)):
            # check all .ini files
        if file.endswith(".ini"):
            # print path name of selected files
            all_configs.append(file)
    for case in all_configs:
        conf = cp.ConfigParser()
        conf.read(os.path.dirname(conf_path)+'/'+case)
        error_path = conf['output']['errorPath']
        if os.path.exists(error_path + str(k_max) + ' Nearest_Neighbors/rf_test_errors.txt'):
            conf = cp.ConfigParser()
            conf.read(os.path.dirname(conf_path)+'/'+case)
            resolution = conf['data']['intensity']
            protein = conf['data']['type']
            intensity = conf['data']['intensity'].split(';')
            if '1e16' in intensity:
                resolution = 'high'
            else:
                if '1e15' in intensity:
                    resolution = "mid"
                else:
                    resolution = "low"
            k_values = [k_values for k_values in range(2, k_max+1)]
            error_stats = generate_statistics((error_path + str(k) + ' Nearest_Neighbors/rf_test_errors.txt' for k in k_values), ['Root Mean Squared', 'Median'])
            # getting the boundaries for plot
            for i in error_stats:
                mx = []
                mn = []
                for j in error_stats[i]:
                    mx.append(max(k for k in j))
                    mn.append(min(k for k in j))
            y_top = max(top for top in mx) + 10
            y_bottom = min(bottom for bottom in mn) - 10
            plt.figure(figsize=(6, 5))
            for (color, metric) in (('C0', 'Root Mean Squared'),
                                    #('C1', 'Mean'),
                                    ('C1', 'Median'),
                                    #('C2', 'Maximum')
                                    ):
                label = metric
                data = error_stats[metric]
                plt.boxplot(data, whis=[5, 95], sym='', positions=k_values, boxprops={'color':color}, whiskerprops={'color':color}, capprops={'color':color}, medianprops={'color':color})
                median = np.median(data, axis=1)
                plt.plot(k_values, median, color+'o-', label=label)
                plt.axvline(x=k_values[np.argmin(median)], color=color, zorder=1, ls=':', label='_nolegend_')
            plt.ylabel("Error (Degrees)", fontsize=14)
            plt.xlabel("K", fontsize=14)
            plt.ylim(y_bottom, y_top)
            plt.legend()
            plt.title('Error for K-Nearest Neighbors - '+resolution+' resolution - '+protein, fontsize=14)
            os.makedirs(error_path+'visual/', exist_ok=True)
            plt.savefig(error_path+'visual/KNN error vs K_box-error'+'-compare.png', dpi=300, transparent='True', bbox_inches='tight', pad_inches=0.05)
    

def box_plots(errors, y_min, y_max, step, name, y_label, title, xticks_labels):
    fig, ax = plt.subplots()
    ax.boxplot(errors, whis=[0, 95], sym='.', showfliers=False)
    ax.set_yticks(np.arange(y_min, y_max+1, step))
    ax.set_ylim(y_min, y_max)
    ax.set_title(title)
    ax.set_xlabel("Resolution")
    ax.set_ylabel(y_label)
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False)         # ticks along the top edge are off
    ax.set_xticklabels(xticks_labels)
    #fig.savefig("Figures/JLESC/" + name + ".png", dpi=300, transparent='True',
                        #bbox_inches='tight', pad_inches=0.05)
        

def validation_comparison_plots_deprecated(configs_path, latent_space, n_neighbors_high, n_neighbors_mid, n_neighbors_low):
    '''
    This function plots 3 plots containing 3 box plots, each containing 
    all 3 resolution.The 3 box plots render the Error Degree, Psi 
    Difference, and Confirmation Accuracy respectively, each with the 
    high, medium, and low resolutions.

    Parameters
    ----------
    configs_path: str; required; path to the configs directory
    latent_space: int; required; the selected latent space size
    n_neighbors_high: int; required; no. of neighbours for KNN 
        regression for high resolution dataset
    n_neighbors_mid: int; required; no. of neighbours for KNN 
        regression for medium resolution dataset
    n_neighbors_low: int; required; no. of neighbours for KNN 
        regression for low resolution dataset
        
    Returns
    ----------
    renders 3 plots (each containing 3 box plots for each resolution) 
        representing error degree, psi difference, and conformation 
        accuracy, respectively.
    '''
    high, mid, low = resolution_separate_directories(configs_path)
    errors_accuracies_high = read_files(high, latent_space, n_neighbors_high)
    errors_accuracies_mid = read_files(mid, latent_space, n_neighbors_mid)
    errors_accuracies_low = read_files(low, latent_space, n_neighbors_low)
    box_plots([errors_accuracies_high[0], errors_accuracies_mid[0], errors_accuracies_low[0]], 0, 180, 10, 'Error_degree_ef2', 'Error Degree', 'Error Degree for EF2', ['High', 'Medium', 'Low'])
    box_plots([errors_accuracies_high[1], errors_accuracies_mid[1], errors_accuracies_low[1]], 0, 120, 10, 'Psi_differece_ef2', 'Psi Difference', 'Psi Difference for EF2', ['High', 'Medium', 'Low'])
    box_plots([errors_accuracies_high[2], errors_accuracies_mid[2], errors_accuracies_low[2]], 0, 1.1, 0.1, 'Conformation_accuracy_ef2', 'Conformation Accuracy', 'Conformation Accuracy for EF2', ['High', 'Medium', 'Low'])
    
    
def validation_comparison_plots(conf_path, latent_space, k_nearest_neighbors):
    all_configs = []
    # return all files as a list
    for file in os.listdir(os.path.dirname(conf_path)):
        # check all .ini files
        if file.endswith(".ini"):
            # print path name of selected files
            all_configs.append(file)
    err_degree = []
    psi_diff = []
    conf_acc = []
    type_acc = []
    n = 0
    x_labels = []
    for case in all_configs:
        conf = cp.ConfigParser()
        conf.read(os.path.dirname(conf_path)+'/'+case)
        intensity = conf['data']['intensity'].split(';')
        if '1e16' in intensity:
            resolution = 'High'
        else:
            if '1e15' in intensity:
                resolution = "Mid"
            else:
                resolution = "Low"
        error_path = conf['output']['errorPath']
        if os.path.isdir(error_path):
            ed, pd, ca, ctype = read_files(os.path.dirname(conf_path)+'/'+case, latent_space, k_nearest_neighbors[n])
            err_degree.append(ed)
            psi_diff.append(pd)
            conf_acc.append(ca)
            type_acc.append(ctype)
            x_labels.append(resolution)
            n += 1
    protein_types = "-".join(conf['data']['type'].split(','))
    conformation_accuracy = []
    for i in conf_acc:
        conformation_accuracy.append(i*100)
    type_accuracy = []
    for i in type_acc:
        type_accuracy.append(i*100)
    box_plots(err_degree, 0, 180, 10, 'Error_degree_'+str(protein_types), 'Error between Φ (Azimuth) and Θ (Altitude)- degrees', 'Error Degree for '+str(protein_types), x_labels)
    plt.text(0.35, -40, "For "+str(x_labels[0])+" resolution dataset, error in degrees for 95% of the images is less than "+str("{:.2f}".format(np.percentile(err_degree[0], 95))+" degrees."), bbox=dict(facecolor='silver',
                    alpha=0.5), fontsize=12)
    box_plots(psi_diff, 0, 120, 10, 'Psi_differece_'+str(protein_types), 'Error in (Ψ) Psi value - degrees', 'Psi Difference for '+str(protein_types), x_labels)
    plt.text(0.35, -40, "For "+str(x_labels[0])+" resolution dataset, psi difference for 95% of the images is less than "+str("{:.2f}".format(np.percentile(psi_diff[0], 95))+" degrees."), bbox=dict(facecolor='silver',
                    alpha=0.5), fontsize=12)
    box_plots(conformation_accuracy, 0, 100, 10, 'Conformation_accuracy_'+str(protein_types), 'Conformation accuracy - %', 'Conformation Accuracy for '+str(protein_types), x_labels)
    mid_value = "{:.2f}".format((np.percentile(conformation_accuracy[0], 50)))
    plt.text(0.35, -40, "For "+str(x_labels[0])+" resolution dataset, the average conformation accuracy is "+str(mid_value)+" percent.", bbox=dict(facecolor='silver',
                    alpha=0.5), fontsize=12)
    box_plots(type_accuracy, 0, 110, 10, 'ProteinType_accuracy_'+str(protein_types), 'Protein Type accuracy - %', 'Protein Type Accuracy for '+str(protein_types), x_labels)
    mid_value = "{:.2f}".format((np.percentile(type_accuracy[0], 50)))
    plt.text(0.35, -40, "For "+str(x_labels[0])+" resolution dataset, the average protein type accuracy is "+str(mid_value)+" percent.", bbox=dict(facecolor='silver',
                    alpha=0.5), fontsize=12)
def conformation_bargraph(conf_path, subset=False):
    '''
    When provided a text file containing information about the dataset, the function takes the count of different conformations and displays a bar graph. The bargraph helps determine the distribution of each conformation types over the dataset.
    
    Parameters
    ----------
    data_file_path: str; required; path to the file that stores information about the dataset
    resolution: str; required; resolution of dataset for labeling
        
    Returns
    ----------
    plots a bar graph where each bar represents the count of each conformation type
    
    '''
    conf = cp.ConfigParser()
    conf.read(conf_path)
    conf_data = conf['data']
    if subset == False:
        data_path = conf_data['euleranglefilepath']
    else:
        data_path = conf_data['subsetfilepath']
    intensity = conf_data['intensity'].split(';')
    if '1e16' in intensity:
        resolution = 'high'
    else:
        if '1e15' in intensity:
            resolution = "mid"
        else:
            resolution = "low"
    conformation_labels = np.array(np.loadtxt(data_path, dtype=str)[:, 3])
    # count unique labels of conformation
    conformations = np.unique(conformation_labels)
    
    conformation_counts = []
    # count no. of unique conformation types
    for i in range(len(conformations)):
        conformation_counts.append(np.count_nonzero(conformation_labels == str(conformations[i])))
    
    # plot bar graph
    fig = plt.figure()
    ax = fig.add_axes([0,0,1.3,1.3])
    ax.bar(conformations, conformation_counts, width=0.5, color=['slategrey', 'lightsteelblue', 'cornflowerblue', 'royalblue'])
    ax.set_ylabel('Quantity of Conformations')
    ax.set_title(f'Conformations - 1n0u, 1n0vc ({resolution} resolution dataset)')
    plt.show()

    
def protein_type_bargraph(conf_path, subset=False):
    '''
    When provided a text file containing information about the dataset, the function 
    takes the count of different protein types and displays a bar graph. The bargraph 
    helps determine the distribution of each protein type over the dataset.
    
    Parameters
    ----------
    data_file_path: str; required; path to the file that stores information about the dataset
    resolution: str; required; resolution of dataset for labeling
        
    Returns
    ----------
    plots a bar graph where each bar represents the count of each protein type
    
    '''
    conf = cp.ConfigParser()
    conf.read(conf_path)
    conf_data = conf['data']
    if subset == False:
        data_path = conf_data['euleranglefilepath']
    else:
        data_path = conf_data['subsetfilepath']
    intensity = conf_data['intensity'].split(';')
    if '1e16' in intensity:
        resolution = 'high'
    else:
        if '1e15' in intensity:
            resolution = "mid"
        else:
            resolution = "low"
    protein_type_labels = np.array(np.loadtxt(data_path, dtype=str)[:, 4])
    # count unique labels of proteins
    protein_types = conf['data']['type'].split(',')
    protein_counts = []
    # count no. of unique proteins
    for i in protein_types:
        protein_counts.append(np.count_nonzero(protein_type_labels == i))
    
    # plot bar graph
    plt.figure(figsize=(10, 7))
    plt.bar(protein_types, protein_counts, width=0.5, color=['khaki', 'darkkhaki'])
    plt.ylabel('Quantity of Protein Type/s')
    plt.title(f'Protein Type/s - Ef2 ({resolution} resolution dataset)')
    plt.show()
    
def conformation_confusion_matrix(conf_path, latent_space, k_nearest_neighbors, knn_trials):
    conf = cp.ConfigParser()
    conf.read(conf_path)
    conformation_confusion_matrix = conf['output']['errorpath']+str(k_nearest_neighbors[0])+" Nearest_Neighbors/visual_knn/conformations_cm_"+str(knn_trials-1)+".jpg"
    error_accuracies = read_files(conf_path, latent_space, k_nearest_neighbors[0])
    img = plt.imread(conformation_confusion_matrix)
    plt.figure(figsize = (15,10))
    #get current axes
    ax = plt.gca()
    #hide x-axis
    ax.set_axis_off()
    imgplot = plt.imshow(img)
    #plt.text(280, 700, 'fdg')
    mid_value = "{:.2f}".format((np.percentile(error_accuracies[2], 50)*100))
    plt.text(120, 1000, "The average conformation accuracy is "+str(mid_value)+" percent.", bbox=dict(facecolor='silver',
                   alpha=0.5), fontsize=12)
    

def protein_confusion_matrix(conf_path, latent_space, k_nearest_neighbors, knn_trials):
    conf = cp.ConfigParser()
    conf.read(conf_path)
    conformation_confusion_matrix = conf['output']['errorpath']+str(k_nearest_neighbors[0])+" Nearest_Neighbors/visual_knn/types_cm_"+str(knn_trials-1)+".jpg"
    img = plt.imread(conformation_confusion_matrix)
    plt.figure(figsize = (15,10))
    #get current axes
    ax = plt.gca()
    #hide x-axis
    ax.set_axis_off()
    imgplot = plt.imshow(img) 
    error_accuracies = read_files(conf_path, latent_space, k_nearest_neighbors[0])
    mid_value = "{:.2f}".format((np.percentile(error_accuracies[3], 50)*100))
    plt.text(120, 1000, "The average protein type accuracy is "+str(mid_value)+" percent.", bbox=dict(facecolor='silver',
                   alpha=0.5), fontsize=12)
    

    