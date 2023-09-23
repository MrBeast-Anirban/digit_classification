from utils import get_hyperparameter_combinations

def test_for_hparam_combinations_count():
    # test case to find all possible combinations of parametres indeed generated
    gamma_list = [0.001, 0.01, 0.1, 1]
    C_list = [1, 10, 100, 1000]
    h_params ={}
    h_params['gamma'] = gamma_list
    h_params['C'] = C_list
    h_params_combinations = get_hyperparameter_combinations(h_params)

    #assert len(h_param_combinations) == 16
    assert len(h_params_combinations) == len(gamma_list) * len(C_list)

def test_for_hparam_combinations_values():
    # test case to find all possible combinations of parametres indeed generated
    gamma_list = [0.001, 0.01]
    C_list = [1]
    h_params ={}
    h_params['gamma'] = gamma_list
    h_params['C'] = C_list
    h_params_combinations = get_hyperparameter_combinations(h_params)

    expected_param_combo_1 = {'gamma' : 0.001, 'C' : 1}
    expected_param_combo_2 = {'gamma' : 0.01, 'C' : 1}

    assert (expected_param_combo_1 in h_params_combinations) and (expected_param_combo_2 in h_params_combinations)