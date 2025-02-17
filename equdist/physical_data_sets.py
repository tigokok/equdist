import pickle
import jax.numpy as jnp

physical_data = {
'n-propane': {
        'cpig': [14.2051, 30.2403, 844.31, 20.5802, 2482.7, 298.15, 1500],
        'dhv': [6.97645, 0.78237, -0.77319, 0.39246, 0, -187.68, 96.68],
        'psat': [47.5651, -3492.6, 0, 0, -6.0669, 1.09E-05, 2, 85.47, 369.83],
        'hform': -25.0024 * 4.184,
        'mw': 44.09652,
        'density': 582.16062595505
    },
'iso-butane': {
        'cpig': [18.2464, 40.1309, 826.54, 24.5653, 2483.1, 298.15, 1500],
        'dhv': [9.4712, 1.274, -1.4255, 0.60708, 0, -159.61, 134.65],
        'psat': [96.9171, -5039.9, 0, 0, -15.012, 0.022725, 1, 113.54, 407.8],
        'hform': -32.2418 * 4.184,	
        'mw': 58.1234,
        'density': 595.443221427136
        },
'n-butane': {
        'cpig': [19.1445, 38.7934, 841.49, 25.258, 2476.1, 298.15, 1500],
        'dhv': [8.6553, 0.8337, -0.82274, 0.39613, 0, -138.29, 151.97],
        'psat': [54.8301, -4363.2, 0, 0, -7.046, 9.45E-06, 2, 134.86, 425.12],
        'hform': -30.0444 * 4.184,
        'mw': 58.1234,
        'density': 603.035386342418
        },
'n-pentane': {
        'cpig': [21.0304, 71.9165, 1650.2, 45.1896, 747.6, 200, 1500],
        'dhv': [10.7688, 0.95886, -0.92384, 0.39393, 0, -129.73, 196.55],
        'psat': [67.2281, -5420.3, 0, 0, -8.8253, 9.62E-06, 2, 143.42, 469.7],
        'hform': -35.053 * 4.184,
        'mw': 72.15028,
        'density': 611.258995343012
        },
'cyclo-pentane': {
        'cpig': [9.93599, 71.9882, 1461.7, 43.2192, 668.8, 100, 1500],
        'dhv': [8.17235, -0.21723, 1.0245, -0.49752, 0, -93.87, 238.55],
        'psat': [54.8281, -5198.5, 0, 0, -6.8103, 6.19E-06, 2, 179.28, 511.7],
        'hform': -18.3983 * 4.184,
        'mw': 70.1344,
        'density': 718.953321022845
        },
'n-hexane': {
        'cpig': [24.9355, 84.1454, 1694.6, 56.5826, 761.6, 200, 1500],
        'dhv': [10.4729, 0.34057, 0.063282, -0.017037, 0, -95.32, 234.45],
        'psat': [93.1371, -6995.5, 0, 0, -12.702, 1.24E-05, 2, 177.83, 507.6],
        'hform': -39.8729 * 4.184,
        'mw': 86.17716,
        'density': 615.501010415606
        },
'trimethyl-butane': {
        'cpig': [22.762, 114.742, 1573, 69.5997, 664.3, 200, 1500],
        'dhv': [11.5062, 0.75694, -0.61674, 0.26462, 0, -24.58, 257.95],
        'psat': [67.8701, -6127.1, 0, 0, -8.7696, 7.36E-06, 2, 248.57, 531.1],
        'hform': -48.8273 * 4.184,
        'mw': 100.20404,
        'density': 639.116574979533
        },
'n-heptane': {
        'cpig': [28.6973, 95.5622, 1676.6, 65.4438, 756.4, 200, 1500],
        'dhv': [12.5432, 0.51283, -0.10982, -0.01018, 0, -90.58, 267.05],
        'psat': [76.3161, -6996.4, 0, 0, -9.8802, 7.21E-06, 2, 182.57, 540.2],
        'hform': -44.8194 * 4.184,
        'mw': 100.20404,
        'density': 614.978050422831
        },
'dimethyl-hexane': {
        'cpig': [25.6043, 119.733, 1541.5, 78.6997, 660, 200, 1500],
        'dhv': [16.0989, 1.3555, -1.4474, 0.55232, 0, -91.15, 276.85],
        'psat': [85.5431, -7517.2, 0, 0, -11.282, 8.34E-06, 2, 182, 550],
        'hform': -53.1456 * 4.184,
        'mw': 114.23092,
        'density': 615.56023902613
        },
'n-octane': {
        'cpig': [32.3732, 105.833, 1635.6, 72.9435, 746.4, 200, 1500],
        'dhv': [16.0356, 1.0769, -1.0124, 0.37075, 0, -56.77, 295.55],
        'psat': [84.5711, -7900.2, 0, 0, -11.003, 7.18E-06, 2, 216.38, 568.7],
        'hform': -49.8591 * 4.184,
        'mw': 114.23092,
        'density': 613.127610604072
        },
'water': {
        'cpig': [7.968615649, 6.398681571, 2610.5, 2.124773096, 1169, -173.15, 2000],
        'dhv': [13.51867775, 0.61204, -0.6257, 0.3988, 0, 0.01, 373.95],
        'psat': [62.13607454, -7258.2, 0, 0, -7.3037, 4.1653E-06, 2, 0.01, 373.95],
        'hform': -57.7572 * 4.184,
        'mw': 18.0153,
        'density': 997.0
        },
'ethanol': {
        'cpig': [11.0484, 11.1136, 2694.3, 2.8784, 1115.8, 200, 2000],
        'dhv': [13.7742, 0.625, -0.625, 0.3988, 0, -277.69, 277.69],
        'psat': [61.3861, -6253.3, 0, 0, -7.3037, 4.1653E-06, 2, 78.37, 351.63],
        'hform': -277.69 * 4.184,
        'mw': 46.06844,
        'density': 789.0
        },
}

def make_component_list(components):
        selected_cpig = [physical_data[component]['cpig'] for component in components]
        selected_dhv = [physical_data[component]['dhv'] for component in components]
        selected_psat = [physical_data[component]['psat'] for component in components]
        selected_hform = [physical_data[component]['hform'] for component in components]
        selected_mw = [physical_data[component]['mw'] for component in components]
        selected_density = [physical_data[component]['density'] for component in components]

        CPIG = jnp.array(selected_cpig, dtype=float)
        DHV = jnp.array(selected_dhv, dtype=float)
        PSAT = jnp.array(selected_psat, dtype=float)
        HFORM = jnp.array(selected_hform, dtype=float)
        MW = jnp.array(selected_mw, dtype=float)
        DENSITY = jnp.array(selected_density, dtype=float)

        return CPIG, DHV, PSAT, HFORM, MW, DENSITY

def save_component_list(components, filename='component_list'):
        CPIG, DHV, PSAT, HFORM, MW, DENSITY = make_component_list(components)
        data_dict = {
                'CPIG': CPIG,
                'DHV': DHV,
                'PSAT': PSAT,
                'HFORM': HFORM,
                'MW': MW,
                'DENSITY': DENSITY
        }
        with open(f'{filename}.pkl', 'wb') as f:
                pickle.dump(data_dict, f)
        print('saved')

def load_component_list(filename='component_list'):
        with open(f'{filename}.pkl', 'rb') as f:
                data = pickle.load(f)
        CPIG = jnp.array(data['CPIG'])
        DHV = jnp.array(data['DHV'])
        PSAT = jnp.array(data['PSAT'])
        HFORM = jnp.array(data['HFORM'])
        MW = jnp.array(data['MW'])
        DENSITY = jnp.array(data['DENSITY'])
        return CPIG, DHV, PSAT, HFORM, MW, DENSITY
