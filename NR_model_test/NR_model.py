import jax.numpy as jnp
import jax
from jax import vmap, lax, jit
from NR_model_test.distillation_types import State, NR_State, Trays, Tray, Thermo
from NR_model_test import functions, initial_composition, matrix_transforms, jacobian, thermodynamics, costing
import os

def initialize():
    n_max = 100
    c_max = 8
    dir = os.path.join(os.getcwd(), 'Pure component parameters')
    #psat_params, cpvap_params, hvap_params, hform_params = thermodynamics.retrieve_params(dir)
    return State(
        L=jnp.zeros(n_max, dtype=float),
        V=jnp.zeros(n_max, dtype=float),
        U=jnp.zeros(n_max, dtype=float),
        W=jnp.zeros(n_max, dtype=float),
        X=jnp.zeros((c_max, n_max), dtype=float),
        Y=jnp.zeros((c_max, n_max), dtype=float),
        temperature=jnp.zeros(n_max, dtype=float),
        F=jnp.zeros(n_max, dtype=float),
        components=jnp.zeros(c_max, dtype=int),
        pressure=jnp.zeros((), dtype=float),
        z=jnp.zeros(c_max, dtype=float),
        RR=jnp.zeros((), dtype=float),
        distillate=jnp.zeros((), dtype=float),
        Nstages=jnp.zeros((), dtype=int),
        Hliq=jnp.zeros(n_max, dtype=float),
        Hvap=jnp.zeros(n_max, dtype=float),
        Hfeed=jnp.zeros(n_max, dtype=float),
        CD=jnp.zeros((), dtype=float),
        RD=jnp.zeros((), dtype=float),
        TAC=jnp.zeros((), dtype=float),
        trays=Trays(
            low_tray=Tray(
                l=jnp.zeros((c_max, n_max)),
                v=jnp.zeros((c_max, n_max)),
                T=jnp.zeros(n_max)
            ),
            high_tray=Tray(
                l=jnp.zeros((c_max, n_max)),
                v=jnp.zeros((c_max, n_max)),
                T=jnp.zeros(n_max)
            ),
            tray=Tray(
                l=jnp.zeros((c_max, n_max)),
                v=jnp.zeros((c_max, n_max)),
                T=jnp.zeros(n_max)
            )
        ),
        
        step_count=jnp.zeros((), dtype=int),  # ()
        action_mask=jnp.ones(7500, dtype=bool), # (4,)
        key=jax.random.PRNGKey(0),  # (2,)
    )
    '''thermo=Thermo(psat_params=psat_params, 
                      hvap_params=hvap_params,
                      hform_params=hform_params,
                      cpvap_params=cpvap_params,
                      ),'''


def initial_guess(state: State, nstages, feedstage, pressure, feed, z, distillate, rr):
    l_init = jnp.where(jnp.arange(len(state.L)) < feedstage-1, rr*distillate, rr*distillate+jnp.sum(feed))
    l_init = l_init.at[nstages-1].set((jnp.sum(feed)-distillate))
    v_init = jnp.where(jnp.arange(len(l_init)) > 0, (rr + jnp.ones_like(l_init))*distillate, distillate)
    f = jnp.where(jnp.arange(len(l_init)) == feedstage-1, feed, 0)
    mask = jnp.where(jnp.arange(len(state.temperature)) < nstages, 1, 0)
    return state.replace(
        L=l_init*mask,
        V=v_init*mask,
        z=z,
        RR=rr,
        distillate=distillate,
        pressure=pressure,
        F=f,
        Nstages=nstages,
        components=jnp.arange(len(z))  #jnp.array([3, 4, 7], dtype=int),  #
    )


def initial_temperature(state: State):
    t_range = functions.t_sat_solver(state.components, state.pressure)
    t_range = jnp.where(state.components > 0, t_range, jnp.max(t_range)-1)

    delta_t = (jnp.max(t_range)-jnp.min(t_range))/state.Nstages
    return state.replace(
        temperature=jnp.where(jnp.arange(len(state.temperature)) < state.Nstages, jnp.min(t_range)+jnp.arange(len(state.temperature))*delta_t, 0),
        )



def f_sol(state: State, tray_low, tray, tray_high, j):
    f_s = jacobian.f_vector_function(
        state,
        Tray(l=tray_low.l[:, j], v=tray_low.v[:, j], T=tray_low.T[j]),
        Tray(l=tray.l[:, j], v=tray.v[:, j], T=tray.T[j]),
        Tray(l=tray_high.l[:, j], v=tray_high.v[:, j], T=tray_high.T[j]),
        j,
    )
    return jnp.concatenate((jnp.asarray(f_s.H)[None], jnp.asarray(f_s.M), jnp.asarray(f_s.E)), axis=0)


def update_NR(state: State):
    a, b, c = jacobian.jacobian_func(state)
    f = vmap(f_sol, in_axes=(None, None, None, None, 0))(state, state.trays.low_tray, state.trays.tray,
                                                           state.trays.high_tray, jnp.arange(len(state.trays.tray.T)))

    dx = jnp.nan_to_num(functions.thomas(a, b, c, -1 * f, state.Nstages)) #.reshape(-1,1)

    def min_res(t, state: State, dx):
        '''
        hopper = len(state.components) * 2 + 1
        dx_v = vmap(lambda j: dx[jnp.arange(len(state.components)) + j * hopper, 1])(
            jnp.arange(len(state.temperature))).transpose()
        dx_l = vmap(lambda j: dx[jnp.arange(len(state.components)) + (len(state.components) + 1) + j * hopper, 1])(
            jnp.arange(len(state.temperature))).transpose()
        dx_t = vmap(lambda j: dx[len(state.components) + j * hopper, 1])(
            jnp.arange(len(state.temperature))).transpose()
        '''
        dx_v = dx[:, :len(state.components)].transpose()
        dx_l = dx[:, -len(state.components):].transpose()
        dx_t = dx[:, len(state.components)].transpose()
        '''
        dx_v = dx[:, :len(state.components)].transpose()
        dx_l = dx[:, -len(state.components):].transpose()
        dx_t = dx[:, len(state.components)].transpose()
        '''
        v_new = (state.trays.tray.v + t * dx_v)
        l_new = (state.trays.tray.l + t * dx_l)
        temp = (state.trays.tray.T + t * dx_t)
        # temp = jnp.where(dx[:, len(state.components)].transpose() < 10., jnp.where(dx[:, len(state.components)].transpose() > -10., state.trays.tray.T + t * dx[:, len(state.components)].transpose(), state.trays.tray.T - 10.), state.trays.tray.T + 10.)
        v_new_final = jnp.where(v_new > 0, v_new, state.trays.tray.v
                                * jnp.exp(t * dx_v / jnp.where(state.trays.tray.v > 0, state.trays.tray.v, 1e30)))
        l_new_final = jnp.where(l_new > 0, l_new, state.trays.tray.l
                                * jnp.exp(t * dx_l / jnp.where(state.trays.tray.l > 0, state.trays.tray.l, 1e30)))
        temp_final = jnp.where(jnp.abs(dx_t) > 7, (state.trays.tray.T + 7 * (dx_t) / (jnp.abs(dx_t))),
                               (state.trays.tray.T + t * dx_t))
        state = state.replace(
            V=jnp.sum(v_new_final, axis=0),
            L=jnp.sum(l_new_final, axis=0),
            Y=jnp.nan_to_num(v_new_final/jnp.sum(v_new_final, axis=0)),
            X=jnp.nan_to_num(l_new_final/jnp.sum(l_new_final, axis=0)),
            temperature=temp,
        )

        state = matrix_transforms.trays_func(state)
        f = vmap(f_sol, in_axes=(None, None, None, None, 0))(state, state.trays.low_tray, state.trays.tray, state.trays.high_tray, jnp.arange(len(state.trays.tray.T)))

        return jnp.sum(f**2), state

    carry = vmap(min_res, in_axes=(0, None, None))(jnp.arange(0.2, 1.3, 0.001), state, dx)
    result, states = carry
    new_t = jnp.max(jnp.where(result == jnp.min(result), jnp.arange(0.2, 1.3, 0.001), 0))

    res, state_new = min_res(new_t, state, dx)

    zeros = None
    dx_final = None

    return res, state_new, zeros, dx_final


def cond_fn(args):
    state, iterations, res = args
    comps = jnp.sum(jnp.where(state.z > 0, 1, 0))
    cond = state.Nstages*(2*comps+1)*jnp.sum(state.F)*1e-9
    return (iterations < 100) & (res > cond)


def body_fn(args):
    state, iterations, res = args
    res_new, nr_state_new, new_t, dx = update_NR(state)
    iterations += 1
    return nr_state_new, iterations, res_new

def store_variable(small_array, larger_array, start_indices):
    # Assuming small_array has shape (2, 20) and larger_array has shape (9000, 20)
    updated_larger_array = lax.dynamic_update_slice(larger_array, small_array, start_indices)
    return updated_larger_array


def converge_column(state: State):
    #nr_state = initialize_NR(state)
    state = matrix_transforms.trays_func(state)
    iterations = 0
    res = 0

    state, iterations, res = (
        lax.while_loop(cond_fun=cond_fn, body_fun=body_fn,
                       init_val=(state,
                                 iterations,
                                 jnp.array(10, dtype=float),
                                 )
                       )
    )
    '''
    res_array = jnp.zeros(1000, dtype=float)
    f_damp = jnp.array(1., dtype=float)
    damping = jnp.ones(1000, dtype=float) * 0.01
    profiler = jnp.zeros((1000, 3, len(state.temperature)), dtype=float)
    
    for rang in range(4):
        res, state, new_t, f = update_NR(state)
    '''

    return state, iterations, res


def condensor_duty(state: State):
    result = jnp.sum(
        state.V[1] * state.Y[:, 1] * thermodynamics.vapor_enthalpy(state.temperature[1], state.components)) + jnp.sum(
        state.F[0] * state.z * state.Hfeed[0]) - jnp.sum(
        (state.L[0]) * state.X[:, 0] * thermodynamics.liquid_enthalpy(state.temperature[0],
                                                                                     state.components)) - jnp.sum(
        state.V[0] * state.Y[:, 0] * thermodynamics.vapor_enthalpy(state.temperature[0], state.components))

    return state.replace(CD=result)


def reboiler_duty(state: State):
    result = jnp.sum(state.F*state.Hfeed) - state.CD - state.L[state.Nstages - 1] * jnp.sum(
        state.X[:, state.Nstages - 1] * thermodynamics.liquid_enthalpy(state.temperature[state.Nstages - 1],
                                                                       state.components)) - state.V[0] * jnp.sum(
        state.Y[:, 0] * thermodynamics.vapor_enthalpy(state.temperature[0], state.components))
    CD = jnp.sum(vmap(thermodynamics.h_evap, in_axes=(None, 0))(state.temperature[0], state.components)*state.Y[:,0])*state.V[0]
    return state.replace(RD=result,
                          CD=state.CD+CD)


def inside_simulation(state, nstages, feedstage, pressure, feed, z, distillate, rr):
    iterations = 0
    res = 0
    #feedstage = jnp.floor((nstages + 1) / 2 )
    #state = initialize()

    state = initial_guess(state=state, nstages=nstages, feedstage=feedstage, pressure=pressure, feed=feed, z=z,
                               distillate=distillate, rr=rr)

    state = initial_temperature(state)
    
    state = state.replace(Hfeed=jnp.where(state.F > 0, jnp.sum(thermodynamics.feed_enthalpy(state)*state.z), 0))

    #state = state.replace(X=(jnp.ones(len(state.temperature))[:, None]*state.z).transpose())
    #state = functions.y_func(state)
    
    def for_body(state, i):
        return initial_composition.bubble_point(state), None
    
    state, _ = jax.lax.scan(for_body, state, jnp.arange(3))

    
    state, iterations, res = converge_column(state)
    state = state.replace(
        Hliq=jnp.sum(vmap(thermodynamics.liquid_enthalpy, in_axes=(0, None))(state.temperature,
                                                                               state.components).transpose() * state.X,
                     axis=0),
        Hvap=jnp.sum(vmap(thermodynamics.vapor_enthalpy, in_axes=(0, None))(state.temperature,
                                                                              state.components).transpose() * state.Y,
                     axis=0)
    )
    
    state = condensor_duty(state)
    state = reboiler_duty(state)
    #state = costing.tac(state)
    #state = condensor_duty(state)
    #state = reboiler_duty(state)

    #feed_stage = jnp.max(jnp.where(state.F > 0, jnp.arange(len(state.temperature)), 0))
    #state, iteration = converge_bubble_point(state)


    return state, iterations, res

