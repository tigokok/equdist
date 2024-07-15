from typing import NamedTuple
import jax.numpy as jnp
from Distillation.Newton_Raphson_directory.distillation_types import State, NR_State, Trays, Tray, Mesh
from jax import vmap, jacfwd, jit, lax
from Distillation.Newton_Raphson_directory import functions, initial_composition, energy_functions, costing
from Distillation.distillation_model import simulation
from Distillation.Newton_Raphson_directory.plot_generation import plot_function
import matplotlib.pyplot as plt
from time import time
from Distillation.Newton_Raphson_directory.Analyticals import set_B, set_A, set_C
from Distillation.Bubble_point_model.bubble_point import simulation as bubble_simulation
from jaxopt import LBFGSB as solver
from matplotlib.animation import FuncAnimation
params_psat, params_Cpvap, params_Hvap, Hform = functions.retrieve_params()


def initialize():
    n_max =30
    c_max = 3
    return State(
        L=jnp.zeros(n_max, dtype=float),
        V=jnp.zeros(n_max, dtype=float),
        U=jnp.zeros(n_max, dtype=float),
        W=jnp.zeros(n_max, dtype=float),
        X=jnp.zeros((c_max,n_max), dtype=float),
        Y=jnp.zeros((c_max,n_max), dtype=float),
        temperature=jnp.zeros(n_max, dtype=float),
        F=jnp.zeros((c_max,n_max), dtype=float),
        components=jnp.zeros(c_max, dtype=int),
        pressure=jnp.zeros(1, dtype=float),
        z=jnp.zeros(c_max, dtype=float),
        RR=jnp.zeros(1, dtype=float),
        distillate=jnp.zeros(1, dtype=float),
        Nstages=jnp.zeros(1, dtype=int),
        Hliq=jnp.zeros(n_max, dtype=float),
        Hvap=jnp.zeros(n_max, dtype=float),
        Hfeed=jnp.zeros(1, dtype=int),
        CD=jnp.zeros(1, dtype=float),
        RD=jnp.zeros(1, dtype=float),
        TAC=jnp.zeros(1, dtype=float),
        storage=jnp.zeros((100, n_max)),
        res=jnp.zeros(1000, dtype=float),
        profiler=jnp.zeros((1000, c_max, n_max), dtype=float),
        damping=jnp.zeros(1000, dtype=float),
    )


def initial_guess(state: State, nstages, feedstage, pressure, feed, z, distillate, rr):
    l = jnp.where(jnp.arange(len(state.L)) < feedstage-1, rr*distillate, rr*distillate+jnp.sum(feed))
    l = l.at[nstages-1].set((jnp.sum(feed)-distillate))
    v = jnp.where(jnp.arange(len(l)) > 0, (rr + jnp.ones_like(l))*distillate, distillate)
    f = jnp.where(jnp.arange(len(l)) == feedstage-1, feed, 0)
    return state._replace(
        L=l,
        V=v,
        #U=state.U.at[0].set(distillate),
        z=z,
        RR=rr,
        distillate=distillate,
        pressure=pressure,
        F=f,
        Nstages=nstages,
        components=jnp.array([3,4,7]), #jnp.where(z > 0, jnp.arange(len(z)), 0)
        #jnp.array([3, 4, 7], dtype=int) #
    )


def initial_temperature(state: State):

    t_range = functions.t_sat_solver(state.components, state.pressure)
    t_range = jnp.where(state.components > 0, t_range, jnp.max(t_range)-1)

    delta_t = (jnp.max(t_range)-jnp.min(t_range))/state.Nstages
    return state._replace(
        temperature=jnp.where(jnp.arange(len(state.temperature)) < state.Nstages, jnp.min(t_range)+jnp.arange(len(state.temperature))*delta_t, 0),
        )

def initialize_NR(state: State):
    return NR_State(
        l=(state.L*state.X),
        v=(state.V*state.Y),
        temperature=state.temperature,
        f=(state.F*state.z[:, None]),
        s=jnp.where(state.U > 0, 1/(state.RR), 0 ),
        components=state.components,
        z=state.z,
        pressure=state.pressure,
        RR=state.RR,
        distillate=state.distillate,
        Nstages=state.Nstages,
        trays=Trays(
            low_tray=Tray(
                l=jnp.zeros_like(state.X),
                v=jnp.zeros_like(state.Y),
                T=jnp.zeros_like(state.temperature)
            ),
            high_tray=Tray(
                l=jnp.zeros_like(state.X),
                v=jnp.zeros_like(state.Y),
                T=jnp.zeros_like(state.temperature)
            ),
            tray=Tray(
                l=jnp.zeros_like(state.X),
                v=jnp.zeros_like(state.Y),
                T=jnp.zeros_like(state.temperature)
            ),
        ),
        h_feed=energy_functions.feed_enthalpy(state),
        t_condenser= jnp.array(300., dtype=float), #functions.t_solver(state, None, jnp.array(500.), jnp.array([0.98, 0.02]))[0],

        #t_condenser=jnp.array(0, dtype=float)
    )



def trays(state:NR_State):
    tray_l = vmap(tray_func, in_axes=(None, 0))(state, jnp.arange(1, len(state.temperature) + 1))
    tray_h = vmap(tray_func, in_axes=(None, 0))(state, jnp.arange(-1, len(state.temperature) - 1))
    tray_m = vmap(tray_func, in_axes=(None, 0))(state, jnp.arange(len(state.temperature)))
    return state._replace(trays=Trays(
        low_tray=Tray(
            l=tray_l.l.transpose(),
            v=tray_l.v.transpose(),
            T=tray_l.T.transpose(),
        ),
        high_tray=Tray(
            l=tray_h.l.transpose(),
            v=tray_h.v.transpose(),
            T=tray_h.T.transpose(),
        ),
        tray=Tray(
            l=tray_m.l.transpose(),
            v=tray_m.v.transpose(),
            T=tray_m.T.transpose(),
        ),
        )
    )


def tray_func(state:NR_State, j):
    return Tray(
        l=jnp.where((j < 0) | (j > state.Nstages-1), jnp.zeros_like(state.l[:, j]), state.l[:, j]),
        v=jnp.where((j < 0) | (j > state.Nstages-1), jnp.zeros_like(state.v[:, j]), state.v[:, j]),
        T=jnp.where((j < 0) | (j > state.Nstages-1), jnp.zeros_like(state.temperature[j]), state.temperature[j])
    )


def m_function(state: NR_State, tray_low, tray, tray_high, i, j):
    #l_flow = tray.v/jnp.sum(tray.v, axis=0)*state.RR*state.distillate
    #jnp.where(j == 0, (tray.l[i] + state.distillate + tray.v[i] - tray_high.l[i] - tray_low.v[i] - state.f[i, j]), tray.l[i]*(1+state.s[j]) + tray.v[i] - tray_high.l[i] - tray_low.v[i] - state.f[i, j])
    #return tray.l[i]*(1+state.s[j]) + tray.v[i] - tray_high.l[i] - tray_low.v[i] - state.f[i, j]
    purity = jnp.array([0.98, 0.02])
    return jnp.asarray(jnp.where(j == 0, (tray.l[i] * (1 + state.s[j]) + tray.v[i] - tray_high.l[i] - tray_low.v[i] - state.f[i, j]), tray.l[i]*(1+state.s[j]) + tray.v[i] - tray_high.l[i] - tray_low.v[i] - state.f[i, j]))

def e_function(state: NR_State, tray_low, tray, tray_high, i, j):
    #jnp.where(j == 0, tray_low.v/jnp.sum(tray_low.v) - tray.l/jnp.sum(tray.l), functions.k_eq(tray.T, state.components[i], state.pressure)*tray.l[i]*jnp.sum(tray.v, axis=0)/jnp.sum(tray.l, axis=0)-tray.v[i])
    #jnp.asarray(jnp.where(j == 0, (functions.k_eq(tray.T, state.components[i], state.pressure) - 1) * tray.l[i]/jnp.sum(tray.l), functions.k_eq(tray.T, state.components[i], state.pressure)*tray.l[i]*jnp.sum(tray.v, axis=0)/jnp.sum(tray.l, axis=0)-tray.v[i]))
    purity = jnp.array([0.98, 0.02])
    #return functions.k_eq(tray.T, state.components[i], state.pressure)*tray.l[i]*jnp.sum(tray.v, axis=0)/jnp.sum(tray.l, axis=0)-tray.v[i]
    return jnp.asarray(functions.k_eq(tray.T, state.components[i], state.pressure)*tray.l[i]*jnp.sum(tray.v, axis=0)/jnp.sum(tray.l, axis=0)-tray.v[i])


def h_function(state: NR_State, tray_low, tray, tray_high, j):
    #l_flow = jnp.where(j == 0, tray.v / jnp.sum(tray.v, axis=0) * state.distillate, tray_high.l)
    #jnp.sum(((1+state.s[j])*tray.l + tray.v) * energy_functions.liquid_enthalpy(tray.T, state.components), axis=0) - jnp.sum(tray_low.v * energy_functions.vapor_enthalpy(tray_low.T, state.components), axis=0)
    #jnp.sum(((1+state.s[j])*tray.l + tray.v) * energy_functions.liquid_enthalpy(tray.T, state.components), axis=0) - jnp.sum(tray_low.v * energy_functions.vapor_enthalpy(tray_low.T, state.components), axis=0)
    '''
    result = jnp.where(j == 0, jnp.sum(tray.l)-state.distillate*state.RR, (
                           jnp.where(j == state.Nstages - 1, jnp.sum(tray.l, axis=0) - (jnp.sum(state.f)-state.distillate), (
                                       jnp.sum(tray.l * energy_functions.liquid_enthalpy(tray.T, state.components), axis=0) + jnp.sum(tray.v * energy_functions.vapor_enthalpy(tray.T, state.components), axis=0)
                                       - jnp.sum(tray_high.l * energy_functions.liquid_enthalpy(tray_high.T, state.components)) - jnp.sum(
                                   tray_low.v * energy_functions.vapor_enthalpy(tray_low.T, state.components))) - jnp.sum(state.f[:, j] * state.h_feed))))
    result = jnp.where(j == 0, jnp.sum(tray.l) - (state.RR)*jnp.sum(tray.v), (
        jnp.where(j == state.Nstages - 1, jnp.sum(tray.l, axis=0) - (jnp.sum(state.f) - state.distillate), (
                jnp.sum(tray.l * energy_functions.liquid_enthalpy(tray.T, state.components), axis=0) + jnp.sum(
            tray.v * energy_functions.vapor_enthalpy(tray.T, state.components), axis=0)
                - jnp.sum(tray_high.l * energy_functions.liquid_enthalpy(tray_high.T, state.components)) - jnp.sum(
            tray_low.v * energy_functions.vapor_enthalpy(tray_low.T, state.components))) - jnp.sum(
            state.f[:, j] * state.h_feed))))
    '''
    result = jnp.where(j == 0, jnp.sum(tray.l) - state.distillate*state.RR, (
        jnp.where(j == state.Nstages - 1, jnp.sum(tray.l, axis=0) - (jnp.sum(state.f) - state.distillate), (
                jnp.sum(tray.l * energy_functions.liquid_enthalpy(tray.T, state.components), axis=0) + jnp.sum(
            tray.v * energy_functions.vapor_enthalpy(tray.T, state.components), axis=0)
                - jnp.sum(tray_high.l * energy_functions.liquid_enthalpy(tray_high.T, state.components)) - jnp.sum(
            tray_low.v * energy_functions.vapor_enthalpy(tray_low.T, state.components))) - jnp.sum(
            state.f[:, j] * state.h_feed))))


    return result


def f_vector_function(state: NR_State, tray_low, tray, tray_high, j):
    h = jnp.asarray(jnp.where(j < state.Nstages, h_function(state, tray_low, tray, tray_high, j), 0))
    m = jnp.asarray(jnp.where(j < state.Nstages, vmap(m_function, in_axes=(None, None, None, None, 0, None))(state, tray_low, tray, tray_high, jnp.arange(len(state.components)), j), 0))
    e = jnp.asarray(jnp.where(j < state.Nstages, vmap(e_function, in_axes=(None, None, None, None, 0, None))(state, tray_low, tray, tray_high, jnp.arange(len(state.components)), j), 0))

    return Mesh(H=h,
                M=m,
                E=e,
                )



def f_jac_a(state: NR_State, tray_low, tray, tray_high, j):
    return jacfwd(f_vector_function, argnums=3)(
        state,
        Tray(l=tray_low.l[:, j], v=tray_low.v[:, j], T=tray_low.T[j]),
        Tray(l=tray.l[:, j], v=tray.v[:, j], T=tray.T[j]),
        Tray(l=tray_high.l[:, j], v=tray_high.v[:, j], T=tray_high.T[j]),
        j,
    )

def f_jac_b(state: NR_State, tray_low, tray, tray_high, j):
    return jacfwd(f_vector_function, argnums=2)(
        state,
        Tray(l=tray_low.l[:, j], v=tray_low.v[:, j], T=tray_low.T[j]),
        Tray(l=tray.l[:, j], v=tray.v[:, j], T=tray.T[j]),
        Tray(l=tray_high.l[:, j], v=tray_high.v[:, j], T=tray_high.T[j]),
        j,
    )


def f_jac_c(state: NR_State, tray_low, tray, tray_high, j):
    return jacfwd(f_vector_function, argnums=1)(
        state,
        Tray(l=tray_low.l[:, j], v=tray_low.v[:, j], T=tray_low.T[j]),
        Tray(l=tray.l[:, j], v=tray.v[:, j], T=tray.T[j]),
        Tray(l=tray_high.l[:, j], v=tray_high.v[:, j], T=tray_high.T[j]),
        j,
    )

def f_sol(state: NR_State, tray_low, tray, tray_high, j):
    return f_vector_function(
        state,
        Tray(l=tray_low.l[:, j], v=tray_low.v[:, j], T=tray_low.T[j]),
        Tray(l=tray.l[:, j], v=tray.v[:, j], T=tray.T[j]),
        Tray(l=tray_high.l[:, j], v=tray_high.v[:, j], T=tray_high.T[j]),
        j,
    )


def tuple_to_matrix(tuple_input: Mesh, tray_index):
    matrix_h = jnp.concatenate((tuple_input.H.v[tray_index], jnp.array([tuple_input.H.T[tray_index]]), tuple_input.H.l[tray_index]))
    matrix_m = jnp.concatenate((tuple_input.M.v[tray_index], jnp.array([tuple_input.M.T[tray_index]]), tuple_input.M.l[tray_index]))
    matrix_e = jnp.concatenate((tuple_input.E.v[tray_index], jnp.array([tuple_input.E.T[tray_index]]), tuple_input.E.l[tray_index]))

    return jnp.concatenate((matrix_h[None, :], matrix_m.transpose(), matrix_e.transpose()))


def non_zero_matrix_b(state, matrix, j):
    non_zeros = jnp.diag(-1 * jnp.ones(len(matrix[j]) - 1), k=+1) + jnp.diag(jnp.ones(len(matrix[j])))
    return jnp.where(j < state.Nstages, matrix[j], non_zeros)


def non_zero_matrix_ac(state, matrix, j):
    non_zeros = jnp.diag(-1*jnp.ones(len(matrix[j])-1), k=+1) + jnp.diag(jnp.ones(len(matrix[j])))
    return jnp.where(j < state.Nstages-1, matrix[j], non_zeros)


def dx_mask(state, matrix, j):
    return jnp.where(j < state.Nstages, matrix[j], jnp.zeros_like(matrix[j]))



def single_tray(tray, j):
    return Tray(l=tray.l[:, j], v=tray.v[:, j], T=tray.T[j])


def single_B(tray, j, state):
    result = set_B(single_tray(tray, j), state)
    first_tray = jnp.zeros(2*len(tray.v)+1)
    first_tray = first_tray.at[-len(tray.v):].set(1)
    return jnp.where(j==0, result.at[0].set(first_tray),  jnp.where(j==state.Nstages-1, result.at[0].set(first_tray), set_B(single_tray(tray, j), state)))


def single_A(tray, j, state):
    result = set_A(single_tray(tray, j), state)
    return jnp.where(j==state.Nstages-2, result.at[0, len(tray.v):].set(0), result)


def single_C(tray, j, state):
    result = set_C(single_tray(tray, j), state)
    return jnp.where(j==0, result.at[0, 0:len(tray.v)+1].set(0), result)


def update_NR(state: NR_State, iteration, res_data, f_damp):
    t_condenser = functions.t_solver(state, None, state.trays.tray.T[0],
                                     jnp.array(state.trays.tray.l[:, 0] / jnp.sum(state.trays.tray.l[:, 0], axis=0)))
    state = state._replace(t_condenser=t_condenser[0])

    #t_condenser = functions.t_solver(state, None, state.trays.tray.T[0], jnp.array(state.trays.tray.v[:, 0] / jnp.sum(state.trays.tray.v[:, 0], axis=0)))
    #state = state._replace(t_condenser=t_condenser)
    #mask = jnp.max(jnp.where(state.f > 0, 1, 0), axis=1)
    b = vmap(f_jac_b, in_axes=(None, None, None, None, 0))(state, state.trays.low_tray, state.trays.tray, state.trays.high_tray, jnp.arange(len(state.trays.tray.T)))
    a = vmap(f_jac_a, in_axes=(None, None, None, None, 0))(state, state.trays.low_tray, state.trays.tray, state.trays.high_tray, jnp.arange(len(state.trays.tray.T)))
    c = vmap(f_jac_c, in_axes=(None, None, None, None, 0))(state, state.trays.low_tray, state.trays.tray, state.trays.high_tray, jnp.arange(len(state.trays.tray.T)))
    f_s = vmap(f_sol, in_axes=(None, None, None, None, 0))(state, state.trays.low_tray, state.trays.tray, state.trays.high_tray, jnp.arange(len(state.trays.tray.T)))

    b_transform = vmap(tuple_to_matrix, in_axes=(None, 0))(b, jnp.arange(len(state.temperature)))
    a_transform = vmap(tuple_to_matrix, in_axes=(None, 0))(a, jnp.arange(1, len(state.temperature)))
    c_transform = vmap(tuple_to_matrix, in_axes=(None, 0))(c, jnp.arange(len(state.temperature)-1))
    f_transform = jnp.concatenate((jnp.asarray(f_s.H)[:,None], jnp.asarray(f_s.M), jnp.asarray(f_s.E)), axis=1)

    b_transform = vmap(non_zero_matrix_b, in_axes=(None, None, 0))(state, b_transform, jnp.arange(len(state.temperature)))
    #a_transform = vmap(non_zero_matrix_ac, in_axes=(None, None, 0))(state, a_transform, jnp.arange(len(state.temperature)-1))
    #c_transform = vmap(non_zero_matrix_ac, in_axes=(None, None, 0))(state, c_transform, jnp.arange(len(state.temperature)-1))
    #f_transform = vmap(non_zero_matrix_f, in_axes=(None, None, 0))(state, f_transform, jnp.arange(len(state.temperature)-1))
    #dx = jnp.zeros_like(f_transform)
    #single_tray = Tray(l=state.trays.tray.l[:, 2], v=state.trays.tray.v[:, 2], T=state.trays.tray.T[2])
    new_B = vmap(single_B, in_axes=(None, 0, None))(state.trays.tray, jnp.arange(len(state.temperature)), state)
    new_A = vmap(single_A, in_axes=(None, 0, None))(state.trays.tray, jnp.arange(len(state.temperature)-1), state)
    new_C = vmap(single_C, in_axes=(None, 0, None))(state.trays.tray, jnp.arange(len(state.temperature)-1), state)

    #dx = functions.thomas(a_transform, b_transform, c_transform, -1*f_transform, state.Nstages)
    dx = functions.thomas(new_A, new_B, new_C, -1 * f_transform, state.Nstages)

    def minimize_res(t, res_state: NR_State, dx):
        '''
        v_new = jnp.where(dx[:, 0:len(state.components)].transpose()/jnp.max(state.trays.tray.v) > 1, (state.trays.tray.v + t * state.trays.tray.v * 1 * dx[:, 0:len(state.components)].transpose()/jnp.abs(dx[:, 0:len(state.components)].transpose())), (state.trays.tray.v + t * dx[:, 0:len(state.components)].transpose()))

        l_new = jnp.where(dx[:, len(state.components) + 1:].transpose() / jnp.max(state.trays.tray.l) > 1, (
                    state.trays.tray.l + t * state.trays.tray.l * 1 * dx[:, len(state.components) + 1:].transpose() / jnp.abs(
                dx[:, len(state.components) + 1:].transpose())),
                          (state.trays.tray.l + t * dx[:, len(state.components) + 1:].transpose()))
        '''
        v_new = (res_state.trays.tray.v + t * dx[:, 0:len(res_state.components)].transpose())
        l_new = (res_state.trays.tray.l + t * dx[:, len(res_state.components) + 1:].transpose())
        temp = (res_state.trays.tray.T + t * dx[:, len(res_state.components)].transpose())
        #temp = jnp.where(dx[:, len(state.components)].transpose() < 10., jnp.where(dx[:, len(state.components)].transpose() > -10., state.trays.tray.T + t * dx[:, len(state.components)].transpose(), state.trays.tray.T - 10.), state.trays.tray.T + 10.)
        v_new_final = jnp.where(v_new>0, v_new, res_state.trays.tray.v
                                * jnp.exp( t * dx[:, 0:len(res_state.components)].transpose()/res_state.trays.tray.v))
        l_new_final = jnp.where(l_new > 0, l_new, res_state.trays.tray.l
                                * jnp.exp(t * dx[:, len(res_state.components) + 1:].transpose() / res_state.trays.tray.l))
        temp_final = jnp.where(jnp.abs(dx[:, len(res_state.components)]) > 7, (res_state.trays.tray.T + 7*(dx[:, len(res_state.components)].transpose())/(jnp.abs(dx[:, len(res_state.components)].transpose()))), (res_state.trays.tray.T + t * dx[:, len(res_state.components)].transpose()))

        res_state = res_state._replace(
            v=v_new_final,
            l=l_new_final,
            temperature=temp,
        )

        res_state = trays(res_state)
        f_vector = vmap(f_sol, in_axes=(None, None, None, None, 0))(res_state, res_state.trays.low_tray, res_state.trays.tray, res_state.trays.high_tray, jnp.arange(len(res_state.trays.tray.T)))
        return jnp.array(sum((f_vector.H/1000) ** 2 + jnp.sum(f_vector.M ** 2 + (f_vector.E/1000.)**2, axis=1)), dtype=float)


    def min_res(state, dx, t):
        #t = jnp.sum(state.trays.tray.l)/jnp.max(dx[:, len(state.components) + 1:])
        '''
        v_new = jnp.where(dx[:, 0:len(state.components)].transpose()/jnp.max(state.trays.tray.v) > 1, (state.trays.tray.v + t * state.trays.tray.v * 1 * dx[:, 0:len(state.components)].transpose()/jnp.abs(dx[:, 0:len(state.components)].transpose())), (state.trays.tray.v + t * dx[:, 0:len(state.components)].transpose()))

        l_new = jnp.where(dx[:, len(state.components) + 1:].transpose() / jnp.max(state.trays.tray.l) > 1, (
                    state.trays.tray.l + t * state.trays.tray.l * 1 * dx[:, len(state.components) + 1:].transpose() / jnp.abs(
                dx[:, len(state.components) + 1:].transpose())),
                          (state.trays.tray.l + t * dx[:, len(state.components) + 1:].transpose()))
        '''
        v_new = (state.trays.tray.v + t * dx[:, 0:len(state.components)].transpose())
        l_new = (state.trays.tray.l + t * dx[:, len(state.components) + 1:].transpose())
        temp = (state.trays.tray.T + t * dx[:, len(state.components)].transpose())
        #temp = jnp.where(dx[:, len(state.components)].transpose() < 10., jnp.where(dx[:, len(state.components)].transpose() > -10., state.trays.tray.T + t * dx[:, len(state.components)].transpose(), state.trays.tray.T - 10.), state.trays.tray.T + 10.)
        v_new_final = jnp.where(v_new >= 0., v_new, state.trays.tray.v
                                * (jnp.exp(t*dx[:, 0:len(state.components)].transpose()/(state.trays.tray.v))))
        l_new_final = jnp.where(l_new >= 0., l_new, state.trays.tray.l
                                * (jnp.exp(t*dx[:, len(state.components) + 1:].transpose() / (state.trays.tray.l))))

        temp_final = jnp.where(jnp.abs(dx[:, len(state.components)]) > 7, (state.trays.tray.T + 7*(dx[:, len(state.components)].transpose())/(jnp.abs(dx[:, len(state.components)].transpose()))), (state.trays.tray.T + t * dx[:, len(state.components)].transpose()))
        state = state._replace(
            v=v_new_final,
            l=l_new_final,
            temperature=temp,
        )

        state = trays(state)
        f_vector = vmap(f_sol, in_axes=(None, None, None, None, 0))(state, state.trays.low_tray, state.trays.tray, state.trays.high_tray, jnp.arange(len(state.trays.tray.T)))
        return jnp.array(sum((f_vector.H/1000) ** 2 + jnp.sum(f_vector.M ** 2 + (f_vector.E/1000.)**2, axis=1)), dtype=float), state, t


    #result = vmap(minimize_res, in_axes=(0, None, None))(jnp.arange(0.1, 0.976, 0.05), state, dx)
    #new_t_vmap = jnp.max(jnp.where(result == jnp.max(result), jnp.arange(0.1, 0.976, 0.05), 0))
    #new_t = jnp.where(res_data[iteration] < 0.99 * res_data[iteration - 1], f_damp * 1.1, f_damp / 1.1)
    #new_t = jnp.where((jnp.asarray(new_t) > new_t_vmap) & (new_t < 1.2), jnp.asarray(new_t), new_t_vmap)
    #  new_t = jnp.where(new_t < jnp.array(1.2, dtype=float), jnp.asarray(new_t), 1.2)
    #new_t = 1/jnp.log((jnp.average(res_data)/jnp.min(res_data))**2+1)
    #new_t = jnp.where(new_t<0, 0.1, new_t)
    #new_t = jnp.where(new_t<1.2, new_t, 1.2)
    #new_t = solver(fun=minimize_res).run(init_params=jnp.array(0.3, dtype=float), bounds=(0.1, 0.6), res_state=state, dx=dx).params

    new_t = 1./jnp.pi
    t_a = 1.
    res, state_new, zeros = min_res(state, dx, new_t)

    return res, state_new, zeros


def nr_to_state(nr_state, state):
    x_condenser = nr_state.trays.tray.v[:, 0]/ jnp.sum(nr_state.trays.tray.v[:,0], axis=0)
    t_condenser = functions.t_solver(nr_state, None, nr_state.trays.tray.T[0], x_condenser)
    x = jnp.nan_to_num(nr_state.trays.tray.l / jnp.sum(nr_state.trays.tray.l, axis=0))
    y = jnp.nan_to_num(nr_state.trays.tray.v / jnp.sum(nr_state.trays.tray.v, axis=0))
    #x = jnp.concatenate((x_condenser[None,:], x_nr.transpose())).transpose()
    #y = jnp.concatenate((x_condenser[None,:], y_nr.transpose())).transpose()
    t = jnp.concatenate((t_condenser, nr_state.trays.tray.T))
    l = jnp.sum(nr_state.trays.tray.l, axis=0)
    v = jnp.sum(nr_state.trays.tray.v, axis=0)
    return state._replace(X=x,
                          Y=y,
                          temperature=nr_state.trays.tray.T,
                          L=l,
                          V=v,
                          #Hfeed=nr_state.h_feed,
                          #Hliq=jnp.sum(vmap(energy_functions.liquid_enthalpy, in_axes=(0, None))(nr_state.trays.tray.T, nr_state.components).transpose()*x, axis=0),
                          #Hvap=jnp.sum(vmap(energy_functions.vapor_enthalpy, in_axes=(0, None))(nr_state.trays.tray.T,
                          #                                                                       nr_state.components).transpose() * y, axis=0),
                          )


def cond_fn(args):
    nr_state, iterations, res, res_array, profiler, damping = args
    cond = nr_state.Nstages*(2*len(nr_state.components)+1)*jnp.sum(nr_state.f)*1e-9
    return (iterations < 1000) & (res > cond) & jnp.where(iterations > 10, (res_array[iterations] < res_array[iterations-1]), True)


def body_fn(args):
    nr_state, iterations, res, res_array, profiler, damping = args
    res_new, nr_state_new, new_t = update_NR(nr_state, iterations, res_array, damping[iterations-1])
    #nr_state_new = nr_state_new._replace(state_list=nr_state_new.state_list.add(nr_state_new))
    res_array = res_array.at[iterations].set(res)
    profiler = profiler.at[iterations, :, :].set(nr_state_new.trays.tray.l)
    damping = damping.at[iterations].set(new_t)
    iterations += 1
    return nr_state_new, iterations, res_new, res_array, profiler, damping


def store_variable(small_array, larger_array, start_indices):
    # Assuming small_array has shape (2, 20) and larger_array has shape (9000, 20)
    updated_larger_array = lax.dynamic_update_slice(larger_array, small_array, start_indices)
    return updated_larger_array


def converge_column(state: State):
    nr_state = initialize_NR(state)
    nr_state = trays(nr_state)
    iterations = 0
    res = 0

    nr_state, iterations, res, res_array, profiler, damping = (
        lax.while_loop(cond_fun=cond_fn, body_fun=body_fn,
                       init_val=(nr_state,
                                 iterations,
                                 jnp.array(1, dtype=float),
                                 jnp.ones(1000, dtype=float)*1e10,
                                 jnp.zeros((1000, len(state.components), len(state.temperature))),
                                 jnp.ones(1000, dtype=float),
                                 )
                       )
    )

    '''
    res_array = jnp.zeros(1000, dtype=float)
    f_damp = jnp.array(1., dtype=float)
    damping = jnp.ones(1000, dtype=float)*0.01
    profiler = jnp.zeros((1000, 3, len(state.temperature)), dtype=float)
    #res, state_new, new_t = (update_NR)(nr_state, iterations, res_data, f_damp)
    
    for i in range(70):
        res, nr_state, new_t = jit(update_NR)(nr_state, iterations, res_array, damping[iterations-1])
        state = nr_to_state(nr_state, state)
        iterations +=1
        state = state._replace(storage=store_variable(jnp.array([state.X[2,:]]), state.storage,
                                                      (iterations, iterations + 1)),
                               res=state.res.at[iterations].set(res))
        res_array = res_array.at[iterations].set(res)
        damping = damping.at[iterations].set(new_t)
        profiler = profiler.at[iterations, :, :].set(state.X)

    for i in range(30):
        res, nr_state, new_t = (update_NR)(nr_state, iterations, res_array, damping[iterations-1])
        state = nr_to_state(nr_state, state)
        iterations +=1
        state = state._replace(storage=store_variable(jnp.array([state.X[2,:]]), state.storage,
                                                      (iterations, iterations + 1)),
                               res=state.res.at[iterations].set(res))
        res_array = res_array.at[iterations].set(res)
        damping = damping.at[iterations].set(new_t)
        profiler = profiler.at[iterations, :, :].set(state.X)
        
    '''
    state = nr_to_state(nr_state, state)


    state = state._replace(res=res_array,
                           profiler=profiler,
                           damping=damping
                           )

    return state, iterations, res

def body_mixed(args):
    state, iteration, res = args
    feed_stage = jnp.max(jnp.where(state.F>0, jnp.arange(len(state.temperature)), 0))

    for c in range(10):
        state = initial_composition.model_solver(state)
        state = functions.stage_temperature(state)

    state = functions.y_func(state)
    state = state._replace(
        Hliq=jnp.sum(vmap(energy_functions.liquid_enthalpy, in_axes=(0, None))(state.temperature, state.components).transpose() * state.X, axis=0),
        Hvap=jnp.sum(vmap(energy_functions.vapor_enthalpy, in_axes=(0, None))(state.temperature, state.components).transpose() * state.Y, axis=0)
    )

    '''
    state, iterator, res = jit(bubble_simulation)(
        state.Nstages,
        feed_stage,
        state.pressure,
        jnp.sum(state.F),
        state.z,
        state.distillate,
        state.RR
    )
    '''
    state, iterations, res = converge_column(state)
    iteration +=iterations
    return state, iterations, res


def cond_mixed(args):
    state, iterations, res = args
    cond = state.Nstages * (2 * len(state.components) + 1) * jnp.sum(state.F) * 1e-9
    return (res > cond) & (iterations<100)


def inside_simulation(nstages, feedstage, pressure, feed, z, distillate, rr):
    iterator = 0
    res = 0
    #feedstage = jnp.floor((nstages + 1) / 2 )
    state = initialize()
    state = initial_guess(state=state, nstages=nstages, feedstage=feedstage, pressure=pressure, feed=feed, z=z,
                               distillate=distillate, rr=rr)

    state = initial_temperature(state)
    state = state._replace(Hfeed=jnp.where(state.F > 0, jnp.sum(energy_functions.feed_enthalpy(state)*state.z), 0))
    '''
    for c in range(10):
        state = initial_composition.model_solver(state)
        state = functions.stage_temperature(state)

    state = functions.y_func(state)
    state = state._replace(
        Hliq=jnp.sum(vmap(energy_functions.liquid_enthalpy, in_axes=(0, None))(state.temperature, state.components).transpose() * state.X, axis=0),
        Hvap=jnp.sum(vmap(energy_functions.vapor_enthalpy, in_axes=(0, None))(state.temperature, state.components).transpose() * state.Y, axis=0)
    )
   
    state = functions.flowrates(state)
    
    state, iterator, res = bubble_simulation(
        nstages,
        feedstage,
        pressure,
        feed,
        z,
        distillate,
        rr
    )
    '''

    state, iterator, res = (
        lax.while_loop(cond_fun=cond_mixed, body_fun=body_mixed,
                       init_val=(state,
                                 iterator,
                                 jnp.array(1, dtype=float),
                                 )
                       )
    )

    #state, iterator, res = converge_column(state)
    #state = condensor_duty(state)
    #state = reboiler_duty(state)
    #state = costing.tac(state)

    return state, iterator, res


def condensor_duty(state: State):
    result = jnp.sum(
        state.V[1] * state.Y[:, 1] * energy_functions.vapor_enthalpy(state.temperature[1], state.components)) + jnp.sum(
        state.F[0] * state.z * state.Hfeed[0]) - jnp.sum(
        (state.L[0] + state.distillate) * state.X[:, 0] * energy_functions.liquid_enthalpy(state.temperature[0],
                                                                                     state.components)) - jnp.sum(
        state.V[0]*0 * state.Y[:, 0] * (1 + 0) * energy_functions.vapor_enthalpy(state.temperature[0], state.components))

    return state._replace(CD=result)

def reboiler_duty(state: State):

    def sidestreams(j):
        return (jnp.sum(state.F[j]*state.z*state.Hfeed[j]) - jnp.sum((state.U[j])*state.X[:, j]*energy_functions.liquid_enthalpy(state.temperature[j], state.components))
                - jnp.sum(state.V[j]*state.Y[:, j]* 0 * energy_functions.vapor_enthalpy(state.temperature[j], state.components)))

    result = jnp.sum(vmap(sidestreams)(jnp.arange(len(state.L)))) - state.CD + jnp.sum(- jnp.sum(state.V[0] * state.Y[:, 0]*energy_functions.vapor_enthalpy(state.temperature[0], state.components)) - jnp.sum(state.L[state.Nstages-1] * state.X[:, state.Nstages-1] * energy_functions.liquid_enthalpy(state.temperature[state.Nstages-1], state.components)))

    return state._replace(RD=result)


#zf = jnp.array([0.8563363, 0.14366373])
st = time()
zf = jnp.array([0., 0.5, 0.5])
state, iterations, res = jit(inside_simulation)(jnp.array(10, dtype=int), jnp.array(5, dtype=int), jnp.array(2.026, dtype=float),
                                      jnp.array(1000.0, dtype=float), jnp.array(zf, dtype=float),
                                      jnp.array(500, dtype=float), jnp.array(1, dtype=float))
print(time()-st)

st = time()
zf = jnp.array([0.2, 0.5, 0.3])
state, iterations, res = jit(inside_simulation)(jnp.array(25, dtype=int), jnp.array(13, dtype=int), jnp.array(2.026, dtype=float),
                                      jnp.array(1000.0, dtype=float), jnp.array(zf, dtype=float),
                                      jnp.array(400, dtype=float), jnp.array(1, dtype=float))
print(time()-st)

plot_function(jnp.arange(1, state.Nstages+1), state.L[0:state.Nstages], state.V[0:state.Nstages], state.temperature[0:state.Nstages], state.Hliq[0:state.Nstages], state.Hvap[0:state.Nstages], state.X[jnp.where(zf != 0, jnp.arange(0, len(zf)), 0), 0:state.Nstages], state.Y[jnp.where(zf != 0, jnp.arange(0, len(zf)), 0), 0:state.Nstages], jnp.where(zf != 0, jnp.arange(0, len(zf)), 0))


'''
zf = jnp.array([0., 0., 0., 0.5, 0.5, 0., 0., 0.], dtype=float)



st = time()
state, iterations, res = (inside_simulation)(
    jnp.array(10, dtype=int),
    jnp.array(5, dtype=int),
    jnp.array(2., dtype=float),
    jnp.array(1000., dtype=float),
    zf,
    jnp.array(500, dtype=float),
    jnp.array(2, dtype=float)
)
print(time()-st)
print(iterations)
'''
'''

tac_array = jnp.zeros(18, dtype=float)

for co in range(18):
    st = time()
    state, iterations, res = jit(inside_simulation)(jnp.array(co+3, dtype=int), jnp.array(jnp.floor((co+3)/2+1), dtype=int), jnp.array(2., dtype=float), jnp.array(1000., dtype=float), zf, jnp.array(500, dtype=float), jnp.array(1., dtype=float))
    tac_array = tac_array.at[co].set(state.TAC)
    print(iterations)
    print(time()-st)
    print(state.TAC)


nt=time()
state, iterations, res = jit(vmap(inside_simulation, in_axes=(0, None, None, None, None, None, None)))(
    jnp.array(jnp.arange(4) + 3, dtype=int),
    jnp.array(5, dtype=int),
    jnp.array(2., dtype=float), jnp.array(1000., dtype=float), zf,
    jnp.array(500, dtype=float), jnp.array(1., dtype=float))
print(time()-nt)
'''
'''
nt=time()
state, iterations, res = jit(vmap(inside_simulation, in_axes=(0, None, None, None, None, None, None)))(
    jnp.array(jnp.arange(2) + 8, dtype=int),
    jnp.array(5, dtype=int),
    jnp.array(1., dtype=float), jnp.array(1000., dtype=float), zf,
    jnp.array(500, dtype=float), jnp.array(2., dtype=float))
print(time()-nt)


jit(vmap(inside_simulation, in_axes=(0, 0, None, None, None, None, None)))(jnp.array(co + 5, dtype=int),
                                                                           jnp.array(jnp.floor((co + 5) / 2 + 1),
                                                                                     dtype=int),
                                                                           jnp.array(1., dtype=float),
                                                                           jnp.array(1000., dtype=float), zf,
                                                                           jnp.array(500, dtype=float),
                                                                           jnp.array(2., dtype=float))
'''

#Sstate, iterations = simulation(jnp.array(30, dtype=int), jnp.array(15, dtype=int), jnp.array(2., dtype=float), jnp.array(1000., dtype=float), zf, jnp.array(400, dtype=float), jnp.array(1., dtype=float))

'''
state = initialize()
state = initial_guess(state, jnp.array(6, dtype=int), jnp.array(3, dtype=int), jnp.array(2., dtype=float), jnp.array(1000., dtype=float), zf, jnp.array(500, dtype=float), jnp.array(1., dtype=float))
state = initial_temperature(state)
state = initial_composition.model_solver(state)
#state, iterator = simulation(jnp.array(6, dtype=int), jnp.array(3, dtype=int), jnp.array(2.026, dtype=float), jnp.array(1000.0, dtype=float), jnp.array(zf, dtype=float), jnp.array(500, dtype=float), jnp.array(1, dtype=float))
nr_state = initialize_NR(state)
nr_state = trays(nr_state)




starting = time()
for i in range(20):
    st = time()
    res, nr_state = jit(update_NR)(nr_state)
    print(time()-st)
    print(nr_state.trays.tray.T)
    print(res)
print(time()-starting)

#res, nr_state =update_NR(nr_state)
#vmap(f_jac_b, (None, None, None, None, 0))(nr_state, tray_low, tray, tray_high, jnp.arange(nr_state.Nstages))
#print(result)
'''
'''
'''

cond = state.Nstages*(2*len(state.components)+1)*jnp.sum(state.F)*1e-9
fig4 = plt.figure(5)
ax4 = fig4.add_subplot(111)
# calculate mole fractions
ax4.plot(jnp.arange(len(state.res[0:iterations])), state.res[0:iterations])
ax4.set_ylabel('residuals')
ax4.set_xlabel('iteration')
ax4.set_ylim([0, 10*jnp.min(res)])
ax4.set_xlim([0, 300])
#ax4.legend()
ax4.grid()

fig5 = plt.figure(6)
ax5 = fig5.add_subplot(111)
# calculate mole fractions
ax5.plot(jnp.arange(len(state.damping[0:iterations])), state.damping[0:iterations])
ax5.set_ylabel('damping_factor')
ax5.set_xlabel('iteration')
#ax4.legend()
ax5.grid()
#print(state.res)


print(iterations)



#print(iterations)
'''
'''

non_zero_mask = jnp.any(state.storage != 0, axis=1)

#datanz = state.storage[non_zero_mask][:, 0:state.Nstages]
#data = datanz[::jnp.array(jnp.ceil(len(datanz)/100), dtype=int)]
# Create a figure and axis
'''
'''

#plot_function(jnp.arange(1, state.Nstages+1), state.L[0:state.Nstages], state.V[0:state.Nstages], state.temperature[0:state.Nstages], state.Hliq[0:state.Nstages], state.Hvap[0:state.Nstages], state.X[jnp.where(zf != 0, jnp.arange(0, len(zf)), 0), 0:state.Nstages], state.Y[jnp.where(zf != 0, jnp.arange(0, len(zf)), 0), 0:state.Nstages], jnp.where(zf != 0, jnp.arange(0, len(zf)), 0))


data = (state.profiler/jnp.sum(state.profiler, axis=1).reshape(len(state.profiler),1,len(state.temperature)))[0:iterations, :, :]

fig, ax = plt.subplots()

# Set limits for the axes
ax.set_xlim(1, len(data[0].transpose()))
#ax.set_ylim(min(min(row) for row in data[1:, :]), max(max(row) for row in data[1:, :]))
ax.set_ylim(0, 1)
# Create an empty line object for the animation
line, = ax.plot([], [], lw=2)


def init():
    line.set_data([], [])
    return line,


# Function to update the plot for each frame of the animation
def update(frame):
    ax.cla()
    x = jnp.arange(1, len(data[frame].transpose())+1)
    y = data[frame].transpose()
    ax.cla()
    ax.plot(x, y)
    ax.legend([f'Iteration {frame}'], loc=9)
    ax.set_xlabel('Stage')
    ax.set_ylabel('C7 mole flow kmole/hr')
    return ax,

ani = FuncAnimation(fig, update, frames=len(data), repeat=True)


plt.show()
'''
#plt.show()


cd = 0
#plot_function(jnp.arange(1, state.Nstages[cd]+1), state.L[cd, 0:state.Nstages[cd]], state.V[cd, 0:state.Nstages[cd]], state.temperature[cd, 0:state.Nstages[cd]], state.Hliq[cd, 0:state.Nstages[cd]], state.Hvap[cd, 0:state.Nstages[cd]], state.X[cd, :, cd:state.Nstages[cd]], state.Y[cd, :, cd:state.Nstages[cd]], jnp.arange(0, len(zf)))
#plt.show()

'''
