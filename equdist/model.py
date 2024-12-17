import jax.numpy as jnp
import jax
from jaxopt import Bisection, Broyden, LevenbergMarquardt, GaussNewton, ScipyRootFinding
from jax import vmap, lax, jit
from equdist.distillation_types import State, Tray
from equdist import functions, matrix_transforms, jacobian, \
    thermodynamics, costing, equimolar
import os

# from NR_model_test.analytic_jacobian import jacobian as pure_jac
# from NR_model_test import analytic_jacobian
import os


def initialize():
    n_max = 90
    c_max = 10
    dir = os.path.join(os.getcwd(), 'Pure component parameters')
    # psat_params, cpvap_params, hvap_params, hform_params = thermodynamics.retrieve_params(dir)
    return State(
        L=jnp.zeros(n_max, dtype=float),
        V=jnp.zeros(n_max, dtype=float),
        U=jnp.zeros(n_max, dtype=float),
        W=jnp.zeros(n_max, dtype=float),
        X=jnp.zeros((c_max, n_max), dtype=float),
        Y=jnp.zeros((c_max, n_max), dtype=float),
        temperature=jnp.zeros(n_max, dtype=float),
        temperature_bounds=jnp.zeros(2, dtype=float),
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
        heavy_key=jnp.zeros((), dtype=int),
        light_key=jnp.zeros((), dtype=int),
        heavy_spec=jnp.zeros((), dtype=float),
        light_spec=jnp.zeros((), dtype=float),
        NR_residuals=jnp.ones((), dtype=float),
        EQU_residuals=jnp.ones((), dtype=float),
        analytics=jnp.zeros((), dtype=bool),
        converged=jnp.ones((), dtype=bool),
        NR_iterations=jnp.zeros((), dtype=int),
        EQU_iterations=jnp.zeros((), dtype=int),
        BP_iterations=jnp.zeros((), dtype=int),
        damping=jnp.zeros((), dtype=float),
        dx=jnp.zeros((100, n_max*(2*c_max+1)), dtype=float),
        profile=jnp.zeros((100, n_max), dtype=float)
    )


def initial_guess(state: State, nstages, feedstage, pressure, feed, z, distillate, rr, analytics: bool, specs: bool,
                  heavy_recovery, light_recovery):
    l_init = jnp.where(jnp.arange(len(state.L)) < feedstage - 1, rr * distillate, rr * distillate + jnp.sum(feed))
    l_init = l_init.at[nstages - 1].set((jnp.sum(feed) - distillate))
    v_init = jnp.where(jnp.arange(len(l_init)) > 0, (rr + jnp.ones_like(l_init)) * distillate, distillate)
    f = jnp.where(jnp.arange(len(l_init)) == feedstage - 1, feed, 0)
    mask = jnp.where(jnp.arange(len(state.temperature)) < nstages, 1, 0)
    return state.replace(
        L=l_init * mask,
        V=v_init * mask,
        z=z,
        RR=rr,
        distillate=distillate,
        pressure=pressure,
        F=f,
        Nstages=nstages,
        components=jnp.arange(len(z)),  #jnp.array([0, 2, 3, 0], dtype=int),  #
        heavy_key=jnp.zeros((), dtype=int),  # jnp.where(specs == True, fug_state.heavy_key, jnp.zeros((), dtype=int)),
        light_key=jnp.zeros((), dtype=int),  # jnp.where(specs == True, fug_state.light_key, jnp.zeros((), dtype=int)),
        heavy_spec=heavy_recovery,
        light_spec=light_recovery
    )


def initial_temperature(state: State):

    t_split = jnp.concatenate(
        (jnp.zeros((), dtype=float)[None], jnp.diff(jnp.sort(jnp.where(state.z > 0, t_range, 0)))))

    max_index = jnp.max(jnp.where(t_split == jnp.max(t_split), jnp.arange(len(t_split)), 0))
    t_split = jnp.where((t_split == t_split[max_index]) & (max_index > 0), 0, t_split)

    t_avg_index = jnp.sum(jnp.where(t_split == jnp.max(t_split), jnp.arange(len(t_split)), 0)) - 1
    t_avg = t_range[t_avg_index]
    delta_t = t_split[t_avg_index + 1] / (state.Nstages)
    t_avg = t_avg - t_split[t_avg_index + 1] / 2


    return state.replace(
        #temperature=jnp.where(jnp.arange(len(state.temperature)) < state.Nstages,
        #                      jnp.max(t_minmax) - t_split[t_avg_index + 1] + jnp.arange(len(state.temperature)) * delta_t, 0),
        temperature_bounds=jnp.array([jnp.min(min_t), jnp.max(t_minmax)])
    )


def split_guess(state: State, feedstage):
    f = jnp.cumsum(jnp.sum(state.F)*state.z)
    bottoms = jnp.where(state.z>0, f, 0)-state.distillate
    bottoms = jnp.where((bottoms<0) & (f>state.distillate), 0, bottoms)
    key = jnp.where(bottoms>0, jnp.arange(len(bottoms)), 1e2)
    bottoms = jnp.where((bottoms>0) & (bottoms==bottoms[jnp.array(jnp.min(key), dtype=int)]), bottoms, jnp.where(bottoms>0, state.z*jnp.sum(state.F), 0))
    tops = state.z*jnp.sum(state.F)-bottoms

    keys = jnp.where((jnp.arange(len(key))==jnp.min(key)-1) | (jnp.arange(len(key))==jnp.min(key)), jnp.arange(len(key)), -1)
    key_bottoms = jnp.where(keys>-1, bottoms, 0)
    f_bottoms = jnp.where(bottoms>0, state.z*f, 0)
    f_tops = jnp.where(tops > 0, state.z * f, 0)
    key_tops = jnp.where(keys>-1, tops, 0)
    hb = jnp.where(keys==-1, bottoms, 0)
    lb = jnp.where(keys==-1, tops, 0)
    '''
    
    def function(tempcalc):
        return 1 - jnp.sum(tops/vmap(thermodynamics.k_eq, in_axes=(None, 0, None))(tempcalc, state.components, state.pressure))
    result = Bisection(optimality_fun=function, lower=500., upper=100., check_bracket=False).run(500.).params
    x_tops = tops/vmap(thermodynamics.k_eq, in_axes=(None, 0, None))(result, state.components, state.pressure)

    #z_bottom = jnp.where(bottoms > 0, state.z, 0)
    #x_strip = vmap(lambda i: jnp.interp(jnp.arange(feedstage+1,state.Nstages,1), jnp.array([feedstage, state.Nstages]), jnp.array([z_bottom[i]/jnp.sum(z_bottom), bottoms[i]])))(jnp.arange(len(state.z)))
    #z_top = jnp.where(x_tops > 0, state.z, 0)
    #x_rec = vmap(lambda i: jnp.interp(jnp.arange(1,feedstage,1), jnp.array([0, feedstage]), jnp.array([x_tops[i], z_top[i]/jnp.sum(z_top)])))(jnp.arange(len(state.z)))
    dx = (bottoms-x_tops)/(state.Nstages+1)
    x = x_tops[:, None] + dx[:, None]*(jnp.arange(len(state.temperature)))
    #x = vmap(lambda i: jnp.interp(jnp.arange(0, state.Nstages, 1), jnp.array([0, state.Nstages]),
    #                                  jnp.array([x_tops[i], bottoms[i] ])))(jnp.arange(len(state.z)))

    #x = jnp.concatenate((x_tops[:, None], x_rec, x_strip, bottoms[:, None]), axis=1)
    #x = jnp.zeros((len(state.z), len(state.temperature))).at[:, 0:state.Nstages].set(x)
    
    d_bottoms = bottoms/(state.Nstages-feedstage)
    d_tops = x_tops/(state.Nstages+1)
    d_fb = state.z/(state.Nstages-feedstage)
    d_fd = state.z/(feedstage-1)
    x_mask = jnp.tile(jnp.arange(len(state.temperature)), (len(state.z), 1))
    x_hk = x_mask * d_bottoms[:, None] + (state.Nstages+1-x_mask) * d_fb[:, None]
    x_lk = (state.Nstages -1 - x_mask) * d_tops[:, None] + x_mask * d_bottoms[:, None] #+ x_mask * d_fd[:, None]
    x_hk = x_lk + (state.Nstages + 1 - x_mask) * d_fb[:, None]
    x_lk = jnp.where(x_lk > 0., x_lk, 0.)
    
    x = jnp.where(jnp.arange(len(state.temperature)) < feedstage+1, x_lk, x_hk/jnp.sum(x_hk, axis=0))*jnp.where(x_mask<state.Nstages, 1, 0)
    #x = (jnp.where(state.z[:, None] > 0, x_lk + x_hk, 0)*jnp.where(x_mask<state.Nstages, 1, 0))
    '''
    l_linear = (tops[:, None] + ((bottoms-tops))[:, None]/(state.Nstages-1)*jnp.arange(len(state.temperature)))*jnp.where(jnp.arange(len(state.temperature))<state.Nstages, 1, 0)
    strip_mask = jnp.where((jnp.arange(len(state.temperature)) >= feedstage) & (jnp.arange(len(state.temperature))<state.Nstages), 1, 0)
    l_keys_strip = (f_bottoms[:, None] + ((bottoms-f_bottoms))[:, None]/(state.Nstages-feedstage)*(jnp.arange(len(state.temperature))-feedstage))*strip_mask
    rec_mask = jnp.where((jnp.arange(len(state.temperature)) < feedstage) , 1, 0)
    l_keys_rec = (tops[:, None] + ((f_tops-tops))[:, None]/(feedstage)*(jnp.arange(len(state.temperature))))*rec_mask
    l = ((l_keys_strip + l_keys_rec)+l_linear)/2

    '''
    l_hb = jnp.where(jnp.arange(len(state.temperature)) >= feedstage, hb[:, None], 0)*jnp.where(jnp.arange(len(state.temperature))<state.Nstages+1, 1, 0)
    l_lb = jnp.where(jnp.arange(len(state.temperature)) <= feedstage, lb[:, None], 0) * jnp.where(
        jnp.arange(len(state.temperature)) < state.Nstages + 1, 1, 0)
    l = (l_keys+l_hb+l_lb)
    '''
    state = state.replace(X=jnp.nan_to_num(l/jnp.sum(l, axis=0)))
    state = functions.stage_temperature(state)
    #state = state.replace(temperature=state.temperature[0] + (state.temperature[state.Nstages-1]-state.temperature[0])/(state.Nstages-1)*jnp.arange(len(state.temperature)))
    temperature = jnp.where(jnp.arange(len(state.temperature))<feedstage, state.temperature[0], state.temperature[state.Nstages-1])
    #state = state.replace(temperature=temperature)
    t_range = functions.t_sat_solver(state.components, state.pressure)
    t_minmax = jnp.where(state.z > 0, t_range, 0)
    min_t = jnp.min(jnp.where((t_minmax > 0) & (t_minmax != jnp.max(t_minmax)), t_minmax, jnp.max(t_minmax) - 1))
    return state

def f_sol(state: State, tray_low, tray, tray_high, j):
    f_s = jacobian.f_vector_function(
        state,
        Tray(l=tray_low.l[:, j], v=tray_low.v[:, j], T=tray_low.T[j]),
        Tray(l=tray.l[:, j], v=tray.v[:, j], T=tray.T[j]),
        Tray(l=tray_high.l[:, j], v=tray_high.v[:, j], T=tray_high.T[j]),
        j,
    )
    return jnp.concatenate((jnp.asarray(f_s.H)[None], jnp.asarray(f_s.M), jnp.asarray(f_s.E)), axis=0)

def update_NR(state: State, a, b, c):
    tray_low, tray_high, tray = matrix_transforms.trays_func(state)
    a_new, b_new, c_new = jacobian.jacobian_func(state, tray_low, tray_high, tray)
    #a = jnp.where(state.EQU_iterations%4 == 3, a_new, a)
    #b = jnp.where(state.EQU_iterations%4 == 3, b_new, b)
    #c = jnp.where(state.EQU_iterations%4 == 3, c_new, c)
    a = a_new
    b = b_new
    c = c_new

    f = vmap(f_sol, in_axes=(None, None, None, None, 0))(state, tray_low, tray,
                                                             tray_high,
                                                             jnp.arange(len(tray.T)))
    dx = jnp.nan_to_num(functions.thomas(a, b, c, -1 * f, state.Nstages), nan=1e-20)  # .reshape(-1,1)
    dx_norm = jnp.where(jnp.max(dx)/jnp.max(state.L)>1, jnp.max(dx)/jnp.max(state.L), 1)
    dx = jnp.nan_to_num(dx/dx_norm, nan=1)
    '''
    x = GaussNewton(residual_fun=res_root).run(x_root, tray=tray, is_state=state).params
    #dx = jnp.nan_to_num( functions.thomas(a, b, c, -1 * f, state.Nstages)) # .reshape(-1,1)
    x_v = x[:, :len(state.components)].transpose()
    x_l = x[:, -len(state.components):].transpose()
    x_t = x[:, len(state.components)].transpose()
    
    state = state.replace(
        V=jnp.sum(x_v, axis=0),
        L=jnp.sum(x_l, axis=0),
        Y=jnp.nan_to_num(x_v / jnp.sum(x_v, axis=0), nan=1e-20),
        X=jnp.nan_to_num(x_l / jnp.sum(x_l, axis=0), nan=1e-20),
        temperature=x_t,
    )
    '''



    def min_res(t, state: State, tray, dx):
        dx_v = dx[:, :len(state.components)].transpose()
        dx_l = dx[:, -len(state.components):].transpose()
        dx_t = dx[:, len(state.components)].transpose()

        #dx_v_norm = jnp.where(jnp.nan_to_num(dx_v / tray.v, nan=1) > 5, jnp.nan_to_num(dx_v / tray.v, nan=1), 1)
        #dx_l_norm = jnp.where(jnp.nan_to_num(dx_l / tray.l, nan=1) > 5, jnp.nan_to_num(dx_l / tray.l, nan=1), 1)
        #dx_t_norm = jnp.where(jnp.nan_to_num(dx_t / tray.T, nan=1) > 5, jnp.nan_to_num(dx_t / tray.T, nan=1), 1)
        v_new = (tray.v + t * dx_v)
        l_new = (tray.l + t * dx_l)
        t_new = (tray.T + t * dx_t)


        v_new_final = jnp.where(v_new >= 0., v_new, tray.v
                                * jnp.exp(dx_v / jnp.where(tray.v > 0, tray.v, 1e-8)))
        #v_new_final = jnp.where(v_new_final >= 1.6 * jnp.max(tray.v, axis=0), 1.6 * tray.v, v_new_final)
        #v_new_final = jnp.where(v_new_final >= 5 * tray.v, v_new_final/6, v_new_final)
        l_new_final = jnp.where(l_new >= 0., l_new, tray.l
                                * jnp.exp(dx_l / jnp.where(tray.l > 0, tray.l, 1e-8)))
       # l_new_final = jnp.where(l_new_final >= 5 * tray.l, l_new_final/6, l_new_final)
        t_new_final = jnp.where(t_new >= state.temperature_bounds[1]+25.5,
                                state.temperature_bounds[1]+25.5,
                                jnp.where(t_new <= state.temperature_bounds[0] - 25.5, state.temperature_bounds[0] - 25.5,
                                          t_new)) * jnp.where(tray.T > 0, 1, 0)


        state = state.replace(
            V=jnp.sum(v_new_final, axis=0),
            L=jnp.sum(l_new_final, axis=0),
            Y=jnp.nan_to_num(v_new_final / jnp.sum(v_new_final, axis=0), nan=1e-15),
            X=jnp.nan_to_num(l_new_final / jnp.sum(l_new_final, axis=0), nan=1e-15),
            temperature=t_new_final,
            #temperature_bounds=jnp.array([jnp.min(jnp.where(t_new_final > 0, t_new_final, 1e3)), jnp.max(t_new_final)]),
        )

        tray_low, tray_high, tray = matrix_transforms.trays_func(state)
        f = vmap(f_sol, in_axes=(None, None, None, None, 0))(state, tray_low, tray,
                                                                 tray_high,
                                                                 jnp.arange(len(tray.T)))
        f = jnp.where(jnp.arange(len(f))[:, None]<state.Nstages-1, f, 0)
        state = state.replace(NR_residuals=jnp.nan_to_num(jnp.sum(f ** 2), nan=1e3),
                              dx=state.dx.at[state.NR_iterations, :].set(f.flatten()),
                              profile=state.profile.at[state.NR_iterations, :].set(state.temperature))

        return state


    #t_min = new_t = 1.1 - jnp.exp(-state.NR_iterations)
    t_min = 0.98;
    states = vmap(min_res, in_axes=(0, None, None, None))(jnp.arange(0.8, 1.2, 0.02)*t_min, state, tray, dx)
    result = states.NR_residuals
    new_t = jnp.max(jnp.where(result == jnp.min(result), jnp.arange(0.8, 1.2, 0.02)*t_min, 0))

    #new_state = min_res(new_t, state, tray, dx)
    #new_t = jnp.where(state.NR_residuals > new_state.NR_residuals, new_t, state.damping)
    #state = state.replace(damping=new_t)
    #new_t = 1.05 - jnp.exp(-0.3*state.NR_iterations)
    #new_t = 1
    state = min_res(new_t, state, tray, dx)
    
    return state


def cond_fn(params):
    state, a, b, c = params
    comps = jnp.sum(jnp.where(state.z > 0, 1, 0))
    cond = state.Nstages * (2 * comps + 1) * jnp.sum(state.F) * 1e-9
    return (state.NR_iterations < 100) & (state.NR_residuals > cond)


def body_fn(params):
    state, a, b, c = params
    state = update_NR(state, a, b, c)
    state = state.replace(NR_iterations=state.NR_iterations + 1)
    return state, a, b, c


def converge_column(state: State):
    tray_low, tray_high, tray = matrix_transforms.trays_func(state)
    state = state.replace(damping=0.1)
    a, b, c = jacobian.jacobian_func(state, tray_low, tray_high, tray)
    '''
    params = state, a, b, c
    params = body_fn(params)
    #params = body_fn(params)
    #params = body_fn(params)
    state, a, b, c = params
    '''
    params = state, a, b, c
    state, _, _, _ = (
        lax.while_loop(cond_fun=cond_fn, body_fun=body_fn,
                       init_val=params
                       )
    )
    params = state, a, b, c
    params = body_fn(params)
    return state


def condensor_duty(state: State):
    result = jnp.sum(
        state.V[1] * state.Y[:, 1] * thermodynamics.vapor_enthalpy(state.temperature[1], state.components)) + jnp.sum(
        state.F[0] * state.z * state.Hfeed[0]) - jnp.sum(
        (state.L[0]) * state.X[:, 0] * thermodynamics.liquid_enthalpy(state.temperature[0],
                                                                      state.components)) - jnp.sum(
        state.V[0] * state.Y[:, 0] * thermodynamics.vapor_enthalpy(state.temperature[0], state.components))

    return state.replace(CD=result)


def reboiler_duty(state: State):
    result = jnp.sum(state.F * state.Hfeed) - state.CD - state.L[state.Nstages - 1] * jnp.sum(
        state.X[:, state.Nstages - 1] * thermodynamics.liquid_enthalpy(state.temperature[state.Nstages - 1],
                                                                       state.components)) - state.V[0] * jnp.sum(
        state.Y[:, 0] * thermodynamics.vapor_enthalpy(state.temperature[0], state.components))
    CD = jnp.sum(
        vmap(thermodynamics.h_evap, in_axes=(None, 0))(state.temperature[0], state.components) * state.Y[:, 0]) * \
         state.V[0]
    return state.replace(RD=result,
                         CD=state.CD + CD)



def inside_simulation(state, nstages, feedstage, pressure, feed, z, distillate, rr, analytics=False, specs=False,
                      heavy_recovery=jnp.array(0.99, dtype=float), light_recovery=jnp.array(0.99, dtype=float)):

    state = initial_guess(state=state, nstages=nstages, feedstage=feedstage, pressure=pressure, feed=feed, z=z,
                          distillate=distillate, rr=rr, analytics=analytics, specs=specs, heavy_recovery=heavy_recovery,
                          light_recovery=light_recovery)

    state = split_guess(state, feedstage)
    state = functions.y_func(state)
    state = state.replace(temperature_bounds=jnp.asarray([state.temperature[0], state.temperature[state.Nstages-1]]))
    #state = initial_temperature(state)

    state = state.replace(Hfeed=jnp.where(state.F > 0, jnp.sum(thermodynamics.feed_enthalpy(state) * state.z), 0))
    #state = equimolar.bubble_point(state)
    state = equimolar.converge_equimolar(state)

    state = converge_column(state)
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
    state = costing.tac(state)

    state = state.replace(
        converged= jnp.asarray((state.NR_residuals < state.Nstages * (2 * jnp.sum(jnp.where(state.z > 0, 1, 0)) + 1) * jnp.sum(state.F) * 1e-9) &
                               (state.NR_iterations < 100)))
    state = state.replace(TAC=jnp.where(state.converged==1, state.TAC, 45/8000))
    '''
    state = state.replace(
        converged= jnp.asarray((state.BP_residuals < 0.1) &
                               (state.BP_iterations < 100)))
                               '''
    return state
