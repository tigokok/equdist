import jax.numpy as jnp
from NR_model_test.distillation_types import Tray, Trays, State
from NR_model_test import thermodynamics
from NR_model_test.analytics import set_B, set_A, set_C
from jax import vmap, jacfwd

def single_tray(tray, j):
    return Tray(l=tray.l[:, j], v=tray.v[:, j], T=tray.T[j])


def single_B(tray, j, state):
    result = set_B(single_tray(tray, j), state)
    first_tray = jnp.zeros(2*len(tray.v)+1).at[-len(tray.v):].set(1/state.RR).at[:len(tray.v)].set(-1)
    return jnp.where(j==0, result.at[0].set(first_tray),  jnp.where(j==state.Nstages-1, result.at[0].set(jnp.zeros(len(first_tray)).at[-len(tray.v):].set(-1)), set_B(single_tray(tray, j), state)))


def single_A(tray, j, state):
    result = set_A(single_tray(tray, j), state)
    return jnp.where(j==state.Nstages-2, result.at[0, len(tray.v):].set(0), result)


def single_C(tray, j, state):
    result = set_C(single_tray(tray, j), state)
    return jnp.where(j==0, result.at[0, 0:len(tray.v)+1].set(0), result)

def jacobian(state):
    new_B = vmap(single_B, in_axes=(None, 0, None))(state.trays.tray, jnp.arange(len(state.temperature)), state)
    new_A = vmap(single_A, in_axes=(None, 0, None))(state.trays.tray, jnp.arange(len(state.temperature)-1), state)
    new_C = vmap(single_C, in_axes=(None, 0, None))(state.trays.tray, jnp.arange(len(state.temperature)-1), state)
    mask = jnp.where(jnp.arange(len(state.L)) < state.Nstages, 1, 0)
    mask_small = jnp.where(jnp.arange(len(state.L)-1) < state.Nstages, 1, 0)
    new_B = jnp.where(mask[:, None, None], new_B, jnp.zeros(new_B.shape))
    new_A = jnp.where(mask_small[:, None, None], new_A, jnp.zeros(new_A.shape))
    new_C = jnp.where(mask_small[:, None, None], new_C, jnp.zeros(new_C.shape))
    return new_A, new_B, new_C


def h_function(state: State, tray_low, tray, tray_high, j, temp):
    tray = tray._replace(T=tray.T[j], l=tray.l[:, j], v=tray.v[:, j])
    h_evap = thermodynamics.h_evap(tray.T, 4)
    h_vap = jnp.average(thermodynamics.liquid_enthalpy(tray.T, state.components))
    #-(jnp.sum(tray.v) - jnp.sum(tray.l) / state.RR)
    cond_spec = jnp.where(state.light_key>0,
                          (tray.v[state.light_key]-state.light_spec*jnp.sum(state.F)*state.z[state.light_key]),
                          -(jnp.sum(tray.v) - jnp.sum(tray.l) / state.RR)
                          )
    reb_spec = jnp.where(state.heavy_key>0,
                         (tray.l[state.heavy_key]-state.heavy_spec * jnp.sum(state.F)*state.z[state.heavy_key]),
                         -(jnp.sum(tray.l) - (jnp.sum(state.F) - state.distillate))
                         )

    result = jnp.where(j == 0, cond_spec, (
        jnp.where(j == state.Nstages - 1, reb_spec, ((
                jnp.sum(tray.l * thermodynamics.liquid_enthalpy(temp, state.components)) + jnp.sum(
            tray.v * thermodynamics.vapor_enthalpy(tray.T, state.components)))/(-h_vap)))))
    return result

