import jax.numpy as jnp
from NR_model_test import thermodynamics
from NR_model_test.distillation_types import Tray, State
from jax import vmap, jacfwd
from NR_model_test.physical_data_sets import psat_params, cpvap_params, hvap_params, hform_params


def dEdlij(tray: Tray, components, index, pressure):
    return thermodynamics.k_eq(tray.T, components[index], pressure) * (
                1 / jnp.sum(tray.l) - tray.l[index] / jnp.sum(tray.l) ** 2)


def dEdlj(tray: Tray, components, index, pressure):
    return -thermodynamics.k_eq(tray.T, components[index], pressure) * (tray.l[index] / (jnp.sum(tray.l) ** 2))


def dEdvij(tray: Tray, components, index, pressure):
    return -(1 / jnp.sum(tray.v) - tray.v[index] / jnp.sum(tray.v) ** 2)


def dEdvj(tray: Tray, components, index, pressure):
    return tray.v[index] / (jnp.sum(tray.v) ** 2)


def dEdTj(tray: Tray, components, index, pressure):
    params = psat_params()[components[index]]
    a, b, c, d, e, f, g, h, i = params

    return tray.l[index] / (pressure * jnp.sum(tray.l)) * jnp.exp(
        a + f * tray.T ** g + d * tray.T + b / (c + tray.T) + e * jnp.log(tray.T)) * (
                d - b / (c + tray.T) ** 2 + e / tray.T + f * g * tray.T ** (g - 1))


def dMdlij(s):
    return 1 + s


def dMdvij(S):
    return 1 + S


def dhvapdT(T, component):
    params = hvap_params()[component]
    a, b, c, d, e, f, g = params
    T0 = 273.15
    Tc = g + T0
    result = a * (-T / Tc + 1) ** (b + c * T / Tc + d * T ** 2 / Tc ** 2 + e * T ** 3 / Tc ** 3) * (
                (c / Tc + 2 * d * T / (Tc ** 2) + 3 * e * T ** 2 / (Tc ** 3)) * jnp.log10(-T / Tc + 1) - (
                    b + c * T / Tc + d * T ** 2 / Tc ** 2 + e * T ** 3 / Tc ** 3) / (Tc * (-T / Tc + 1)))
    return jnp.where(T/Tc > 1, 0, result * 4.182)


def dhv_pure_dT(T, component):
    params = cpvap_params()[component]
    a, b, c, d, e, f, g = params
    return (a + b * ((c / T) / (jnp.sinh(c / T))) ** 2 + d * ((e / T) / (jnp.cosh(e / T))) ** 2) / 1000 * 4.184


def dhvdT(tray: Tray, component):
    params = cpvap_params()[component]
    a, b, c, d, e, f, g = params
    return dhv_pure_dT(tray.T, component)
    # return -(tray.l[component]*sum(tray.v)*((d * e ** 2 * (jnp.tanh(e/tray.T) ** 2-1))/tray.T**2
    #                                        - a + (b*c**2*(jnp.tanh(c/tray.T)**2-1))/(tray.T**2*jnp.tanh(c/tray.T)**2)))/jnp.sum(tray.l)


def dhldT(tray: Tray, component):
    return (dhv_pure_dT(tray.T, component) - dhvapdT(tray.T, component))


def dHdT(tray: Tray, components):
    return  jnp.sum(tray.l * vmap(dhldT, in_axes=(None, 0))(tray, components)) + jnp.sum(tray.v * vmap(dhvdT, in_axes=(None, 0))(tray, components))



def dHdT_A(tray: Tray, components):
    return -jnp.sum(tray.l * vmap(dhldT, in_axes=(None, 0))(tray, components))


def dHdT_C(tray: Tray, components):
    return -jnp.sum(tray.v * vmap(dhvdT, in_axes=(None, 0))(tray, components))


def dHdlij(tray: Tray, components, index):
    return (jnp.sum(thermodynamics.liquid_enthalpy(tray.T, components)) / jnp.sum(tray.l)
            - jnp.sum(tray.l) * (jnp.sum(thermodynamics.liquid_enthalpy(tray.T, components)) / jnp.sum(tray.l) ** 2
                                 - thermodynamics.liquid_enthalpy(tray.T, components)[index] / jnp.sum(
                        tray.l)))


def dHdlij1(tray: Tray, components, index):
    return thermodynamics.liquid_enthalpy(tray.T, components)[index] * tray.l[index]


def dHdvij(tray: Tray, components, index):
    return (jnp.sum(thermodynamics.vapor_enthalpy(tray.T, components)) / jnp.sum(tray.v)
            - jnp.sum(tray.v) * (jnp.sum(thermodynamics.vapor_enthalpy(tray.T, components)) / jnp.sum(tray.v) ** 2
                                 - thermodynamics.vapor_enthalpy(tray.T, components)[index] / jnp.sum(tray.v)))


def dHdvij1(tray: Tray, components, index):
    return thermodynamics.vapor_enthalpy(tray.T, components)[index] * tray.v[index]


def h_func(state: State, T, tray, j):
    result = jnp.where(j == 0, jnp.sum(tray.l) - state.RR * state.distillate, (
        jnp.where(j == state.Nstages - 1, jnp.sum(tray.l, axis=0) - (jnp.sum(state.f) - state.distillate), (
                jnp.sum(tray.l * thermodynamics.liquid_enthalpy(T, state.components), axis=0) + jnp.sum(
            tray.v * thermodynamics.vapor_enthalpy(T, state.components), axis=0)))))

    return result


def set_B(tray: Tray, state: State):
    h_vap = -jnp.average(thermodynamics.liquid_enthalpy(tray.T, state.components))

    b_mat = jnp.zeros((2 * len(tray.l) + 1, 2 * len(tray.l) + 1))
    b_mat = b_mat.at[0, 0:len(tray.v)].set(
        vmap(dHdvij, in_axes=(None, None, 0))(tray, state.components, jnp.arange(len(tray.v))) / (h_vap))
    b_mat = b_mat.at[0, len(tray.v) + 1:].set(
        vmap(dHdlij, in_axes=(None, None, 0))(tray, state.components, jnp.arange(len(tray.v))) / (h_vap))
    # b_mat = b_mat.at[0, len(tray.v)].set(jacfwd(h_func, argnums=1)(state, tray.T, tray, 2)/(h_vap*10))
    b_mat = b_mat.at[0, len(tray.v)].set(dHdT(tray, state.components) / (h_vap))
    b_mat = b_mat.at[1:len(tray.v) + 1, 0:len(tray.v)].set(jnp.eye(len(tray.v)))
    b_mat = b_mat.at[1:len(tray.v) + 1, len(tray.v) + 1:].set(jnp.eye(len(tray.v)))
    b_mat = b_mat.at[len(tray.v) + 1:, 0:len(tray.v)].set(
        jnp.eye(len(tray.v)) * vmap(dEdvij, in_axes=(None, None, 0, None))(tray, state.components,
                                                                           jnp.arange(len(tray.v)), state.pressure))
    b_mat = b_mat.at[len(tray.v) + 1:, len(tray.v) + 1:].set(
        jnp.eye(len(tray.v)) * vmap(dEdlij, in_axes=(None, None, 0, None))(tray, state.components,
                                                                           jnp.arange(len(tray.v)), state.pressure))
    b_mat = b_mat.at[len(tray.v) + 1:, len(tray.v)].set(
        vmap(dEdTj, in_axes=(None, None, 0, None))(tray, state.components, jnp.arange(len(tray.l)), state.pressure))
    b_mat = (b_mat.at[len(tray.v) + 1:, len(tray.v) + 1:]
             .set(jnp.where(jnp.eye(len(tray.l)) == 1, b_mat[len(tray.v) + 1:, len(tray.v) + 1:],
                            vmap(dEdlj, in_axes=(None, None, 0, None))(tray, state.components, jnp.arange(len(tray.l)),
                                                                       state.pressure)[:, None])))
    b_mat = (b_mat.at[len(tray.l) + 1:, 0:len(tray.l)]
             .set(jnp.where(jnp.eye(len(tray.l)) == 1, b_mat[len(tray.l) + 1:, 0:len(tray.l)],
                            vmap(dEdvj, in_axes=(None, None, 0, None))(tray, state.components, jnp.arange(len(tray.l)),
                                                                       state.pressure)[:, None])))
    return b_mat


def set_A(tray: Tray, state: State):
    h_vap = -jnp.average(thermodynamics.liquid_enthalpy(tray.T, state.components))

    a_mat = jnp.zeros((2 * len(tray.l) + 1, 2 * len(tray.l) + 1))
    a_mat = a_mat.at[0, len(tray.v) + 1:].set(
        -vmap(dHdlij, in_axes=(None, None, 0))(tray, state.components, jnp.arange(len(tray.v))) / (h_vap))
    # b_mat = b_mat.at[0, len(tray.v)].set(jacfwd(h_func, argnums=1)(state, tray.T, tray, 2))
    a_mat = a_mat.at[0, len(tray.v)].set(dHdT_A(tray, state.components) / (h_vap))

    a_mat = a_mat.at[1:len(tray.v) + 1, len(tray.v) + 1:].set(-jnp.eye(len(tray.v)))
    return a_mat


def set_C(tray: Tray, state: State):
    h_vap = -jnp.average(thermodynamics.liquid_enthalpy(tray.T, state.components))

    c_mat = jnp.zeros((2 * len(tray.l) + 1, 2 * len(tray.l) + 1))
    c_mat = c_mat.at[0, 0:len(tray.v)].set(
        -vmap(dHdvij, in_axes=(None, None, 0))(tray, state.components, jnp.arange(len(tray.v))) / (h_vap))
    # b_mat = b_mat.at[0, len(tray.v)].set(jacfwd(h_func, argnums=1)(state, tray.T, tray, 2))
    c_mat = c_mat.at[0, len(tray.v)].set(dHdT_C(tray, state.components) / (h_vap))

    c_mat = c_mat.at[1:len(tray.v) + 1, 0:len(tray.v)].set(-jnp.eye(len(tray.v)))
    return c_mat