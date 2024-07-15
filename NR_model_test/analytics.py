import jax.numpy as jnp
import Distillation.Newton_Raphson_directory.functions as functions
import Distillation.Newton_Raphson_directory.energy_functions as energy_functions
from Distillation.Newton_Raphson_directory.distillation_types import Tray, State
from jax import vmap, jacfwd
from jax import jit

params_psat, params_Cpvap, params_Hvap, Hform = functions.retrieve_params()

def dEdlij(tray: Tray, components, index, pressure):
    return functions.k_eq(tray.T, components[index], pressure) * (1/jnp.sum(tray.l) - tray.l[index]/jnp.sum(tray.l) ** 2)


def dEdlj(tray: Tray, components, index, pressure):
    return -functions.k_eq(tray.T, components[index], pressure)*(tray.l[index]/(jnp.sum(tray.l)**2))


def dEdvij(tray: Tray, components, index, pressure):
    return -(1/jnp.sum(tray.v) - tray.v[index]/jnp.sum(tray.v) ** 2)


def dEdvj(tray: Tray, components, index, pressure):
    return tray.v[index] / (jnp.sum(tray.v) ** 2)



def dEdTj(tray: Tray, components, index, pressure):
    params = params_psat[components[index]]
    a, b, c, d, e, f, g, h, i = params

    return tray.l[index]/(pressure*jnp.sum(tray.l))*jnp.exp(a + f*tray.T**g + d*tray.T + b/(c+tray.T) + e*jnp.log(tray.T))*(d - b/(c + tray.T) ** 2 + e/tray.T + f*g*tray.T**(g-1))


def dMdlij(s):
    return 1+s


def dMdvij(S):
    return 1+S


def dhvapdT(T, component):
    params = params_Cpvap[component]
    a, b, c, d, e, f, g = params
    T0 = 273.15
    #Tr = T/(T0+g)
    #result = jnp.where(Tr<1, -a*((1-Tr)**(e*Tr**3 + d*Tr**2 + c*Tr + b-1)*(e*Tr**3 + d*Tr**2 + c*Tr +b)-jnp.log(1-Tr)*(1-Tr)**(e*Tr**3+d*Tr**3+c*Tr+b)*(3*e*Tr**2+2*d*Tr+c)), 0)
    #result = jnp.where(Tr < 1, -a*(((1-Tr)**(b+(d*Tr**2)+(e*Tr**3)+(c*Tr)-1)*(b+(d*Tr**2)+(e*Tr**3)+(c*Tr)))/(T0+g)-jnp.log(1-Tr)*(1-Tr)*(1-Tr)**(b+d*Tr**2+(e*Tr**3)**3+(c*Tr))*(c/(T0+g)+(3*e*T**2)/(T0+g)**3+(2*d*T)/(T0+g)**2)), 0)
    '''
    term1 = (1 - T / (T0 + g)) ** (
                b + (d * T ** 2) / (T0 + g) ** 2 + (e * T ** 3) / (T0 + g) ** 3 + (c * T) / (T0 + g) - 1)
    term2 = b + (d * T ** 2) / (T0 + g) ** 2 + (e * T ** 3) / (T0 + g) ** 3 + (c * T) / (T0 + g)
    term3 = T0 + g
    term4 = jnp.log(1 - T / term3)
    term5 = (1 - T / term3) ** (b + (d * T ** 2) / term3 ** 2 + (e * T ** 3) / term3 ** 3 + (c * T) / term3)
    term6 = c / term3 + (3 * e * T ** 2) / term3 ** 3 + (2 * d * T) / term3 ** 2

    result = -a * ((term1 * term2) / term3 - term4 * term5 * term6)*4.182

    
    return jnp.where(T/(T0 + g) < 1, -a*((1-T/(T0+g)) ** (b+d*T**2/(T0+g)**2 + e*T**3/(T0 + g)**3 + c*T/(T0+g))/(T0 +g)
               - jnp.log(1 - T/(T0 + g))*(1 - (T**2/(T0+g)**2 + e*T**3/(T0+g)**3 + c*T/(T0+g)))
               *(c/(T0+g) + 3*e*T**2/(T0+g)**3 + 2*d*T/(T0+g)**2))* 4.184, 0)
    '''
    Tc=g+273.15
    result = a*(-T/Tc + 1)**(b+c*T/Tc + d*T**2/Tc**2+c*T**3/Tc**3)*((c/Tc+2*d*T/(Tc**2)+3*e*T**2/(Tc**3))*jnp.log(-T/Tc+1)-(b+c*T/Tc+d*T**2/Tc**2+e*T**3/Tc**3)/(Tc*(-T/Tc+1)))
    return result*4.182

def dhv_pure_dT(T, component):
    params = params_Cpvap[component]
    a, b, c, d, e, f, g = params
    return (a + b * ((c / T) / (jnp.sinh(c / T))) ** 2 + d * ((e / T) / (jnp.cosh(e / T))) ** 2)/ 1000 * 4.184


def dhvdT(tray: Tray, component):
    params = params_Cpvap[component]
    a, b, c, d, e, f, g = params
    return dhv_pure_dT(tray.T, component)/1000
    #return -(tray.l[component]*sum(tray.v)*((d * e ** 2 * (jnp.tanh(e/tray.T) ** 2-1))/tray.T**2
    #                                        - a + (b*c**2*(jnp.tanh(c/tray.T)**2-1))/(tray.T**2*jnp.tanh(c/tray.T)**2)))/jnp.sum(tray.l)


def dhldT(tray: Tray, component):
    return (dhv_pure_dT(tray.T, component)-dhvapdT(tray.T, component))/1000


def dHdT(tray: Tray):
    return (jnp.sum(tray.l*vmap(dhldT, in_axes=(None, 0))(tray, jnp.arange(len(tray.l))))
            + jnp.sum(tray.v*vmap(dhvdT, in_axes=(None, 0))(tray, jnp.arange(len(tray.v)))))/1000


def dHdT_A(tray: Tray):
    return -jnp.sum(tray.l*vmap(dhldT, in_axes=(None, 0))(tray, jnp.arange(len(tray.l))))/1000


def dHdT_C(tray: Tray):
    return -jnp.sum(tray.v*vmap(dhvdT, in_axes=(None, 0))(tray, jnp.arange(len(tray.v))))/1000


def dHdlij(tray: Tray, components, index):
    return (jnp.sum(energy_functions.liquid_enthalpy(tray.T, components))/jnp.sum(tray.l)
            - jnp.sum(tray.l)*(jnp.sum(energy_functions.liquid_enthalpy(tray.T, components))/jnp.sum(tray.l) ** 2
            - energy_functions.liquid_enthalpy(tray.T, components)[index]/jnp.sum(tray.l)))/1000


def dHdlij1(tray: Tray, components, index):
    return energy_functions.liquid_enthalpy(tray.T, components)[index]*tray.l[index]


def dHdvij(tray: Tray, components, index):
    return (jnp.sum(energy_functions.vapor_enthalpy(tray.T, components))/jnp.sum(tray.v)
            - jnp.sum(tray.v)*(jnp.sum(energy_functions.vapor_enthalpy(tray.T, components))/jnp.sum(tray.v) ** 2
            - energy_functions.vapor_enthalpy(tray.T, components)[index]/jnp.sum(tray.v)))/1000


def dHdvij1(tray: Tray, components, index):
    return energy_functions.vapor_enthalpy(tray.T, components)[index]*tray.v[index]



def h_func(state: State, T, tray, j):

    result = jnp.where(j == 0, jnp.sum(tray.l) - state.RR * state.distillate, (
        jnp.where(j == state.Nstages - 1, jnp.sum(tray.l, axis=0) - (jnp.sum(state.f) - state.distillate), (
                jnp.sum(tray.l * energy_functions.liquid_enthalpy(T, state.components), axis=0) + jnp.sum(
            tray.v * energy_functions.vapor_enthalpy(T, state.components), axis=0)))))

    return result


def set_B(tray: Tray, state: State):
    h_vap = jnp.min(vmap(energy_functions.h_evap, in_axes=(None, 0))(tray.T, state.components))

    b_mat = jnp.zeros((2*len(tray.l)+1,2*len(tray.l)+1))
    b_mat = b_mat.at[0, 0:len(tray.v)].set(
        vmap(dHdvij, in_axes=(None, None, 0))(tray, state.components, jnp.arange(len(tray.v)))/(h_vap))
    b_mat = b_mat.at[0, len(tray.v)+1:].set(
        vmap(dHdlij, in_axes=(None, None, 0))(tray, state.components, jnp.arange(len(tray.v)))/(h_vap))
    #b_mat = b_mat.at[0, len(tray.v)].set(jacfwd(h_func, argnums=1)(state, tray.T, tray, 2)/(h_vap*10))
    b_mat = b_mat.at[0, len(tray.v)].set(dHdT(tray)/(h_vap))
    b_mat = b_mat.at[1:len(tray.v)+1, 0:len(tray.v)].set(jnp.eye(len(tray.v))/1000)
    b_mat = b_mat.at[1:len(tray.v) + 1, len(tray.v)+1:].set(jnp.eye(len(tray.v))/1000)
    b_mat = b_mat.at[len(tray.v) + 1:, 0:len(tray.v)].set(jnp.eye(len(tray.v))*vmap(dEdvij, in_axes=(None, None, 0, None))(tray, state.components, jnp.arange(len(tray.v)), state.pressure))
    b_mat = b_mat.at[len(tray.v) + 1:, len(tray.v)+1:].set(jnp.eye(len(tray.v))*vmap(dEdlij, in_axes=(None, None, 0, None))(tray, state.components, jnp.arange(len(tray.v)), state.pressure))
    b_mat = b_mat.at[len(tray.v) + 1:, len(tray.v)].set(vmap(dEdTj, in_axes=(None, None, 0, None))(tray, state.components, jnp.arange(len(tray.l)), state.pressure))
    b_mat = (b_mat.at[len(tray.v) + 1:, len(tray.v)+1:]
             .set(jnp.where(jnp.eye(len(tray.l)) == 1, b_mat[len(tray.v) + 1:, len(tray.v)+1:],
                            vmap(dEdlj, in_axes=(None, None, 0, None))(tray, state.components, jnp.arange(len(tray.l)), state.pressure))))
    b_mat = (b_mat.at[len(tray.l) + 1:, 0:len(tray.l)]
             .set(jnp.where(jnp.eye(len(tray.l)) == 1, b_mat[len(tray.l) + 1:, 0:len(tray.l)],
                            vmap(dEdvj, in_axes=(None, None, 0, None))(tray, state.components, jnp.arange(len(tray.l)), state.pressure))))
    return b_mat


def set_A(tray: Tray, state: State):
    h_vap = jnp.min(vmap(energy_functions.h_evap, in_axes=(None, 0))(tray.T, state.components))

    a_mat = jnp.zeros((2*len(tray.l)+1,2*len(tray.l)+1))
    a_mat = a_mat.at[0, len(tray.v)+1:].set(
        -vmap(dHdlij, in_axes=(None, None, 0))(tray, state.components, jnp.arange(len(tray.v)))/(h_vap))
    #b_mat = b_mat.at[0, len(tray.v)].set(jacfwd(h_func, argnums=1)(state, tray.T, tray, 2))
    a_mat = a_mat.at[0, len(tray.v)].set(dHdT_A(tray)/(h_vap))

    a_mat = a_mat.at[1:len(tray.v) + 1, len(tray.v)+1:].set(-jnp.eye(len(tray.v))/1000)
    return a_mat


def set_C(tray: Tray, state: State):
    h_vap = jnp.min(vmap(energy_functions.h_evap, in_axes=(None, 0))(tray.T, state.components))

    c_mat = jnp.zeros((2*len(tray.l)+1,2*len(tray.l)+1))
    c_mat = c_mat.at[0, 0:len(tray.v)].set(
        -vmap(dHdvij, in_axes=(None, None, 0))(tray, state.components, jnp.arange(len(tray.v)))/(h_vap))
    #b_mat = b_mat.at[0, len(tray.v)].set(jacfwd(h_func, argnums=1)(state, tray.T, tray, 2))
    c_mat = c_mat.at[0, len(tray.v)].set(dHdT_C(tray)/(h_vap))

    c_mat = c_mat.at[1:len(tray.v)+1, 0:len(tray.v)].set(-jnp.eye(len(tray.v))/1000)
    return c_mat
