import jax.numpy as jnp
from NR_model_test.distillation_types import State

def diameter(state: State, l_s):
    r_gas = 8.314
    mw_series = jnp.array([16.04, 30.07, 44.097, 58.12, 72.15, 72.15, 100.21, 114.23])
    rho_series = jnp.array([422.6, 544.0, 493.0, 625.0, 626.0, 616.0, 670.0, 690.0])
    #mw_series = jnp.array([58.12, 72.15, 114.23])
    #rho_series = jnp.array([625.0, 626.0, 690.0])
    rho_v = state.pressure * 1e5 / (r_gas * state.temperature)*jnp.sum(state.Y*mw_series[:, None], axis=0)/1000
    rho_l = jnp.sum(state.X*rho_series[:, None], axis=0)
    u_sup = (-0.171 * l_s**2 + 0.27 * l_s - 0.047)*((rho_l-rho_v)/rho_v)**0.5
    #d_eff = jnp.sqrt(4*jnp.max(state.V/3.6*state.Y*mw_series[:, None]/1000)/(jnp.pi*rho_v*u_sup))
    return jnp.max(jnp.nan_to_num(jnp.sqrt(4 * jnp.max(state.V / 3.6 * state.Y * mw_series[:, None] / 1000) / (jnp.pi * rho_v * u_sup))))


def column_instalcost(state: State, ms):
    spacing = 0.609
    d_column = diameter(state, spacing)
    sump = 4
    height = spacing*state.Nstages+sump
    f_m = 1
    f_p = 1
    column_cost = ms/280*937.64*d_column**1.066*height**0.802*f_m*f_p

    f_s = 1.4
    f_t = 0
    f_m = 0
    internals_cost = ms/280*97.24*d_column**1.55*height*(f_s + f_t + f_m)
    return column_cost+internals_cost


def hex_instalcost(state: State, ms):
    f_p = 0
    f_d = 0.8
    f_m = 1

    t_in = 30+273
    t_out = 40+273
    #log_temp = ((state.temperature[0] - t_in) * (state.temperature[0] - t_out) * ((state.temperature[0] - t_in) + (state.temperature[0] - t_out)) / 2) ** (3 / 2)
    log_temp = 15.
    k_cd = 500.
    area_cd = state.CD/0.0036/(k_cd*log_temp)
    cost_cd = ms/280*474.67*area_cd**0.65*(f_p+f_d)*f_m

    k_rb = 800
    t_steam = 250+273
    area_rb = jnp.abs(state.RD/0.0036)/(k_rb*(t_steam-state.temperature[state.Nstages-1]))
    cost_rb = ms/280*474.67*area_rb**0.65*(f_p+f_d)*f_m
    return cost_cd + cost_rb


def installed_cost(state: State):
    ms = 2571.6
    cost_eqp = (column_instalcost(state, ms) + hex_instalcost(state, ms))/1e6
    f_lang = 5
    return cost_eqp*f_lang


def operational_cost(state: State):
    uptime = 8000
    e_cd = uptime*state.CD/1000
    e_cost = 2
    cd_cost = e_cd*e_cost/1e6

    e_rb = uptime*jnp.abs(state.RD)/1000
    e_cost = 40
    rb_cost = e_rb*e_cost/1e6
    return cd_cost + rb_cost

def tac(state: State):
    depreciation = 5  #years
    lang = 5
    return state.replace(TAC=(installed_cost(state)**2/depreciation * lang + operational_cost(state)*1.1** jnp.pow(10,(jnp.pow(0.5, jnp.max(state.X[:, 0]))-0.5))**10))
