import jax.numpy as jnp
from time import time
from jax import jit, vmap, lax, random

import matplotlib.pyplot as plt
from jumanji.environments.distillation.NR_model_test.plot_generation import plot_function
from jumanji.environments.distillation.NR_model_test.NR_model import inside_simulation as simulation
from jumanji.environments.distillation.NR_model_test.NR_model import initialize
from jumanji.environments.distillation.NR_model_test.NR_model import purity_constraint
import seaborn as sns
plt.rcParams.update({'font.size': 13})

def random_entry_in_zeros_array(key, length=10):
    key = random.PRNGKey(key)
    key, key1, key2, key3, key4 = random.split(key, 5)
    zeros_array = jnp.zeros(length)
    random_index = random.randint(key, (), 0, length - 3)
    random_value1 = random.uniform(key1, ())
    random_value2 = random.uniform(key2, ())
    random_value3 = random.uniform(key3, ())
    random_value4 = random.uniform(key4, ())
    zeros_array = zeros_array.at[random_index].set(random_value1)
    zeros_array = zeros_array.at[random_index + 1].set(random_value2)
    zeros_array = zeros_array.at[random_index + 2].set(random_value3)
    zeros_array = zeros_array.at[random_index + 3].set(random_value4)
    zeros_array = zeros_array / jnp.sum(zeros_array)
    zeros_array = jnp.where((zeros_array < 0.07) & (zeros_array > 0), zeros_array + 0.07, zeros_array)
    zeros_array = zeros_array / jnp.sum(zeros_array)
    return zeros_array



iters = 1000
stages = jnp.zeros(iters, dtype=int)
distillate = jnp.zeros(iters, dtype=float)
reflux = jnp.zeros(iters, dtype=float)
feed_stage = jnp.zeros(iters, dtype=int)
iteration_array = jnp.zeros(iters, dtype=int)
converged = jnp.zeros(iters, dtype=bool)
duration = jnp.zeros(iters, dtype=float)
zf = vmap(jit(random_entry_in_zeros_array))(jnp.arange(iters))


#def scan_body(carry, i):
for i in range(iters):
    pressure = jnp.array(1., dtype=float)
    feed = jnp.array(1000., dtype=float)
    h = jnp.array(0.97, dtype=float)
    l = jnp.array(0.97, dtype=float)

    fug_state = jit(purity_constraint.FUG)(zf[i], h, l, pressure, feed)
    state_init = initialize()
    st = time()
    state, iterations, res = jit(simulation)(
        state=state_init,
        nstages=fug_state.stages,
        feedstage=fug_state.feed_stage,
        pressure=jnp.array(1.0, dtype=float),
        feed=jnp.array(1000.0, dtype=float),
        z=jnp.array(zf[i], dtype=float),
        distillate=fug_state.distillate,
        rr=fug_state.reflux,
        specs=False
    )

    print(iterations)
    duration = duration.at[i].set(time()-st)
    stages = stages.at[i].set(fug_state.stages)
    distillate = distillate.at[i].set(fug_state.distillate)
    reflux = reflux.at[i].set(fug_state.reflux)
    feed_stage = feed_stage.at[i].set(fug_state.feed_stage)
    iteration_array = iteration_array.at[i].set(iterations)
    converged = converged.at[i].set(state.converged)
    carry = zf, stages, reflux, distillate, feed_stage




x = jnp.arange(iters)
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
# First subplot
axs[0, 0].grid(True)
axs[0, 0].tick_params(axis='both', which='major', labelsize=20)
axs[0, 0].plot(x, stages)
axs[0, 0].set_title('Stages per evaluation', fontsize=26)
axs[0, 0].set_xlabel('iteration', fontsize=26)
axs[0, 0].set_ylabel('stages (-) ', fontsize=26)
axs[0, 0].set_ylim([0, 75])

# Second subplot
axs[0, 1].grid(True)
axs[0, 1].tick_params(axis='both', which='major', labelsize=20)
axs[0, 1].plot(x, reflux)
axs[0, 1].set_title('Reflux per evaluation', fontsize=26)
axs[0, 1].set_xlabel('iteration', fontsize=26)
axs[0, 1].set_ylabel('reflux ratio (-)', fontsize=26)
axs[0, 1].set_ylim([0, 15])

# Third subplot
axs[1, 0].grid(True)
axs[1, 0].tick_params(axis='both', which='major', labelsize=20)
axs[1, 0].plot(x, distillate)
axs[1, 0].set_title('Distillate per evaluation', fontsize=26)
axs[1, 0].set_xlabel('iteration', fontsize=26)
axs[1, 0].set_ylabel('distillate (kmol/hr)', fontsize=26)
axs[1, 0].set_ylim([0, 1000])

# Fourth subplot
axs[1, 1].grid(True)
axs[1, 1].tick_params(axis='both', which='major', labelsize=20)
axs[1, 1].plot(x, stages/feed_stage)
axs[1, 1].set_title('Stage/feed location ratio', fontsize=26)
axs[1, 1].set_xlabel('iteration', fontsize=26)
axs[1, 1].set_ylabel('stage/feed stage (-)', fontsize=26)
axs[1, 1].set_ylim([1, 3])

# Adjust spacing between subplots
plt.tight_layout()

x = jnp.arange(iters)
fig1, axs1 = plt.subplots(1, 2, figsize=(10, 8))
axs1[0].grid(True)
# First subplot

sns.histplot(iteration_array, stat='percent', ax=axs1[0], bins=30)
axs1[0].tick_params(axis='both', which='major', labelsize=20)
axs1[0].set_title('Iterations per evaluation', fontsize=26)
axs1[0].set_xlabel('iterations (-)', fontsize=26)
axs1[0].set_ylabel('%', fontsize=26)
axs1[0].set_xlim([0,40])


sns.histplot(duration[1:], stat='percent', ax=axs1[1], bins=30)
axs1[1].grid(True)
axs1[1].tick_params(axis='both', which='major', labelsize=20)
axs1[1].set_title('Duration per evaluation', fontsize=26)
axs1[1].set_xlabel('time (s)', fontsize=26)
axs1[1].set_ylabel('%', fontsize=26)
axs1[1].set_xlim([0,1])

'''
# Second subplot
sns.histplot(y=converged, stat='percent', ax=axs1[1], bins=jnp.array([-0.05, 0.05, 0.95, 1.05]))
axs1[1].set_title('Convergence check per evaluation')
axs1[1].set_xlabel('1/0')
axs1[1].set_ylabel('%')
axs1[1].set_ylim(-0.5, 1.5)
axs1[1].set_yticks([0, 1])
'''
# Third subplot


plt.show()
