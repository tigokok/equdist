import jax.numpy as jnp
from time import time
from jax import jit, vmap, lax, random

import matplotlib.pyplot as plt
from equdist.plot_generation import plot_function
from equdist.model import inside_simulation as simulation
from equdist.model import initialize
from equdist import purity_constraint
import seaborn as sns
plt.rcParams.update({'font.size': 13})

def random_entry_in_zeros_array(key, length=10):
    key = random.PRNGKey(key)
    key, key1, key2, key3, key4, key5, key6, key7, key8, key9  = random.split(key, 10)
    zeros_array = jnp.zeros(length)
    random_index = random.randint(key, (), 0, length - 7)
    random_value1 = random.uniform(key1, ())
    random_value2 = random.uniform(key2, ())
    random_value3 = random.uniform(key3, ())
    random_value4 = random.uniform(key4, ())
    random_value5 = random.uniform(key5, ())
    random_value6 = random.uniform(key6, ())
    random_value7 = random.uniform(key7, ())
    random_value8 = random.uniform(key8, ())
    random_value9 = random.uniform(key9, ())
    zeros_array = zeros_array.at[random_index].set(random_value1)
    zeros_array = zeros_array.at[random_index + 1].set(random_value2)
    zeros_array = zeros_array.at[random_index + 2].set(random_value3)
    zeros_array = zeros_array.at[random_index + 3].set(random_value4)
    zeros_array = zeros_array.at[random_index + 4].set(random_value5)
    zeros_array = zeros_array.at[random_index + 5].set(random_value6)
    zeros_array = zeros_array.at[random_index + 6].set(random_value7)
    #zeros_array = zeros_array.at[random_index + 7].set(random_value8)
    #zeros_array = zeros_array.at[random_index + 8].set(random_value9)

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
    i = i
    pressure = jnp.array(1., dtype=float)
    feed = jnp.array(1000., dtype=float)
    h = jnp.array(0.97, dtype=float)
    l = jnp.array(0.97, dtype=float)

    fug_state = jit(purity_constraint.FUG)(zf[i], h, l, pressure, feed)
    state_init = initialize()
    st = time()
    state = jit(simulation)(
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

    print(state.NR_iterations)
    duration = duration.at[i].set(time()-st)
    stages = stages.at[i].set(fug_state.stages)
    distillate = distillate.at[i].set(fug_state.distillate)
    reflux = reflux.at[i].set(fug_state.reflux)
    feed_stage = feed_stage.at[i].set(fug_state.feed_stage)
    iteration_array = iteration_array.at[i].set(state.NR_iterations)
    converged = converged.at[i].set(state.converged)
    carry = zf, stages, reflux, distillate, feed_stage



print(jnp.sum(jnp.where(iteration_array == 100, 1, 0)))
plot_function(jnp.arange(1, state.Nstages+1), state.L[0:state.Nstages], state.V[0:state.Nstages], state.temperature[0:state.Nstages], state.Hliq[0:state.Nstages], state.Hvap[0:state.Nstages], jnp.array(state.X*jnp.where(state.z != 0, 1, jnp.nan)[:, None])[:, 0:state.Nstages], jnp.array(state.Y*jnp.where(state.z != 0, 1, jnp.nan)[:, None])[:, 0:state.Nstages], jnp.where(state.z != 0, jnp.arange(0, len(state.z))+1, 0))

from matplotlib.animation import FuncAnimation



# Example data array of shape (50, 200)
data = state.dx  # Replace this with your actual data
print(state.EQU_residuals)
# x-axis values for plotting (assuming 200 data points per array)
x = jnp.arange(data.shape[1])

fig1, ax1 = plt.subplots()
line1, = ax1.plot(x, data[40], label="Array 0")
#ax1.set_ylim(-0.05, 0.05)  # Adjust based on your data range
ax1.set_title("Iterating Over Arrays")
ax1.legend()

# Set up the figure and axis
fig, ax = plt.subplots()

ax.set_ylim(-0.05, 0.05)  # Adjust based on your data range
#ax.set_ylim(jnp.min(data)-5, jnp.max(data)+5)  # Adjust based on your data range
ax.set_title("Iterating Over Arrays")

# Animation function
line, = ax.plot(x, data[0], label="Array 0")
#text = ax.text(0.5, 0.5, '', ha='center', va='center', fontsize=20, color='blue')
legend = ax.legend()
def init_anim():
    line.set_data(x,data[0])
    legend.set_title(f"Iteration: 0")
    return line, legend


def update(frame):
    line.set_ydata(data[frame])  # Update y-data with the current array
    line.set_label(f"Array {frame}")
    legend.set_title(f"Iteration: {frame}")
    return line, legend


# Create the animation
ani = FuncAnimation(fig, update, frames=data.shape[0], interval=200, blit=True, init_func=init_anim)

plt.show()
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


