from matplotlib import pyplot as plt
import numpy as np

def Enthalpy_temperature(component):
    T_array = np.linspace(273, 373)
    H_Larray = np.zeros(len(T_array))
    H_Varray = np.zeros(len(T_array))
    for i in range(0, len(components)):
        for j in range(0, len(T_array)):
            H_Larray[j] = (Hform[i] + self.Cp_vap(i, T_array[j]) - self.H_evap(i, T_array[j])) / 4.184
            H_Varray[j] = (Hform[i] + self.Cp_vap(i, T_array[j])) / 4.184

        plt.plot(T_array - 273.15, H_Larray)
        plt.plot(T_array - 273.15, H_Varray)
    plt.xlabel('Temperature (C)')
    plt.ylabel('Enthalpy (KJ/mole')
    plt.title('Temperature dependency of enthalpy (C5)')
    plt.grid(True)
    # plt.xticks(np.arange(290, T_array[-1] + 273.15, 10))
    # plt.yticks(np.arange(round(min(H_Larray)), round(max(H_Varray)) + 2, 1))

    # Show the plot
    plt.show()


def plot_function(stages, L, V, temperature, H_liq, H_vap, xnorm, y, components):
    fig, axs = plt.subplots(2, 3, figsize=(12, 4))

    # Plot data on each subplot
    axs[0, 0].plot(stages, L)
    axs[0, 0].set_title('Liquid flow ')
    axs[0, 0].set_xlabel('Stages')
    axs[0, 0].set_ylabel('Mole flow (kmol/hr)')
    axs[0, 0].grid(True)
    axs[0, 0].set_xticks(np.arange(1, len(stages)))

    axs[0, 1].plot(stages, V)
    axs[0, 1].set_xlabel('Stages')
    axs[0, 1].set_ylabel('Mole flow (kmol/hr)')
    axs[0, 1].set_title('Vapor flow')
    axs[0, 1].grid(True)
    axs[0, 1].set_xticks(np.arange(1, len(stages)))

    axs[0, 2].plot(stages, temperature - 273.15)
    axs[0, 2].set_xlabel('Stages')
    axs[0, 2].set_ylabel('Temperature (C)')
    axs[0, 2].set_title('Stage temperature')
    axs[0, 2].grid(True)
    axs[0, 2].set_xticks(np.arange(1, len(stages)))

    axs[1, 0].plot(stages, np.array(H_liq) / 4.184)
    axs[1, 0].set_title('Liquid enthalpy')
    axs[1, 0].set_xlabel('Stages')
    axs[1, 0].set_ylabel('Molar enthalpy (kj/mol)')
    axs[1, 0].grid(True)
    axs[1, 0].set_xticks(np.arange(1, len(stages)))

    axs[1, 1].plot(stages, np.array(H_vap) / 4.184)
    axs[1, 1].set_title('Vapor enthalpy')
    axs[1, 1].set_xlabel('Stages')
    axs[1, 1].set_ylabel('Molar enthalpy (kj/mol)')
    axs[1, 1].grid(True)
    axs[1, 1].set_xticks(np.arange(1, len(stages)))

    for xc in range(0, len(components)):
        axs[1, 2].plot(stages, xnorm[xc, :], label=components[xc]) #, label=[key for key, value in components.items() if value == xc]

    axs[1, 2].set_title('Composition')
    axs[1, 2].set_xlabel('Stages')
    axs[1, 2].set_ylabel('Composition(-)')
    axs[1, 2].set_ylim(0, 1)
    axs[1, 2].grid(True)
    axs[1, 2].legend()
    axs[1, 2].set_xticks(np.arange(1, len(stages)))

    # Adjust layout
    plt.tight_layout()
    #plt.savefig(f'pictures/{iter}.png')
    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(111)
    # calculate mole fractions
    for i in range(0, len(components)):
        ax2.plot(stages, temperature - 273.15, label=i)
    ax2.set_ylabel('Liquid phase mole fraction')
    ax2.set_xlabel('Stage Number')
    ax2.legend()
    ax2.grid()

    fig3 = plt.figure(3)
    ax3 = fig3.add_subplot(111)

    for i in range(0, len(components)):
        ax3.plot(stages, L, label='x_matrix')
    #   ax3.plot(stages, xnormalized[i, :], label='Aspen', linestyle=':')
    ax3.set_ylabel('Liquid phase mole fraction')
    ax3.set_xlabel('Stage Number')
    ax3.legend()
    ax3.grid()