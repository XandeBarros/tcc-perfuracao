import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from controladores import control_strategies

# =============================================================================
# 1. BANCO DE PARÂMETROS FÍSICOS
# =============================================================================
parametros = {
    'first_param_set': {
        'RHO_ACO': 7850.0, 'E_MOD': 210e9, 'NU_POISSON': 0.33, 'L_DP': 2780.0, 
        'OD_DP': 0.1397, 'ID_DP': 0.1186, 'L_BHA': 400.0, 'OD_BHA': 0.2095, 
        'ID_BHA': 0.0714, 'J_MESA_fixo': 1000.0, 'J_BHA_fixo': None, 'G_MOD_fixo': None,
    },
    'second_param_set': {
        'RHO_ACO': 8010.0, 'L_DP': 3000.0, 'OD_DP': 0.0635 * 2, 'ID_DP': 0.0543 * 2,
        'J_MESA_fixo': 500.0, 'J_BHA_fixo': 394.0, 'G_MOD_fixo': 79.6e9,
    }
}
parametros_comuns = {'XI': 0.02}

# =============================================================================
# 2. FUNÇÃO PARA GERAR PARÂMETROS DO MODELO N-GDL
# =============================================================================
def gerar_parametros_sistema(n_gdl, p):
    if n_gdl < 2: raise ValueError("O número de GDL deve ser >= 2.")
    G_MOD = p.get('G_MOD_fixo') or p['E_MOD'] / (2 * (1 + p['NU_POISSON']))
    I_p_DP = (np.pi / 32) * (p['OD_DP']**4 - p['ID_DP']**4)
    k_total_DP = (G_MOD * I_p_DP) / p['L_DP']
    J_total_DP = p['RHO_ACO'] * I_p_DP * p['L_DP']
    J_vec = np.zeros(n_gdl)
    J_vec[0] = p['J_MESA_fixo']
    J_vec[-1] = p.get('J_BHA_fixo') or p['RHO_ACO'] * (np.pi / 32) * (p['OD_BHA']**4 - p['ID_BHA']**4) * p['L_BHA']
    if n_gdl > 2: J_vec[1:-1] = J_total_DP / (n_gdl - 2)
    else: J_vec[-1] += J_total_DP
    k_vec = np.full(n_gdl - 1, k_total_DP * (n_gdl - 1))
    c_vec = np.zeros(n_gdl - 1)
    for i in range(n_gdl - 1):
        J_equiv = J_total_DP / (n_gdl - 2)
        c_vec[i] = 2 * p['XI'] * np.sqrt(k_vec[i] * J_equiv)
    return {'J': J_vec, 'k': k_vec, 'c': c_vec}

# =============================================================================
# 3. MODELOS DE ATRITO
# =============================================================================
def atrito_karnopp(y, WOB, params, p_ativo_global):
    a1=0.026; a2=0.085; beta=0.098; delta=1e-4
    theta_penultimo, omega_penultimo = y[-4], y[-3]
    theta_broca, omega_broca = y[-2], y[-1]
    k_final, c_final = params['k'][-1], params['c'][-1]
    T_transmitido = k_final * (theta_penultimo - theta_broca) + c_final * (omega_penultimo - omega_broca)

    if np.abs(omega_broca) <= delta and np.abs(T_transmitido) <= a2*WOB:
        return T_transmitido
    elif np.abs(omega_broca) <= delta and np.abs(T_transmitido) > a2*WOB:
        return a2*WOB*np.sign(T_transmitido)
    else:
        return (a1 + (a2 - a1) * np.exp(-beta * np.abs(omega_broca))) * WOB * np.sign(omega_broca)
modelos_atrito = {'karnopp': atrito_karnopp}

# =============================================================================
# 4. FUNÇÃO DO MODELO GENERALIZADA (N-GDL com Controlador Genérico)
# =============================================================================
def modelo_Ndof_controlado(t, y, params, WOB, omega_alvo, atrito_func, p_ativo_global, controller):
    n_gdl = len(params['J'])
    n_estados_mecanicos = 2 * n_gdl
    y_mecanico, y_controlador = y[:n_estados_mecanicos], y[n_estados_mecanicos:]
    J, k, c = params['J'], params['k'], params['c']
    theta, omega = y_mecanico[0::2], y_mecanico[1::2]
    dydt_mecanico = np.zeros_like(y_mecanico)
    dydt_mecanico[0::2] = omega
    T_mesa, dydt_controlador = controller.calcular_torque_e_derivadas(t, y_mecanico, y_controlador, omega_alvo)
    wob_aplicado = WOB if t > 10 else 0
    T_bit = atrito_func(y_mecanico, wob_aplicado, params, p_ativo_global)
    for i in range(n_gdl):
        T_esq = k[i-1]*(theta[i-1]-theta[i]) + c[i-1]*(omega[i-1]-omega[i]) if i > 0 else 0
        T_dir = k[i]*(theta[i]-theta[i+1]) + c[i]*(omega[i]-omega[i+1]) if i < n_gdl - 1 else 0
        dydt_mecanico[2*i+1] = (T_mesa if i==0 else T_esq) - (T_bit if i==n_gdl-1 else T_dir)
        dydt_mecanico[2*i+1] /= J[i]
    return np.concatenate((dydt_mecanico, dydt_controlador))

# =============================================================================
# 5. FUNÇÃO DE EXECUÇÃO DA SIMULAÇÃO
# =============================================================================
def run_simulation(N_GDL, PARAM_SET_ATIVO, MODELO_DE_ATRITO_ATIVO, WOB_operacional, SRPM_operacional, 
                   CONTROLLER_TYPE, controller_gains, t_final=100, show_plot=True, 
                   initial_conditions={'theta0_rpm': 0, 'omega0_rpm': 0}):
    
    p_ativo = {**parametros[PARAM_SET_ATIVO], **parametros_comuns}
    system_params = gerar_parametros_sistema(N_GDL, p_ativo)

    print(f"Parâmetros do sistema: \n J:{system_params['J']} \n C:{system_params['c']} \n K:{system_params['k']}")

    controller = control_strategies[CONTROLLER_TYPE](controller_gains, system_params)
    omega_alvo_rads = SRPM_operacional * (2 * np.pi / 60)
    t_eval = np.linspace(0, t_final, int(t_final * 50))
    
    # --- IMPLEMENTAÇÃO DAS CONDIÇÕES INICIAIS ---
    y0_mecanico = np.zeros(2 * N_GDL)
    # Posições iniciais (theta) nos índices pares
    y0_mecanico[0::2] = initial_conditions.get('theta0_rpm', 0)
    # Velocidades iniciais (omega) nos índices ímpares
    omega0_rads = initial_conditions.get('omega0_rpm', 0) * (2 * np.pi / 60)
    y0_mecanico[1::2] = omega0_rads

    y0_controlador = np.zeros(controller.get_num_estados_internos())
    y0 = np.concatenate((y0_mecanico, y0_controlador))
    
    sol = solve_ivp(
        fun=modelo_Ndof_controlado, t_span=[0, t_final], y0=y0, 
        method='RK45', 
        t_eval=t_eval,
        max_step=0.1,
        args=(system_params, WOB_operacional, omega_alvo_rads, modelos_atrito[MODELO_DE_ATRITO_ATIVO], p_ativo, controller)
    )
    
    if show_plot:
        t, y = sol.t, sol.y
        omega_broca_rpm = y[2*N_GDL - 1] * (60 / (2*np.pi))
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(t, y[1]*(60/(2*np.pi)), label='Velocidade da Mesa', linestyle='--', color='gray', alpha=0.8)
        ax.plot(t, omega_broca_rpm, label=f'Velocidade da Broca - Controlador: {CONTROLLER_TYPE}', color='blue')
        ax.axhline(SRPM_operacional, color='red', linestyle=':', label=f'Alvo ({SRPM_operacional} RPM)')
        ax.set_title(f"Simulação ({N_GDL} GDL | Params: {PARAM_SET_ATIVO})", fontsize=16)
        ax.set_xlabel('Tempo (s)'); ax.set_ylabel('Velocidade Angular (RPM)'); ax.legend(); ax.grid(True)
        ax.set_ylim(bottom=min(-20, np.min(omega_broca_rpm)*1.1), top=np.max(omega_broca_rpm)*1.1)
        plt.tight_layout(); plt.show()
    return sol.t, sol.y

# =============================================================================
# 6. BLOCO DE EXECUÇÃO PRINCIPAL
# =============================================================================
if __name__ == '__main__':
    # Execute este arquivo para um teste único.
    run_simulation(
        N_GDL=3, 
        PARAM_SET_ATIVO='second_param_set', 
        MODELO_DE_ATRITO_ATIVO='karnopp',
        WOB_operacional=140e3, 
        SRPM_operacional=100,
        CONTROLLER_TYPE='pi', 
        controller_gains={'Kp': 600.0, 'Ki': 120.0},
        initial_conditions={'omega0_rpm': 70}, # Condição inicial de 70 RPM
        show_plot=True
    )
