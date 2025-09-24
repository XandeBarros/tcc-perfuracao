import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from controladores import control_strategies

# =============================================================================
# 1. BANCO DE PARÂMETROS FÍSICOS
# =============================================================================
parametros = {
    'second_param_set': {
        'RHO_ACO': 8010.0, 'L_DP': 3000.0, 'OD_DP': 0.0635 * 2, 'ID_DP': 0.0543 * 2,
        'J_MESA_fixo': 500.0, 'J_BHA_fixo': 394.0, 'G_MOD_fixo': 79.6e9,
    }
}
parametros_comuns = {'XI': 0.02, 'C_MUD_FACTOR': 0}

# =============================================================================
# 2. FUNÇÃO PARA MONTAR AS MATRIZES DO SISTEMA (M, C, K, A, B)
# =============================================================================
def montar_matrizes_sistema(n_gdl, p):
    if n_gdl < 2: raise ValueError("O número de GDL deve ser >= 2.")
    G_MOD = p['G_MOD_fixo']
    I_p_DP = (np.pi / 32) * (p['OD_DP']**4 - p['ID_DP']**4)
    k_total_DP = (G_MOD * I_p_DP) / p['L_DP']
    J_total_DP = p['RHO_ACO'] * I_p_DP * p['L_DP']
    
    J_vec = np.zeros(n_gdl)
    J_vec[0] = p['J_MESA_fixo'] + (J_total_DP / (n_gdl - 1)) / 2
    J_vec[-1] = p['J_BHA_fixo'] + (J_total_DP / (n_gdl - 1)) / 2
    if n_gdl > 2: J_vec[1:-1] = J_total_DP / (n_gdl - 1)
    else: J_vec[-1] += J_total_DP
    
    k_elem = k_total_DP * (n_gdl - 1)
    
    M = np.diag(J_vec)
    C_int = np.zeros((n_gdl, n_gdl))
    K = np.zeros((n_gdl, n_gdl))
    C_ext = np.diag(p['C_MUD_FACTOR'] * J_vec)

    for i in range(n_gdl - 1):
        J_equiv = J_total_DP / (n_gdl - 1) # (J_vec[i] * J_vec[i+1]) / (J_vec[i] + J_vec[i+1])
        c_elem = 2 * p['XI'] * np.sqrt(k_elem * J_equiv)
        K[i,i] += k_elem; K[i+1,i+1] += k_elem; K[i,i+1] -= k_elem; K[i+1,i] -= k_elem
        C_int[i,i] += c_elem; C_int[i+1,i+1] += c_elem; C_int[i,i+1] -= c_elem; C_int[i+1,i] -= c_elem

    C_total = C_int + C_ext

    M_inv = np.linalg.inv(M)
    A_state = np.zeros((2 * n_gdl, 2 * n_gdl))
    A_state[0:n_gdl, n_gdl:] = np.eye(n_gdl)
    A_state[n_gdl:, 0:n_gdl] = -M_inv @ K
    A_state[n_gdl:, n_gdl:] = -M_inv @ C_total
    
    B_table = np.zeros(2 * n_gdl)
    B_table[n_gdl] = M_inv[0, 0] 

    B_bit = np.zeros(2 * n_gdl)
    B_bit[2*n_gdl - 1] = M_inv[n_gdl-1, n_gdl-1]

    return {'M': M, 'C': C_total, 'K': K, 'A': A_state, 'B_table': B_table, 'B_bit': B_bit}

# =============================================================================
# 3. MODELOS DE ATRITO
# =============================================================================
def atrito_karnopp(y, WOB, matrizes):
    a1=0.026; a2=0.085; beta=0.098; delta=1e-4
    n_gdl = matrizes['M'].shape[0]
    theta_penultimo, omega_penultimo = y[n_gdl - 2], y[2*n_gdl - 2]
    theta_broca, omega_broca = y[n_gdl - 1], y[2*n_gdl - 1]
    k_final = -matrizes['K'][n_gdl-2, n_gdl-1]
    c_final = -matrizes['C'][n_gdl-2, n_gdl-1]
    T_transmitido = k_final * (theta_penultimo - theta_broca) + c_final * (omega_penultimo - omega_broca)
    if np.abs(omega_broca) <= delta and np.abs(T_transmitido) <= a2*WOB: return T_transmitido
    elif np.abs(omega_broca) <= delta and np.abs(T_transmitido) > a2*WOB: return a2*WOB*np.sign(T_transmitido)
    else: return (a1 + (a2-a1) * np.exp(-beta * np.abs(omega_broca))) * WOB * np.sign(omega_broca)

def atrito_regularizado(y, WOB, matrizes):
    """
    Implementa o Modelo M3 (Proposto Regularizado) do artigo COB-2025-1731.
    Equação (5): Tb = WOB * [(c2-c1)*exp(-c3*|ωb|) + c1] * tanh(c4*ωb)
    """
    c1 = 0.02692; c2 = 0.08689; c3 = 0.1051; c4 = 10.0013

    n_gdl = matrizes['M'].shape[0]
    omega_broca = y[2*n_gdl - 1] # Apenas a velocidade da broca é necessária

    # Componente Stribeck (decaimento exponencial)
    stribeck_part = (c2 - c1) * np.exp(-c3 * np.abs(omega_broca)) + c1
    
    # Componente de regularização (transição suave em zero)
    regularization_part = np.tanh(c4 * omega_broca)

    return WOB * stribeck_part * regularization_part

modelos_atrito = {
    'karnopp': atrito_karnopp,
    'regularizado': atrito_regularizado
}

# =============================================================================
# 4. FUNÇÃO DO MODELO MATRICIAL
# =============================================================================
def modelo_matricial(t, y, matrizes, WOB, omega_alvo, atrito_func, controller):
    n_gdl = matrizes['M'].shape[0]
    n_estados_mecanicos = 2 * n_gdl
    y_mecanico, y_controlador = y[:n_estados_mecanicos], y[n_estados_mecanicos:]
    
    dydt_linear = matrizes['A'] @ y_mecanico
    
    T_mesa, dydt_controlador = controller.calcular_torque_e_derivadas(t, y_mecanico, y_controlador, omega_alvo)
    
    # WOB agora é aplicado desde o início
    T_bit = atrito_func(y_mecanico, WOB, matrizes)
    
    dydt_mesa = matrizes['B_table'] * T_mesa
    dydt_bit = matrizes['B_bit'] * (-T_bit)
    
    dydt_mecanico = dydt_linear + dydt_mesa + dydt_bit
    
    return np.concatenate((dydt_mecanico, dydt_controlador))

# =============================================================================
# 5. FUNÇÃO DE EXECUÇÃO DA SIMULAÇÃO
# =============================================================================
def run_simulation(N_GDL, PARAM_SET_ATIVO, MODELO_DE_ATRITO_ATIVO, WOB_operacional, SRPM_operacional, 
                   CONTROLLER_TYPE, controller_gains, t_final=100, show_plot=True, 
                   initial_conditions={'theta0_rpm': 0, 'omega0_rpm': 0}):
    
    p_ativo = {**parametros[PARAM_SET_ATIVO], **parametros_comuns}
    matrizes = montar_matrizes_sistema(N_GDL, p_ativo)
    controller = control_strategies[CONTROLLER_TYPE](controller_gains)
    omega_alvo_rads = SRPM_operacional * (2 * np.pi / 60)
    t_eval = np.linspace(0, t_final, int(t_final * 50))
    
    y0_mecanico = np.zeros(2 * N_GDL)
    y0_mecanico[0:N_GDL] = initial_conditions.get('theta0_rpm', 0)
    omega0_rads = initial_conditions.get('omega0_rpm', 0) * (2 * np.pi / 60)
    y0_mecanico[N_GDL:] = omega0_rads
    
    y0_controlador = np.zeros(controller.get_num_estados_internos())
    y0 = np.concatenate((y0_mecanico, y0_controlador))
    
    sol = solve_ivp(
        fun=modelo_matricial, t_span=[0, t_final], y0=y0, 
        method='RK45', t_eval=t_eval, max_step=0.1,
        args=(matrizes, WOB_operacional, omega_alvo_rads, modelos_atrito[MODELO_DE_ATRITO_ATIVO], controller)
    )
    
    if show_plot:
        t, y = sol.t, sol.y
        
        torques_mesa, torques_bit = [], []
        n_estados_mecanicos = 2 * N_GDL
        for i in range(len(t)):
            y_mecanico_i, y_controlador_i = y[:n_estados_mecanicos, i], y[n_estados_mecanicos:, i]
            T_mesa_i, _ = controller.calcular_torque_e_derivadas(t[i], y_mecanico_i, y_controlador_i, omega_alvo_rads)
            torques_mesa.append(T_mesa_i)
            T_bit_i = modelos_atrito[MODELO_DE_ATRITO_ATIVO](y_mecanico_i, WOB_operacional, matrizes)
            torques_bit.append(T_bit_i)

        omega_mesa_rpm = y[N_GDL] * (60 / (2 * np.pi))
        omega_broca_rpm = y[2*N_GDL - 1] * (60 / (2*np.pi))
        
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        axs[0].plot(t, omega_mesa_rpm, label='Velocidade da Mesa', linestyle='--', color='gray', alpha=0.8)
        axs[0].plot(t, omega_broca_rpm, label='Velocidade da Broca', color='blue')
        axs[0].axhline(SRPM_operacional, color='red', linestyle=':', label=f'Alvo ({SRPM_operacional} RPM)')
        axs[0].set_title('Resposta de Velocidade do Sistema', fontsize=16)
        axs[0].set_ylabel('Velocidade Angular (RPM)'); axs[0].legend(); axs[0].grid(True)

        axs[1].plot(t, np.array(torques_mesa)/1000, label='Torque da Mesa (Controle)', color='green')
        axs[1].plot(t, np.array(torques_bit)/1000, label='Torque no Bit (Atrito)', color='purple', linestyle='-.')
        axs[1].set_title('Análise dos Torques', fontsize=16)
        axs[1].set_xlabel('Tempo (s)'); axs[1].set_ylabel('Torque (kNm)'); axs[1].legend(); axs[1].grid(True)

        fig.suptitle(f"Simulação ({N_GDL} GDL | Atrito: {MODELO_DE_ATRITO_ATIVO})", fontsize=20, weight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()
        
    return sol.t, sol.y

# =============================================================================
# 6. BLOCO DE EXECUÇÃO PRINCIPAL
# =============================================================================
if __name__ == '__main__':
    run_simulation(
        N_GDL=5, 
        PARAM_SET_ATIVO='second_param_set', 
        # Alterne entre 'karnopp' e 'regularizado' para comparar os modelos
        MODELO_DE_ATRITO_ATIVO='regularizado',
        WOB_operacional=140e3, 
        SRPM_operacional=100,
        CONTROLLER_TYPE='pi', 
        controller_gains={'Kp': 650.0, 'Ki': 175.0},
        initial_conditions={'omega0_rpm': 70}, 
        show_plot=True
    )
