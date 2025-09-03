import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import simulador_stick_slip as simulador

# =============================================================================
# 1. CONFIGURAÇÃO DA ANÁLISE
# =============================================================================
CONTROLLER_ATIVO = 'soft_torque' # Opções: 'pi', 'soft_torque', 'ztorque'

gains_config = {
    'pi': {
        'nominais': {'Kp': 1000.0, 'Ki': 500.0},
        'incertezas': {'Kp': 0.3, 'Ki': 0.3}
    },
    'soft_torque': {
        'nominais': {'Kp': 1000.0, 'Ki': 500.0, 'h': 0.1, 'fc': 0.2},
        'incertezas': {'h': 0.5, 'fc': 0.5} 
    },
    'ztorque': {
        'nominais': {'Kp': 1000.0, 'Ki': 500.0, 'Z': 0.0001, 'thp': 1.0, 'tlp': 0.1},
        'incertezas': {'Z': 0.5, 'thp': 0.3, 'tlp': 0.3}
    }
}

NUM_SIMULACOES = 50
N_GDL_sim = 15
PARAM_SET_sim = 'second_param_set'
MODELO_ATRITO_sim = 'karnopp'
WOB_sim = 140e3
SRPM_sim = 100
T_FINAL_sim = 100

# =============================================================================
# 2. EXECUÇÃO DO LOOP DE MONTE CARLO
# =============================================================================
print(f"Iniciando análise de Monte Carlo para o controlador: {CONTROLLER_ATIVO}")
config_atual = gains_config[CONTROLLER_ATIVO]
gains_nominais = config_atual['nominais']
incertezas = config_atual['incertezas']
gains_samples = []
for _ in range(NUM_SIMULACOES):
    sample = gains_nominais.copy()
    for gain_name, incert in incertezas.items():
        valor_nominal = gains_nominais[gain_name]
        # CORREÇÃO: Garante que o valor mínimo seja um número pequeno e positivo (1e-6)
        # para evitar a divisão por zero, em vez de apenas 0.
        sample[gain_name] = max(1e-6, np.random.normal(valor_nominal, valor_nominal * incert))
    gains_samples.append(sample)

t_nominal, y_nominal = simulador.run_simulation(
    N_GDL_sim, PARAM_SET_sim, MODELO_ATRITO_sim, WOB_sim, SRPM_sim,
    CONTROLLER_ATIVO, gains_nominais, t_final=T_FINAL_sim, show_plot=False
)
omega_broca_nominal_rpm = y_nominal[2*N_GDL_sim - 1] * (60 / (2 * np.pi))

resultados_broca = []
for gains in tqdm(gains_samples, desc="Simulando"):
    t, y = simulador.run_simulation(
        N_GDL_sim, PARAM_SET_sim, MODELO_ATRITO_sim, WOB_sim, SRPM_sim,
        CONTROLLER_ATIVO, gains, t_final=T_FINAL_sim, show_plot=False
    )
    omega_broca_rpm = y[2*N_GDL_sim - 1] * (60 / (2 * np.pi))
    resultados_broca.append(omega_broca_rpm)
print("Análise de Monte Carlo concluída.")

# =============================================================================
# 3. PÓS-PROCESSAMENTO E ANÁLISE ESTATÍSTICA
# =============================================================================
resultados_matrix = np.array(resultados_broca)
mean_response = np.mean(resultados_matrix, axis=0)
std_dev_response = np.std(resultados_matrix, axis=0)
conf_interval_upper = mean_response + 1.96 * std_dev_response
conf_interval_lower = mean_response - 1.96 * std_dev_response

# =============================================================================
# 4. VISUALIZAÇÃO DOS RESULTADOS
# =============================================================================
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6))
ax.fill_between(t_nominal, conf_interval_lower, conf_interval_upper, color='lightgray', alpha=0.8, label='95% Intervalo de Confiança')
ax.plot(t_nominal, omega_broca_nominal_rpm, 'r--', label='Resposta Nominal')
ax.plot(t_nominal, mean_response, 'b-', label='Resposta Média')
ax.set_title(f'Análise de Robustez do Controlador {CONTROLLER_ATIVO.upper()} ({NUM_SIMULACOES} Amostras)', fontsize=16)
ax.set_xlabel('Tempo (s)'); ax.set_ylabel('Velocidade Angular da Broca (rpm)'); ax.legend(); ax.grid(True); ax.set_ylim(bottom=0)
plt.tight_layout(); plt.show()
