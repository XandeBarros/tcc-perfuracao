import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import simulador_stick_slip as simulador

# =============================================================================
# 1. CONFIGURAÇÃO DA ANÁLISE
# =============================================================================
CONTROLLER_ATIVO = 'pi' # Opções: 'pi', 'soft_torque', 'ztorque'

gains_config = {
    'pi': {
        'nominais': {'Kp': 600.0, 'Ki': 120.0},
        'incertezas': {'Kp': 0.3, 'Ki': 0.3}
    }
}

NUM_SIMULACOES = 50
N_GDL_sim = 5
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
        # para evitar a divisão por zero e instabilidades (como h < 0).
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
    
    # Adiciona uma verificação para garantir que a simulação foi bem-sucedida
    if y.shape[1] == len(t_nominal): # Verifica se o array de resultado tem o tamanho esperado
        omega_broca_rpm = y[2*N_GDL_sim - 1] * (60 / (2 * np.pi))
        # Verifica se há NaNs ou Infs no resultado antes de adicionar
        if not np.any(np.isnan(omega_broca_rpm)) and not np.any(np.isinf(omega_broca_rpm)):
            resultados_broca.append(omega_broca_rpm)

print(f"Análise de Monte Carlo concluída. Simulações bem-sucedidas: {len(resultados_broca)}/{NUM_SIMULACOES}")

# =============================================================================
# 3. PÓS-PROCESSAMENTO E ANÁLISE ESTATÍSTICA
# =============================================================================
if len(resultados_broca) > 1:
    resultados_matrix = np.array(resultados_broca)
    mean_response = np.mean(resultados_matrix, axis=0)
    std_dev_response = np.std(resultados_matrix, axis=0)
    conf_interval_upper = mean_response + 1.96 * std_dev_response
    conf_interval_lower = mean_response - 1.96 * std_dev_response
else:
    print("Não há simulações suficientes para plotar o intervalo de confiança.")
    # Define valores padrão para evitar erros no plot
    mean_response = omega_broca_nominal_rpm
    conf_interval_lower = omega_broca_nominal_rpm
    conf_interval_upper = omega_broca_nominal_rpm


# =============================================================================
# 4. VISUALIZAÇÃO DOS RESULTADOS
# =============================================================================
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6))
if len(resultados_broca) > 1:
    ax.fill_between(t_nominal, conf_interval_lower, conf_interval_upper, color='lightgray', alpha=0.8, label='95% Intervalo de Confiança')

ax.plot(t_nominal, omega_broca_nominal_rpm, 'r--', label='Resposta Nominal')
if len(resultados_broca) > 1:
    ax.plot(t_nominal, mean_response, 'b-', label='Resposta Média')

ax.set_title(f'Análise de Robustez do Controlador {CONTROLLER_ATIVO.upper()} ({len(resultados_broca)} Amostras Válidas)', fontsize=16)
ax.set_xlabel('Tempo (s)'); ax.set_ylabel('Velocidade Angular da Broca (rpm)'); ax.legend(); ax.grid(True); ax.set_ylim(bottom=0)
plt.tight_layout(); plt.show()
