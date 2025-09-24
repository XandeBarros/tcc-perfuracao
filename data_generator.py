import numpy as np
from scipy.stats import qmc
from tqdm import tqdm
import os
import stick_slip_state_space as simulador

# =============================================================================
# 1. CONFIGURAÇÃO DA GERAÇÃO DE DADOS
# =============================================================================
NUM_AMOSTRAS = 1000  # Número de simulações a serem executadas
PASTA_DADOS = "banco_de_dados_teste_Ks" # Pasta para salvar os resultados

# Definição das faixas de variação para cada um dos 9 parâmetros de entrada
# [min, max]
# faixas_parametros = {
#     'L_DP': [2500.0, 3500.0],    # Comprimento da coluna
#     'c1': [0.02, 0.035],         # Parâmetro de atrito regularizado
#     'c2': [0.07, 0.10],          # Parâmetro de atrito regularizado
#     'c3': [0.08, 0.12],          # Parâmetro de atrito regularizado
#     'c4': [8.0, 12.0],           # Parâmetro de atrito regularizado
#     'Kp': [500.0, 1500.0],       # Ganho Proporcional
#     'Ki': [100.0, 700.0],        # Ganho Integral
#     'WOB': [100e3, 180e3],       # Peso sobre a Broca
#     'SRPM': [80.0, 120.0]        # Rotação da Mesa
# }

faixas_parametros = {
    'Kp': [500.0, 1500.0],       # Ganho Proporcional
    'Ki': [100.0, 700.0],        # Ganho Integral
}

# Cria a pasta de dados se ela não existir
os.makedirs(PASTA_DADOS, exist_ok=True)

# =============================================================================
# 2. GERAÇÃO DE AMOSTRAS COM LATIN HYPERCUBE SAMPLING (LHS)
# =============================================================================
# Extrai os limites inferiores e superiores das faixas
limites_inf = [faixas_parametros[k][0] for k in faixas_parametros]
limites_sup = [faixas_parametros[k][1] for k in faixas_parametros]

# Inicializa o sampler LHS
sampler = qmc.LatinHypercube(d=len(faixas_parametros))
amostras_unitarias = sampler.random(n=NUM_AMOSTRAS)

# Mapeia as amostras (que estão entre 0 e 1) para as faixas de parâmetros
amostras_mapeadas = qmc.scale(amostras_unitarias, limites_inf, limites_sup)

print(f"Geradas {NUM_AMOSTRAS} combinações de parâmetros usando LHS.")

# =============================================================================
# 3. EXECUÇÃO DAS SIMULAÇÕES EM LOTE
# =============================================================================
for i in tqdm(range(NUM_AMOSTRAS), desc="Gerando dados"):
    parametros_entrada = amostras_mapeadas[i]
    
    # Mapeia os valores da amostra para os dicionários de parâmetros
    # (Isso é um pouco mais complexo porque os parâmetros estão em lugares diferentes)
    
    # Parâmetros de simulação física
    # Por simplicidade, vamos variar apenas L_DP aqui, mantendo o resto fixo
    # Em um caso real, você ajustaria a função do simulador para aceitar todos os parâmetros
    L_DP_atual = 3000
    # ... (código para modificar os parâmetros de atrito e outros seria necessário aqui)
    
    # Ganhos do controlador
    kp_atual = parametros_entrada[0]
    ki_atual = parametros_entrada[1]
    gains = {'Kp': kp_atual, 'Ki': ki_atual}
    
    # Parâmetros operacionais
    wob_atual = 140e3
    srpm_atual = 100
    
    # Nome do arquivo de saída
    nome_arquivo = os.path.join(PASTA_DADOS, f"amostra_{i:04d}.npz")
    
    # Pula se o arquivo já existir
    if os.path.exists(nome_arquivo):
        continue

    try:
        # Executa a simulação (com plotagem desativada)
        # NOTA: O simulador precisaria ser modificado para aceitar L_DP e os params de atrito como argumentos
        t, y = simulador.run_simulation(
            N_GDL=15, 
            PARAM_SET_ATIVO='second_param_set', 
            MODELO_DE_ATRITO_ATIVO='regularizado', # Usando o modelo suave para estabilidade
            WOB_operacional=wob_atual, 
            SRPM_operacional=srpm_atual,
            CONTROLLER_TYPE='pi', 
            controller_gains=gains,
            initial_conditions={'omega0_rpm': 70},
            show_plot=False 
        )

        # Extrai as 4 saídas de interesse
        omega_mesa = y[15] * (60 / (2*np.pi))
        omega_broca = y[2*15 - 1] * (60 / (2*np.pi))
        # ... (código para extrair torques) ...
        
        # Salva os resultados
        np.savez(
            nome_arquivo,
            parametros_entrada=parametros_entrada,
            tempo=t,
            omega_mesa=omega_mesa,
            omega_broca=omega_broca
            # ... (salvar torques) ...
        )

    except Exception as e:
        print(f"Erro na amostra {i}: {e}")

print("Geração de dados concluída!")
