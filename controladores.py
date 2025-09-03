import numpy as np

# =============================================================================
# Classe Base para Controladores
# =============================================================================
class BaseController:
    """Define a interface padrão para todos os controladores."""
    def __init__(self, gains, params_sistema):
        self.gains = gains
        self.params_sistema = params_sistema

    def get_num_estados_internos(self):
        """Retorna o número de variáveis de estado internas do controlador."""
        return 0

    def calcular_torque_e_derivadas(self, t, y_mecanico, y_controlador, omega_alvo):
        """
        Calcula o torque de controle e as derivadas dos estados internos.
        Deve retornar uma tupla: (torque_de_controle, derivadas_dos_estados_internos)
        """
        raise NotImplementedError

# =============================================================================
# Controlador Proporcional-Integral (PI)
# =============================================================================
class PIController(BaseController):
    def get_num_estados_internos(self):
        return 1  # Apenas 1 estado: o erro integral de velocidade

    def calcular_torque_e_derivadas(self, t, y_mecanico, y_controlador, omega_alvo):
        omega_mesa = y_mecanico[1]
        erro_velocidade = omega_alvo - omega_mesa
        
        # O estado do controlador é o erro integral
        integral_erro = y_controlador[0]
        
        # A derivada do estado é o próprio erro de velocidade
        d_integral_erro_dt = erro_velocidade
        
        # Torque de controle
        torque = self.gains['Kp'] * erro_velocidade + self.gains['Ki'] * integral_erro
        
        return torque, [d_integral_erro_dt]

# =============================================================================
# Controlador SoftTorque
# =============================================================================
class SoftTorqueController(BaseController):
    def get_num_estados_internos(self):
        return 2  # Estados: [erro_posicao_modificado, torque_filtrado_Tc]

    def calcular_torque_e_derivadas(self, t, y_mecanico, y_controlador, omega_alvo):
        theta_mesa, omega_mesa = y_mecanico[0], y_mecanico[1]
        theta_prox, omega_prox = y_mecanico[2], y_mecanico[3]

        # Torque de contato medido na superfície
        k0 = self.params_sistema['k'][0]
        c0 = self.params_sistema['c'][0]
        T_contact = k0 * (theta_prox - theta_mesa) + c0 * (omega_prox - omega_mesa)

        # Estados internos do controlador
        erro_pos_mod_state = y_controlador[0]
        Tc_state = y_controlador[1]

        # Parâmetros do SoftTorque
        h = self.gains.get('h', 0.1)
        fc_hz = self.gains.get('fc', 0.2)
        fc_rads = fc_hz * 2 * np.pi # Frequência de corte (rad/s)

        # CORREÇÃO: Verificação de segurança para evitar divisão por zero
        if abs(fc_rads) < 1e-9:
            fc_rads = 1e-9

        # Dinâmica do filtro passa-baixa para Tc (Eq. 7)
        d_Tc_state_dt = -fc_rads * Tc_state + fc_rads * T_contact
        
        # Torque de alta frequência
        Tf = T_contact - Tc_state
        
        # Erro de velocidade modificado
        erro_vel_mod = (omega_alvo - omega_mesa) - h * Tf

        # Derivada do estado de erro de posição modificado (Eq. 6)
        d_erro_pos_mod_state_dt = (omega_alvo - omega_mesa) - (h / fc_rads) * d_Tc_state_dt

        # Lei de controle (Eq. 6)
        torque = self.gains['Kp'] * erro_vel_mod + self.gains['Ki'] * erro_pos_mod_state
        
        return torque, [d_erro_pos_mod_state_dt, d_Tc_state_dt]

# =============================================================================
# Controlador ZTorque
# =============================================================================
class ZTorqueController(BaseController):
    def get_num_estados_internos(self):
        return 3 # Estados: [z1_filtro, z2_filtro, integral_erro_pi]

    def calcular_torque_e_derivadas(self, t, y_mecanico, y_controlador, omega_alvo):
        theta_mesa, omega_mesa = y_mecanico[0], y_mecanico[1]
        theta_prox, omega_prox = y_mecanico[2], y_mecanico[3]

        # Torque de contato
        k0, c0 = self.params_sistema['k'][0], self.params_sistema['c'][0]
        T_contact = k0 * (theta_prox - theta_mesa) + c0 * (omega_prox - omega_mesa)

        # Estados internos do controlador
        z1, z2, integral_erro_pi = y_controlador

        # Parâmetros do ZTorque
        Z = self.gains.get('Z', 0.0001) # 1 / impedância característica
        thp = self.gains.get('thp', 1.0) # Constante de tempo high-pass
        tlp = self.gains.get('tlp', 0.1) # Constante de tempo low-pass
        
        # Segurança para evitar divisão por zero
        if abs(tlp) < 1e-9: tlp = 1e-9
        if abs(thp) < 1e-9: thp = 1e-9
        
        # Coeficientes do filtro band-pass de 2ª ordem H(s) = (b1*s) / (s^2 + a1*s + a0)
        a0 = 1 / (tlp * thp)
        a1 = (1 / tlp) + (1 / thp)
        b1 = 1 / thp # A saída do filtro é T_filtered = b1*dz1/dt = b1*z2
        
        # Saída do filtro (torque filtrado)
        T_filtered = Z * b1 * z2
        
        # Erro para o PI (Eq. 8)
        erro_pi = (omega_alvo - omega_mesa) - T_filtered
        
        # Lei de controle PI
        torque = self.gains['Kp'] * erro_pi + self.gains['Ki'] * integral_erro_pi
        
        # Derivadas dos estados do filtro band-pass
        d_z1_dt = z2
        d_z2_dt = -a0 * z1 - a1 * z2 + T_contact
        
        # Derivada do estado do integrador PI
        d_integral_erro_pi_dt = erro_pi
        
        return torque, [d_z1_dt, d_z2_dt, d_integral_erro_pi_dt]

# Dicionário para fácil seleção dos controladores
control_strategies = {
    'pi': PIController,
    'soft_torque': SoftTorqueController,
    'ztorque': ZTorqueController
}
