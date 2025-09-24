import numpy as np

class BaseController:
    def __init__(self, gains):
        self.gains = gains
        self.torque_limit = 50000.0

    def get_num_estados_internos(self):
        return 0

    def calcular_torque_e_derivadas(self, t, y_mecanico, y_controlador, omega_alvo):
        raise NotImplementedError

class PIController(BaseController):
    def get_num_estados_internos(self):
        return 1

    def calcular_torque_e_derivadas(self, t, y_mecanico, y_controlador, omega_alvo):
        n_gdl = len(y_mecanico) // 2
        omega_mesa = y_mecanico[n_gdl]
        
        erro_velocidade = omega_alvo - omega_mesa
        integral_erro = y_controlador[0]
        d_integral_erro_dt = erro_velocidade
        
        torque = self.gains['Kp'] * erro_velocidade + self.gains['Ki'] * integral_erro
        
        # Lógica de Anti-Windup: Essencial para estabilidade numérica do controlador
        if (torque >= self.torque_limit and erro_velocidade > 0) or \
           (torque <= -self.torque_limit and erro_velocidade < 0):
            d_integral_erro_dt = 0

        torque_saturado = np.clip(torque, -self.torque_limit, self.torque_limit)
        
        return torque_saturado, [d_integral_erro_dt]

control_strategies = {'pi': PIController}

