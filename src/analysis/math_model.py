import random
import numpy as np

class MathematicalModel:
    """
    Modelo probabilístico para la DES:
    
    - Tiempos entre eventos: Distribución Exponencial
    - Resultado de tiros: Regresión Logística
    - Distribución de posesión: Proporcional a tasas de eventos
    
    Parámetros estocásticos:
    - attack_rate: λ para distribución exponencial de ataques
    - foul_rate: λ para distribución exponencial de faltas
    - shot_prob: Probabilidad base de convertir un tiro
    """
    
    @staticmethod
    def time_to_next_event(rate: float) -> float:
        """Genera tiempo hasta próximo evento usando distribución exponencial"""
        return random.expovariate(rate)
    
    @staticmethod
    def shot_outcome(attack_strength: float, defense_strength: float) -> bool:
        """Determina resultado de tiro usando función logística"""
        odds = (attack_strength - defense_strength) / 100
        prob = 1 / (1 + np.exp(-odds * 6))  # Escalado para rango razonable
        return random.random() < prob
