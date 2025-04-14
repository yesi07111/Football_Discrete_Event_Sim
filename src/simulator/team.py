from dataclasses import dataclass

@dataclass
class Team:
    """Representa un equipo con parámetros de simulación"""
    name: str
    attack_rate: float    # Eventos por minuto
    foul_rate: float      # Eventos por minuto
    attack_strength: float  # 0-100
    defense_strength: float # 0-100
    stamina_decay: float = 0.1  # Porcentaje de reducción por minuto de posesión
    
    def __post_init__(self):
        self.validate_parameters()
    
    def validate_parameters(self):
        """Valida rangos de parámetros"""
        if any(rate < 0 for rate in [self.attack_rate, self.foul_rate]):
            raise ValueError("Las tasas deben ser positivas")
        if not 0 <= self.attack_strength <= 100:
            raise ValueError("Fuerza de ataque debe estar entre 0-100")
        if not 0 <= self.defense_strength <= 100:
            raise ValueError("Fuerza defensiva debe estar entre 0-100")
        
    def copy(self):
        """Crea una copia independiente del equipo"""
        return Team(
            name=self.name,
            attack_rate=self.attack_rate,
            foul_rate=self.foul_rate,
            attack_strength=self.attack_strength,
            defense_strength=self.defense_strength,
            stamina_decay=self.stamina_decay
        )