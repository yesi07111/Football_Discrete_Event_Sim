import heapq
import random

from typing import Dict
from src.simulator.team import Team
from src.simulator.events import SimulationEvent
from src.analysis.math_model import MathematicalModel

class FootballDES:
    """Motor principal de simulación de eventos discretos"""
    
    def __init__(self, home: Team, away: Team):
        home.validate_parameters()
        away.validate_parameters()
        self.teams = {'home': home, 'away': away}        
        self.state = {
            'clock': 0.0,
            'possession': None,
            'goals': {'home': 0, 'away': 0},
            'shots': {'home': 0, 'away': 0},
            'fouls': {'home': 0, 'away': 0},
            'possession_time': {'home': 0.0, 'away': 0.0},
            'stamina': {'home': 100.0, 'away': 100.0}
        }
        self.event_queue = []
        self.event_counter = 0
        self._initialize_simulation()
    
    def _initialize_simulation(self):
        """Inicializa la simulación con primer evento"""
        # Empezar con saque inicial aleatorio
        self.state['possession'] = random.choice(['home', 'away'])
        self._schedule_next_event()
    
    def _schedule_next_event(self):
        """Programa el próximo evento según el estado actual"""
        current_team = self.state['possession']
        team = self.teams[current_team]
        
        # Calcular tasa efectiva considerando stamina
        effective_attack = team.attack_rate * (self.state['stamina'][current_team] / 100)
        effective_foul = team.foul_rate * (1 + (1 - self.state['stamina'][current_team] / 100))
        total_rate = effective_attack + effective_foul
        
        if total_rate <= 0:
            raise ValueError("Tasa de eventos inválida")
        
        # Generar tiempo para próximo evento
        interval = MathematicalModel.time_to_next_event(total_rate)
        next_time = self.state['clock'] + interval
        
        # Determinar tipo de evento
        event_prob = effective_attack / total_rate
        event_type = 'shot' if random.random() < event_prob else 'foul'
        
        # Agregar a la cola de prioridad
        heapq.heappush(
            self.event_queue,
            SimulationEvent(
                time=next_time,
                counter=self.event_counter,
                event_type=event_type,
                team=current_team
            )
        )
        self.event_counter += 1
    
    def _update_stamina(self, duration: float):
        """Actualiza la stamina por tiempo de posesión"""
        current_team = self.state['possession']
        decay = self.teams[current_team].stamina_decay * duration
        self.state['stamina'][current_team] = max(
            40.0, 
            self.state['stamina'][current_team] - decay
        )
    
    def _switch_possession(self):
        """Cambia la posesión al equipo contrario"""
        self.state['possession'] = 'away' if self.state['possession'] == 'home' else 'home'
    
    def _process_shot(self, team: str):
        """Procesa un evento de tiro al arco"""
        attacker = self.teams[team]
        defender = self.teams['away'] if team == 'home' else self.teams['home']

        self.state['shots'][team] += 1

        if MathematicalModel.shot_outcome(
            attacker.attack_strength, 
            defender.defense_strength
        ):
            self.state['goals'][team] += 1

        self._switch_possession() 
    
    def _process_foul(self, team: str):
        """Procesa un evento de falta"""
        self.state['fouls'][team] += 1
        # Faltas resultan en cambio de posesión
        self._switch_possession()
    
    def run(self, match_duration: float = 90.0) -> Dict:
        """Ejecuta la simulación hasta completar el tiempo del partido"""
        event_times = []
        while self.event_queue:
            current_event = heapq.heappop(self.event_queue)
            
            # Verificar si el evento supera el tiempo del partido
            if current_event.time > match_duration:
                break
            
            # Actualizar reloj y tiempo de posesión
            event_times.append(current_event.time)
            time_delta = current_event.time - self.state['clock']
            self.state['clock'] = current_event.time
            self.state['possession_time'][current_event.team] += time_delta
            self._update_stamina(time_delta)
            
            # Procesar evento
            if current_event.event_type == 'shot':
                self._process_shot(current_event.team)
            elif current_event.event_type == 'foul':
                self._process_foul(current_event.team)
            
            # Programar próximo evento
            self._schedule_next_event()
        
        # Asegurar tiempo total del partido
        if self.state['clock'] < match_duration:
            remaining_time = match_duration - self.state['clock']
            self.state['possession_time'][self.state['possession']] += remaining_time
            self.state['clock'] = match_duration
        
        self.state['event_times'] = event_times
        return self.state

