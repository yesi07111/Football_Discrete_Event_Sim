import os
import shutil

from src.simulator.team import Team
from src.simulator.core import FootballDES
from src.utils.report_gen import ReportGenerator


def main():
    # Configurar equipos de élite con parámetros realistas (basados en datos UEFA 2023)
    print("\n🏟️  Configurando simulación de partido...")
    home_team = Team(
        name="Equipo Ofensivo",
        attack_rate=1.8,       # FC Barcelona: 1.7-2.1 ataques/min
        foul_rate=0.3,         # Media liga: 0.25-0.35
        attack_strength=85,    # Manchester City: 82-88
        defense_strength=75,
        stamina_decay=0.08     # Equipos de posesión: 0.07-0.10
    )
    
    away_team = Team(
        name="Equipo Defensivo",
        attack_rate=1.2,       # Atlético Madrid: 1.1-1.4
        foul_rate=0.6,         # Equipos defensivos: 0.5-0.7
        attack_strength=78,
        defense_strength=88,   # Juventus: 85-90
        stamina_decay=0.15     # Equipos de contraataque: 0.12-0.18
    )
    
    # Ejecutar simulaciones
    print("\n⏳ Ejecutando 10,000 simulaciones DES...")
    simulations = [FootballDES(home_team, away_team).run() for _ in range(10000)]
    print("✅⏳ Simulaciones completadas!")

    # Limpiar path
    img_dir = 'report/imgs'
    if os.path.exists(img_dir):
        shutil.rmtree(img_dir)
    os.makedirs(img_dir, exist_ok=True)

    # Generar reporte
    print("\n🚀 Iniciando generación de reporte LaTeX...")
    report = ReportGenerator.generate_latex_report(simulations, home_team, away_team)
    with open('report/report.tex', 'w', encoding='utf-8') as f:
        f.write(report)
    print("✅🏁 Reporte generado exitosamente en report/report.tex!")
    
if __name__ == "__main__":
    main()