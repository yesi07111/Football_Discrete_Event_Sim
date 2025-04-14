import os
import datetime
import itertools
import warnings
import numpy as np
import pandas as pd
import seaborn as sns

from scipy import stats
from scipy.spatial import KDTree
from typing import Dict, List, Tuple
from matplotlib import pyplot as plt

from src.simulator.team import Team
from src.simulator.core import FootballDES
from src.analysis.stats import SimulationAnalyzer

from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.operators.mutation.pm import PM
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.sampling.rnd import FloatRandomSampling

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from SALib.analyze import sobol
from SALib.sample import saltelli

from sklearn.linear_model import LogisticRegression

class SensitivityAnalyzer:
    """Realiza análisis de sensibilidad avanzados."""
    
    def __init__(self, base_team: Team, away_team: Team, simulations: List[Dict], original_stats: Dict, num_simulations=10000, mcmc_warmup=20,
    mcmc_samples=50):
        self.base_team = base_team
        self.away_team = away_team
        self.num_simulations = num_simulations
        self.base_sim_results = simulations
        self.base_stats = original_stats[1]
        self.parameter_descriptions = {
            'attack_rate': 'Frecuencia de ataques (ataques por minuto)',
            'foul_rate': 'Frecuencia de faltas (faltas por minuto)',
            'attack_strength': 'Efectividad ofensiva (0-100)',
            'defense_strength': 'Efectividad defensiva (0-100)',
            'stamina_decay': 'Desgaste físico (% de reducción por minuto de posesión)'
        }
        self.mcmc_warmpup = mcmc_warmup
        self.mcmc_samples = mcmc_samples
    
    def _modify_team(self, param: str, value: float, team: Team = None) -> Team:
        """Crea una copia modificada del equipo con el parámetro ajustado"""
        original = team if team else self.base_team
        return Team(
            name=original.name,
            attack_rate=value if param == 'attack_rate' else original.attack_rate,
            foul_rate=value if param == 'foul_rate' else original.foul_rate,
            attack_strength=value if param == 'attack_strength' else original.attack_strength,
            defense_strength=value if param == 'defense_strength' else original.defense_strength,
            stamina_decay=value if param == 'stamina_decay' else original.stamina_decay
        )
    
    def _run_simulations(self, team: Team, opponent: Team = None) -> List[Dict]:
        """Ejecuta múltiples simulaciones y captura parámetros de equipos"""
        simulations = []
        if opponent is None:
            opponent = self.away_team
        for _ in range(self.num_simulations):
            # Crear nueva instancia para cada simulación
            des = FootballDES(team.copy(), opponent.copy())
            result = des.run()
            
            # Guardar parámetros de ambos equipos
            result['home_params'] = des.teams['home'].__dict__.copy()
            result['away_params'] = des.teams['away'].__dict__.copy()
            
            simulations.append(result)
        return simulations
    
    # Métodos principales de análisis
    def analyze_single_parameter(self, param: str, values: List[float], target_team: str = 'home') -> List[Dict]:
        """Analiza variaciones en un solo parámetro para cualquier equipo"""
        original_value = getattr(self.base_team if target_team == 'home' else self.away_team, param)
        team_name = "Local" if target_team == 'home' else "Visitante"
        
        explanation = (
            f"Análisis de sensibilidad para parámetro {param} ({self.parameter_descriptions[param]}) en equipo {team_name}:\n"
            f"• Valor original: {original_value:.2f}\n"
            f"• Valores probados: {', '.join(map(str, values))}\n"
            f"• Equipo contrario mantuvo configuración base\n\n"
            "Estructura del resultado (lista de diccionarios):\n"
            "- parameter: Nombre completo del parámetro modificado (equipo_param)\n"
            "- value: Valor específico probado\n"
            "- stats: Dict con:\n"
            "   • base_stats: Estadísticas descriptivas originales\n"
            "   • modified_stats: Estadísticas con modificación\n"
            "   • comparison: Comparativa detallada (diferencia media, tamaño efecto, p-valor)\n"
            "- performance: Dict con:\n"
            "   • time_metrics: Eventos por periodo temporal\n"
            "   • momentum: Tendencia de actividad\n"
            "   • correlations: Matriz de correlaciones\n"
            "   • win_probability: Probabilidad de victoria"
        )
        
        results = []
        for value in values:
            if target_team == 'home':
                modified = self._modify_team(param, value, self.base_team)
                sim_results = self._run_simulations(modified, self.away_team)
                team_ref = modified
            else:
                modified = self._modify_team(param, value, self.away_team)
                sim_results = self._run_simulations(self.base_team, modified)
                team_ref = self.base_team

            results.append({
                'parameter': f"{target_team}_{param}",
                'value': value,
                'stats': self.calculate_statistics(sim_results, team_ref),
                'performance': self.calculate_performance_metrics(sim_results, target_team, [param])
            })
        return explanation, results

    def analyze_multi_parameters(self, param_specs: List[Tuple[str, str]], values: List[Tuple], target_team: str = 'home') -> List[Dict]:
        """Analiza combinaciones de múltiples parámetros en diferentes equipos"""
        params_info = []
        for team, param in param_specs:
            original = getattr(self.base_team if team == 'home' else self.away_team, param)
            params_info.append(
                f"• {param} ({self.parameter_descriptions[param]}) en equipo {'Local' if team == 'home' else 'Visitante'}: "
                f"Valor original {original:.2f}"
            )
        
        explanation = (
            "Análisis de múltiples parámetros simultáneos:\n"
            + "\n".join(params_info) + "\n"
            "Combinaciones probadas:\n" +
            "\n".join([f"- {combo}" for combo in values]) + "\n\n"
            "Estructura del resultado (lista de diccionarios):\n"
            "- parameters: Lista de tuplas (equipo, parámetro) modificados\n"
            "- values: Valores específicos probados\n"
            "- stats: Dict anidado con:\n"
            "   • base_stats: Referencia estadística original\n"
            "   • modified_stats: Estadísticas con modificaciones\n"
            "   • comparison: Métricas comparativas detalladas\n"
            "- performance: Dict con:\n"
            "   • time_metrics: Distribución temporal de eventos\n"
            "   • momentum: Evolución de actividad\n"
            "   • correlations: Relaciones entre variables\n"
            "   • win_probability: Modelo predictivo"
        )

        modified_params = [p for _, p in param_specs]
        results = []
        for combo in values:
            mod_home = self.base_team
            mod_away = self.away_team
            
            for (team, param), value in zip(param_specs, combo):
                if team == 'home':
                    mod_home = self._modify_team(param, value, mod_home)
                else:
                    mod_away = self._modify_team(param, value, mod_away)
            
            sim_results = self._run_simulations(mod_home, mod_away)
            
            results.append({
                'parameters': param_specs,
                'values': combo,
                'stats': self.calculate_statistics(sim_results, mod_home),
                'performance': self.calculate_performance_metrics(sim_results, target_team, modified_params)
            })
        return explanation, results

    def compare_tactic(self, configs: Dict[str, Team]) -> Dict:
        """Compara configuraciones tácticas completas"""
        configs_info = []
        for name, team in configs.items():
            params = [
                f"{param}: {getattr(team, param):.2f}" 
                for param in ['attack_rate', 'foul_rate', 'attack_strength', 'defense_strength']
            ]
            configs_info.append(f"- {name}: {', '.join(params)}")
        
        explanation = (
            "Comparación de configuraciones tácticas completas:\n"
            "Configuraciones evaluadas:\n"
            + "\n".join(configs_info) + "\n\n"
            "Estructura del resultado (dict):\n"
            "- Por cada configuración (key):\n"
            "   • stats: Dict con:\n"
            "       - base_stats: Referencia base\n"
            "       - modified_stats: Estadísticas modificadas\n"
            "       - comparison: Análisis comparativo detallado\n"
            "   • performance: Dict con:\n"
            "       - time_metrics: Eventos por periodos\n"
            "       - momentum: Tendencia temporal\n"
            "       - correlations: Matriz de correlaciones\n"
            "       - win_probability: Probabilidad de victoria"
        )
        
        results = {}
        for name, team in configs.items():
            sim_results = self._run_simulations(team, self.away_team)
            
            # Determinar parámetros modificados dinámicamente
            modified_params = [
                param for param in self.parameter_descriptions.keys()
                if abs(getattr(team, param) - getattr(self.base_team, param)) > 1e-6
            ]
            
            results[name] = {
                'stats': self.calculate_statistics(sim_results, team),
                'performance': self.calculate_performance_metrics(
                    sim_results, 
                    target_team='home',  # Asumiendo que siempre modificamos el equipo local
                    modified_params=modified_params  # Parámetros que difieren del base
                )
            }
        return explanation, results
    
    # Principal análisis estadístico
    def calculate_statistics(self, sim_results: List[Dict], modified_team: Team) -> Tuple[str, Dict, List[str]]:
        """Analiza estadísticas comparativas con visualización"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Cálculo de métricas para goles ya que es lo más importante
        goals = [sim['goals']['home'] for sim in sim_results]
        base_goals = [sim['goals']['home'] for sim in self.base_sim_results]
        
        t_test = stats.ttest_ind(goals, base_goals)
        effect_size = self._calculate_effect_size(goals, base_goals)
        bayesian = self._bayesian_comparison(goals, base_goals)
        prob_superiority = self._probability_of_superiority(goals, base_goals)
        practical_sig = self._practical_significance(effect_size)

        plt.figure(figsize=(10, 6))
        plt.bar(['Configuración', 'Base'], [np.mean(goals), np.mean(base_goals)], 
                yerr=[[np.mean(goals)-np.percentile(goals,5), np.mean(base_goals)-np.percentile(base_goals,5)], 
                    [np.percentile(goals,95)-np.mean(goals), np.percentile(base_goals,95)-np.mean(base_goals)]],
                capsize=10)
        img_path = f"report/imgs/stats_comparison_{timestamp}.png"
        plt.savefig(img_path, dpi=150)
        plt.close()

        # Comparaciones detalladas para todas las métricas y equipos
        modified_stats = SimulationAnalyzer(sim_results, modified_team, self.away_team).analyze_basic_statistics(sim_results)[1]
        base_descriptive = self.base_stats['descriptive_stats']
        modified_descriptive = modified_stats['descriptive_stats']

        def get_entry(stats_list, team, metric):
            for entry in stats_list:
                if entry['team'] == team.upper() and entry['metric'] == metric:
                    return entry
            return None

        comparison_imgs = []
        team_data_home = []
        team_data_away = []

        for team in ['home', 'away']:
            team_data = []
            for metric in ['goals', 'shots', 'fouls', 'possession']:
                base_entry = get_entry(base_descriptive, team, metric)
                modified_entry = get_entry(modified_descriptive, team, metric)
                
                if not base_entry or not modified_entry:
                    continue  # Manejar error si es necesario
                
                row = {'metric': metric}
                for stat in ['Media', 'Mediana', 'Moda', 'DE', 'IQR', 'CV%']:
                    base_val = base_entry.get(stat, 0)
                    modified_val = modified_entry.get(stat, 0)
                    delta = modified_val - base_val
                    
                    # Formatear cambio
                    if stat == 'CV%':
                        delta_str = f"{delta:+.2f}%"
                    else:
                        delta_str = f"{delta:+.2f}"
                    
                    row[f'{stat} (Base)'] = f"{base_val:.2f}"
                    row[f'{stat} (Modificado)'] = f"{modified_val:.2f}"
                    row[f'Δ {stat}'] = delta_str
                
                team_data.append(row)
            
            if team == "home":
                team_data_home = team_data.copy()
            else:
                team_data_away = team_data.copy()

            # Generar y guardar tabla
            df_team = pd.DataFrame(team_data)
            team_img = f"report/imgs/{team}_comparison_{timestamp}.png"
            self._save_table_image(df_team, team_img, f"Comparación {team.capitalize()}")
            comparison_imgs.append(team_img)

        explanation = (
            "Análisis estadístico comparativo:\n"
            "1. Método: Test t-Student para muestras independientes\n"
            "2. Comparación con configuración base del modelo\n"
            f"3. Resultados clave (goles):\n"
            f"   - Diferencia media: {np.mean(goals)-np.mean(base_goals):.2f} goles\n"
            f"   - Significancia estadística (p-valor): {t_test.pvalue:.4f}\n"
            f"   - Tamaño del efecto (Cohen's d): {effect_size:.2f}\n"
            "4. Tablas comparativas por equipo y métrica:\n"
            f"- {comparison_imgs[0]}: Comparación detallada para el equipo modificado (HOME)\n"
            f"- {comparison_imgs[1]}: Comparación para el equipo contrario (AWAY)\n"
            "   Estructura de las tablas:\n"
            "   - Filas: Métricas (goles, tiros, faltas, posesión)\n"
            "   - Columnas: Cada estadística (Media, Mediana, Moda, DE, IQR, CV%) con valores base, modificados y diferencia (Δ)\n"
            "   - Ejemplo: Δ Media = +0.50 indica un aumento de 0.5 unidades respecto a la base\n\n"
            "Estructura del resultado 'result':\n"
            "- base_stats: Estadísticas descriptivas de referencia\n"
            "- modified_stats: Estadísticas de la configuración modificada\n"
            "- comparison: Resultados comparativos detallados\n\n"
            "Imágenes generadas:\n"
            f"- {img_path}: Gráfico de barras comparando medias de goles\n"
            f"- {comparison_imgs[0]}: Tabla comparativa para HOME\n"
            f"- {comparison_imgs[1]}: Tabla comparativa para AWAY"
        )
        
        result = {
            'base_stats': self.base_stats,
            'modified_stats': modified_stats,
            'comparison': {
                'mean_difference': np.mean(goals) - np.mean(base_goals),
                'effect_size': effect_size,
                'p_value': t_test.pvalue,
                'confidence_interval': stats.t.interval(0.95, len(goals)-1, loc=np.mean(goals), scale=stats.sem(goals)),
                'prob_superiority': prob_superiority,
                'practical_significance': practical_sig,
                'bayesian_hdi': bayesian['hdi_95'],
                'home_comparison_data': team_data_home,  
                'away_comparison_data': team_data_away   
            }
        }
        
        return explanation, result, [img_path] + comparison_imgs 
    
    # Auxiliares
    def _save_table_image(self, df: pd.DataFrame, path: str, title: str) -> None:
        """Genera y guarda una tabla como imagen PNG"""
        plt.figure(figsize=(12, len(df)*0.5))
        ax = plt.subplot(frame_on=False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        
        table = pd.plotting.table(ax, df, loc='center', cellLoc='center', colWidths=[0.15]*len(df.columns))
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
        
        plt.title(title, fontsize=12, pad=20)
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()

    def _calculate_effect_size(self, group1, group2) -> float:
        """Calcula tamaño del efecto (Cohen's d)"""
        pooled_std = np.sqrt((np.std(group1)**2 + np.std(group2)**2) / 2)
        return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std != 0 else 0.0
    
    def _bayesian_comparison(self, group1, group2) -> Dict:
        """Comparación bayesiana"""
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS
        import arviz as az
        import jax
        
        try:
            # Modelo Bayesiano
            def model():
                # Priors
                mu1 = numpyro.sample('mu1', dist.Normal(np.mean(group1), np.std(group1)))
                mu2 = numpyro.sample('mu2', dist.Normal(np.mean(group2), np.std(group2)))
                
                # Likelihoods
                numpyro.sample('obs1', dist.Normal(mu1, np.std(group1)), obs=np.array(group1))
                numpyro.sample('obs2', dist.Normal(mu2, np.std(group2)), obs=np.array(group2))
                
                # Diferencia
                numpyro.deterministic('diff', mu1 - mu2)

             # Configurar MCMC
            nuts_kernel = NUTS(model)
            mcmc = MCMC(nuts_kernel, num_warmup=self.mcmc_warmpup, num_samples=self.mcmc_samples)
            
            # Usar JAX para generación de números aleatorios
            rng_key = jax.random.PRNGKey(42)  
            
            mcmc.run(rng_key)
            
            # Convertir a formato ArviZ
            idata = az.from_numpyro(mcmc)
            
            # Calcular métricas
            diff_samples = idata.posterior['diff'].values.flatten()
            hdi = az.hdi(diff_samples, hdi_prob=0.95)
            
            return {
                'prob_superiority': np.mean(diff_samples > 0),
                'hdi_95': [hdi[0].item(), hdi[1].item()]
            }
        
        except Exception as e:
            print(f"Error en análisis bayesiano: {str(e)}")
            return {
                'prob_superiority': np.nan,
                'hdi_95': [np.nan, np.nan]
            }
    
    def _probability_of_superiority(self, group1, group2) -> float:
        """Calcula probabilidad de que grupo1 > grupo2"""
        return np.mean([x > y for x, y in zip(group1, group2)])

    def _practical_significance(self, effect_size: float) -> Dict:
        """Evalúa significancia práctica"""
        return {
            'effect_size': effect_size,
            'interpretation': 'Grande' if effect_size >= 0.8 else
                              'Moderado' if effect_size >= 0.5 else
                              'Pequeño' if effect_size >= 0.2 else
                              'Irrelevante'
        }
    
    # Análisis estadístico secundario
    def calculate_performance_metrics(self, sim_results: List[Dict], target_team: str, modified_params: List[str]) -> Tuple[str, Dict, List[str]]:
        """Analiza métricas de rendimiento con visualizaciones avanzadas"""
        img_paths = []
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  
        
        # 1. Análisis temporal con nombre único
        time_metrics = self._calculate_time_metrics(sim_results)
        plt.figure(figsize=(12, 6))
        periods = list(time_metrics['period_density'].keys())
        events = [time_metrics['period_density'][p] for p in periods]
        plt.bar(periods, events)
        time_img = f"report/imgs/time_metrics_{timestamp}_{target_team}.png"
        plt.savefig(time_img, dpi=150)
        plt.close()
        img_paths.append(time_img)
        
        # 2. Momentum con nombre único
        momentum = self._calculate_momentum(sim_results)
        plt.figure(figsize=(10, 6))
        plt.plot(
            ['0-15','15-30','30-45','45-60','60-75','75-90'], 
            momentum['trend_data'],  
            marker='o',
            label=f"Tendencia ({momentum['trend_label']})"  
        )
        momentum_img = f"report/imgs/momentum_{timestamp}_{target_team}.png"
        plt.savefig(momentum_img, dpi=150)
        plt.close()
        img_paths.append(momentum_img)
        
        # 3. Correlaciones con manejo seguro
        correlations = self._calculate_parameter_correlations(sim_results, target_team, modified_params)
        corr_df = None

        if correlations:
            valid_params = {k:v for k,v in correlations.items() if not v.get('error')}
            
            if valid_params:
                corr_data = {
                    param: [
                        vals.get('pearson', np.nan),
                        vals.get('spearman', np.nan),
                        vals.get('effect_size', np.nan)
                    ] 
                    for param, vals in valid_params.items()
                }
                
                corr_df = pd.DataFrame(corr_data, 
                    index=['Pearson', 'Spearman', 'Tamaño de Efecto']
                )
                
                plt.figure(figsize=(12, 6))
                sns.heatmap(corr_df, annot=True, cmap='coolwarm', center=0,
                        annot_kws={'size':8}, fmt=".2f")
                plt.title("Relación Parámetros-Rendimiento")
                
                # Generar hash hexadecimal de 6 caracteres
                params_hash = hex(abs(hash(tuple(modified_params))))[2:8]  
                corr_img = f"report/imgs/correlations_{timestamp}_{target_team}_{params_hash}.png"
                plt.savefig(corr_img, dpi=150, bbox_inches='tight')
                plt.close()
                img_paths.append(corr_img)

        explanation_parts = [
            "Análisis integral de rendimiento:",
            "1. Métricas temporales: Distribución de eventos por periodos de 15 minutos",
            "2. Momentum: Tendencia de actividad durante el partido",
            "3. Correlaciones: Relación entre parámetros del equipo y resultados",
            "4. Probabilidad de victoria: Modelo de regresión logística\n",
            "Estructura del resultado 'result':",
            "- time_metrics: Densidad de eventos por intervalos (period_density)",
            "- momentum: Datos de tendencia (slope, trend, trend_data)",
            "- correlations: Matriz de correlaciones de Pearson y Spearman",
            "- win_probability: Coeficientes del modelo predictivo\n",
            "Imágenes generadas (formato nombre):",
            f"- {time_img}: Distribución temporal (datos de 'period_density')\n"
            f"- {momentum_img}: Tendencia usando 'trend_data' del momentum\n"
        ]
        
        if corr_df is not None:
            explanation_parts.append(
                f"- {corr_img}: Heatmap de 'correlations' entre parámetros"
            )

        explanation = "\n".join(explanation_parts)
        
        return explanation, {
            'time_metrics': time_metrics,
            'momentum': momentum,
            'correlations': correlations if corr_df is not None else {},
            'win_probability': self._calculate_win_probability(sim_results, target_team)
        }, img_paths

    # Auxiliares
    def _calculate_time_metrics(self, results: List[Dict]) -> Dict:
        """Analiza distribución temporal de eventos"""
        return {
            'period_density': self.__analyze_period_density(results),
            'second_half_ratio': self.__calculate_second_half_ratio(results),
            'late_events': self.__calculate_late_events(results),
            'events_per_minute': self.__calculate_events_per_minute(results)
        }
    
    def _calculate_momentum(self, results: List[Dict]) -> Dict:
        """Analiza tendencia temporal de eventos"""
        periods = ['0-15', '15-30', '30-45', '45-60', '60-75', '75-90']
        event_counts = [self.__analyze_period_density(results)[f'{p}_events'] for p in periods]
        slope, _, r_value, _, _ = stats.linregress(range(len(periods)), event_counts)
        return {
            'slope': slope,
            'trend_label': 'Ascendente' if slope > 0 else 'Descendente',  
            'trend_data': event_counts,  
            'correlation': r_value
        }
    
    def _calculate_parameter_correlations(self, results: List[Dict], target_team: str, modified_params: List[str]) -> Dict:
        """Calcula correlaciones solo para parámetros modificados con análisis de sensibilidad"""
        correlations = {}
        
        # Obtener todos los parámetros de ambos equipos desde los resultados
        all_params = list(set(
            list(results[0]['home_params'].keys()) + 
            list(results[0]['away_params'].keys())
        ))
        
        for param in all_params:
            try:
                # Determinar equipo real del parámetro
                team_type = 'home' if param in results[0]['home_params'] else 'away'
                x_data = np.array([sim[f'{team_type}_params'][param] for sim in results])
                y_data = np.array([sim['goals'][target_team] for sim in results])
                
                # Calcular solo si el parámetro fue modificado
                if param in modified_params:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        pearson = stats.pearsonr(x_data, y_data)[0]
                        spearman = stats.spearmanr(x_data, y_data)[0]
                        
                    correlations[param] = {
                        'pearson': pearson,
                        'spearman': spearman,
                        'effect_size': np.mean(y_data)/np.std(y_data),
                        'data_range': {
                            'min': np.min(x_data),
                            'max': np.max(x_data),
                            'delta': np.max(x_data) - np.min(x_data)
                        }
                    }
                    
            except Exception as e:
                correlations[param] = {
                    'error': str(e),
                    'pearson': np.nan,
                    'spearman': np.nan
                }
        
        return correlations

    def _calculate_win_probability(self, results: List[Dict], target_team: str) -> Dict:
        """Modela probabilidad de victoria con regresión logística"""
        try:
            # Usar parámetros almacenados en los resultados
            X = np.array([
                [
                    sim[f'{target_team}_params']['attack_rate'], 
                    sim[f'{target_team}_params']['defense_strength']
                ] for sim in results
            ])
            
            # Determinar variable objetivo (victoria del equipo modificado)
            y = np.array([1 if sim['goals'][target_team] > sim['goals']['away' if target_team == 'home' else 'home'] else 0 for sim in results])
            
            model = LogisticRegression().fit(X, y)
            return {
                'coefficients': model.coef_[0].tolist(),
                'intercept': model.intercept_[0],
                'odds_ratios': [np.exp(coef) for coef in model.coef_[0]],
                'accuracy': model.score(X, y)
            }
        except Exception as e:
            print(f"Error en modelo predictivo: {str(e)}")
            return {
                'coefficients': [np.nan, np.nan],
                'intercept': np.nan,
                'odds_ratios': [np.nan, np.nan],
                'accuracy': np.nan
            }
    
    def __analyze_period_density(self, results: List[Dict]) -> Dict:
        """Calcula densidad de eventos por intervalos de 15 minutos"""
        period_ranges = {
            '0-15': (0, 15),
            '15-30': (15, 30),
            '30-45': (30, 45),
            '45-60': (45, 60),
            '60-75': (60, 75),
            '75-90': (75, 90)
        }
        
        period_counts = {}
        
        for period_name, (start_time, end_time) in period_ranges.items():
            total_events = 0
            
            for simulation in results:
                # Contar eventos en el periodo actual para esta simulación
                events_in_period = [
                    t for t in simulation['event_times'] 
                    if start_time < t <= end_time
                ]
                total_events += len(events_in_period)
            
            # Calcular promedio evitando división por cero
            average_events = total_events / len(results) if len(results) > 0 else 0.0
            period_counts[f"{period_name}_events"] = round(average_events, 2)
        
        return period_counts

    def __calculate_second_half_ratio(self, results: List[Dict]) -> float:
        """Ratio de eventos en segundo tiempo vs primer tiempo"""
        second = sum(len([t for t in sim['event_times'] if t > 45]) for sim in results)
        first = sum(len([t for t in sim['event_times'] if t <= 45]) for sim in results)
        return second / first if first > 0 else 0.0

    def __calculate_late_events(self, results: List[Dict]) -> float:
        """Porcentaje de eventos en últimos 10 minutos"""
        total = sum(len(sim['event_times']) for sim in results)
        late = sum(len([t for t in sim['event_times'] if t > 80]) for sim in results)
        return late / total if total > 0 else 0.0

    def __calculate_events_per_minute(self, results: List[Dict]) -> float:
        """Calcula tasa global de eventos"""
        total = sum(len(sim['event_times']) for sim in results)
        return total / (90 * len(results)) if len(results) > 0 else 0.0

    # Análisis avanzados
    def analyze_interaction_effects(self, param_pairs: List[Tuple[str, str]], value_ranges: Dict[str, Tuple[float, float]]) -> Tuple[str, Dict, List[str]]:
        """Analiza efectos de interacción entre parámetros con ANOVA y heatmaps"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  
        results = []
        img_paths = []
        
        for p1, p2 in param_pairs:
            combinations = list(itertools.product(value_ranges[p1], value_ranges[p2]))
            data = []
            
            # 1. Generación de datos robusta
            for v1, v2 in combinations:
                try:
                    modified_team = self._modify_team(p1, v1)
                    modified_team = self._modify_team(p2, v2, modified_team)
                    sim_results = self._run_simulations(modified_team, self.away_team)
                    
                    # Calcular métricas con manejo de NaNs
                    goals = np.nanmean([s['goals']['home'] for s in sim_results]) if sim_results else np.nan
                    fouls = np.nanmean([s['fouls']['home'] for s in sim_results]) if sim_results else np.nan
                    
                    data.append({
                        p1: v1,
                        p2: v2,
                        'goals': goals,
                        'fouls': fouls
                    })
                except Exception as e:
                    print(f"Error en combinación {p1}={v1}, {p2}={v2}: {str(e)}")
                    continue
            
            if not data:
                continue  # Saltar si no hay datos válidos
                
            df = pd.DataFrame(data).dropna()  # Eliminar filas con NaNs
            
            # 2. Validación de datos antes de ANOVA
            if df.empty or len(df['goals'].unique()) == 1:
                print(f"Advertencia: Datos insuficientes para {p1}-{p2}")
                continue
                
            # 3. Heatmap con validación
            try:
                plt.figure(figsize=(10, 6))
                pivot = df.pivot(index=p1, columns=p2, values='goals')
                sns.heatmap(pivot, annot=True, cmap='viridis', 
                        cbar_kws={'label': 'Goles Promedio'})
                plt.title(f"Interacción {p1} vs {p2}")
                heatmap_path = f"report/imgs/interaction_{p1}_{p2}_{timestamp}.png"
                plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
                plt.close()
                img_paths.append(heatmap_path)
            except Exception as e:
                print(f"Error generando heatmap para {p1}-{p2}: {str(e)}")
            
            # 4. ANOVA con manejo seguro
            try:
                formula = f"goals ~ C({p1}) * C({p2})"
                model = ols(formula, data=df).fit()
                
                # Verificar residuos antes de ANOVA
                if np.isinf(model.resid).any() or np.isnan(model.resid).any():
                    raise ValueError("Residuos contienen valores inválidos")
                    
                anova_results = anova_lm(model, typ=2)
                residuals = model.resid
                
                # Validación de supuestos
                _, sw_p = stats.shapiro(residuals)
                _, lev_p = stats.levene(*[df[df[p1]==v]['goals'] for v in df[p1].unique()])
                
                # Calcular efecto de interacción
                interaction_key = f"C({p1}):C({p2})"
                effect_size = np.sqrt(anova_results.loc[interaction_key, 'sum_sq']/anova_results['sum_sq'].sum()) if interaction_key in anova_results.index else 0.0
                
                results.append({
                    'parameters': (p1, p2),
                    'anova_table': anova_results.to_dict(),
                    'effect_size': effect_size,
                    'assumptions': {
                        'normality': sw_p > 0.05,
                        'homoscedasticity': lev_p > 0.05
                    }
                })
            except Exception as e:
                print(f"Error en ANOVA para {p1}-{p2}: {str(e)}")
                results.append({
                    'parameters': (p1, p2),
                    'error': str(e)
                })
        
        explanation = (
            "Análisis de efectos de interacción:\n"
            "1. Método: ANOVA factorial con validación de supuestos\n"
            "2. Heatmaps: Visualización de combinaciones de parámetros\n"
            "3. Resultados clave: Tamaño del efecto de interacción y significancia\n\n"
            "Estructura del resultado 'result':\n"
            "- parameters: Par de parámetros analizados\n"
            "- anova_table: Resultados completos del ANOVA\n"
            "- effect_size: Proporción de varianza explicada\n"
            "- assumptions: Validación de normalidad y homocedasticidad\n\n"
            "Imágenes generadas (formato nombre):\n" +
            '\n'.join([f"- {path}: Heatmap de interacción para {os.path.basename(path).split('_')[2]} vs {os.path.basename(path).split('_')[3].split('.')[0]}" for path in img_paths])
        )
        
        return explanation, {'interactions': results}, img_paths

    def full_factorial_analysis(self, params: List[str], levels: Dict[str, List[float]]):
        """Análisis factorial completo con ANOVA multivariante"""
        from statsmodels.multivariate.manova import MANOVA
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  
        
        # Generar todas las combinaciones posibles
        factors = list(itertools.product(*[levels[p] for p in params]))
        
        data = []
        for combo in factors:
            modified_team = self.base_team
            for i, param in enumerate(params):
                modified_team = self._modify_team(param, combo[i], modified_team)
            sim_results = self._run_simulations(modified_team, self.away_team)
            
            # Estructuración correcta de datos con columnas individuales
            row = {
                'goals': np.nanmean([s['goals']['home'] for s in sim_results]),
                'fouls': np.nanmean([s['fouls']['home'] for s in sim_results])
            }
            # Agregar cada parámetro como columna individual
            for param, value in zip(params, combo):
                row[param] = value
                
            data.append(row)
        
        df = pd.DataFrame(data).dropna()
        
        # Verificar existencia de columnas
        if not all(param in df.columns for param in params):
            missing = [p for p in params if p not in df.columns]
            raise ValueError(f"Columnas faltantes: {missing}")
        
        # MANOVA con fórmula dinámica
        manova_formula = f"goals + fouls ~ {' + '.join([f'C({p})' for p in params])}"
        manova = MANOVA.from_formula(manova_formula, data=df)
        
        # Gráfico de efectos principales corregido
        plt.figure(figsize=(12, 6))
        for param in params:
            effects = df.groupby(param, observed=False)[['goals']].mean()
            plt.plot(
                effects.index.astype(str),  # Asegurar tipo string para ejes
                effects.values, 
                marker='o', 
                label=f"{param} ({self.parameter_descriptions.get(param, '')})"
            )
        
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title("Efectos Principales de Parámetros")
        effects_img = f"report/imgs/main_effects_{timestamp}.png"
        plt.savefig(effects_img, dpi=150, bbox_inches='tight')
        plt.close()
        
        explanation = (
            "Análisis factorial completo:\n"
            "1. Método: MANOVA multivariante\n"
            "2. Variables: " + ", ".join(params) + "\n"
            "3. Niveles: " + "; ".join([f"{p}: {levels[p]}" for p in params]) + "\n\n"
            "Estructura del resultado:\n"
            "- manova: Resultados multivariados\n"
            "- univariate: ANOVA univariado por métrica\n"
            "- effects: Tamaños de efecto detallados\n\n"
            f"Imagen generada: {effects_img}"
        )
        
        return explanation, {
            'manova': manova.mv_test().summary(),
            'univariate': {
                'goals': anova_lm(ols(f'goals ~ {" + ".join(["C("+p+")" for p in params])}', df).fit()).to_dict(),
                'fouls': anova_lm(ols(f'fouls ~ {" + ".join(["C("+p+")" for p in params])}', df).fit()).to_dict()
            },
            'effects': df.groupby(params)[['goals', 'fouls']].mean().to_dict()
        }, [effects_img]

    def tornado_analysis(self, params: List[str], variation: float = 0.2) -> Tuple[str, Dict, List[str]]:
        """Análisis de sensibilidad Tornado con gráfico de barras"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  
        base_goals = np.mean([sim['goals']['home'] for sim in self.base_sim_results])
        impacts = []
        
        for param in params:
            # Variación positiva
            high_team = self._modify_team(param, self.base_team.__dict__[param]*(1+variation))
            high_goals = np.mean([sim['goals']['home'] for sim in self._run_simulations(high_team, self.away_team)])
            
            # Variación negativa
            low_team = self._modify_team(param, self.base_team.__dict__[param]*(1-variation))
            low_goals = np.mean([sim['goals']['home'] for sim in self._run_simulations(low_team, self.away_team)])
            
            impacts.append({
                'parameter': param,
                'impact': max(abs(high_goals-base_goals), abs(low_goals-base_goals)),
                'delta': (high_goals - low_goals)/base_goals
            })
        
        # Generar gráfico Tornado
        plt.figure(figsize=(10, 6))
        df = pd.DataFrame(impacts).sort_values('impact', ascending=True)
        plt.barh(df['parameter'], df['impact'], color='skyblue')
        tornado_img = f"report/imgs/tornado_plot_{timestamp}.png"
        plt.savefig(tornado_img, dpi=150)
        plt.close()
        
        explanation = (
            "Análisis de sensibilidad Tornado:\n"
            "1. Método: Variación ±20% en cada parámetro\n"
            "2. Métrica: Impacto en goles promedio\n"
            "3. Visualización: Gráfico de barras horizontales\n\n"
            "Estructura del resultado 'result':\n"
            "- parameters: Lista de parámetros analizados\n"
            "- base_value: Valor base de goles\n"
            "- impacts: Magnitud del impacto por parámetro\n\n"
            f"Imágenes generadas:\n- {tornado_img}: Impacto relativo de parámetros (datos de 'impacts')"
        )
        
        return explanation, {
            'base_value': base_goals,
            'impacts': impacts
        }, [tornado_img]
    
    def temporal_stability_analysis(self, param: str, values: List[float]) -> Tuple[str, Dict, List[str]]:
        """Análisis de estabilidad temporal con gráfico de líneas"""
        trends = {}
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  
        for value in values:
            modified_team = self._modify_team(param, value)
            sim_results = self._run_simulations(modified_team, self.away_team)
            
            # Calcular tendencia temporal
            time_metrics = self._calculate_time_metrics(sim_results)
            trends[value] = [time_metrics['period_density'][f"{i*15}-{(i+1)*15}_events"] for i in range(6)]
        
        # Generar gráfico
        plt.figure(figsize=(12, 6))
        for value, trend in trends.items():
            plt.plot(['0-15','15-30','30-45','45-60','60-75','75-90'], trend, marker='o', label=f'{param}={value}')
        plt.legend()
        trend_img = f"report/imgs/trend_{param}_{timestamp}.png"
        plt.savefig(trend_img, dpi=150)
        plt.close()
        
        explanation = (
            "Análisis de estabilidad temporal:\n"
            f"1. Parámetro variado: {param}\n"
            "2. Métrica: Eventos por intervalo de 15 minutos\n"
            "3. Resultado: Patrones temporales para cada valor\n\n"
            "Estructura del resultado 'result':\n"
            "- parameter: Parámetro analizado\n"
            "- values: Valores probados\n"
            "- trends: Datos temporales por valor\n\n"
            f"Imágenes generadas:\n- {trend_img}: Tendencia temporal (datos de 'trends')"
        )
        
        return explanation, {
            'parameter': param,
            'values': values,
            'trends': trends
        }, [trend_img]

    # Auxiliares
    def _create_team_from_params(self, params: np.ndarray) -> Team:
        """Crea un equipo a partir de un array de parámetros"""
        return Team(
            name=f"Optimized_{hash(params.tobytes())}",
            attack_rate=params[0],
            foul_rate=params[1],
            attack_strength=params[2],
            defense_strength=params[3],
            stamina_decay=params[4]
           )
    
    def _calculate_anova_effect_size(self, results: Dict) -> float:
            """Calcula eta-squared para tamaño del efecto"""
            ss_total = np.sum([(x - np.mean(list(results.values()))) ** 2 for group in results.values() for x in group])
            ss_between = np.sum([len(group) * (np.mean(group) - np.mean(list(results.values()))) ** 2 for group in results.values()])
            return  ss_between / ss_total if ss_total != 0 else 0.0

    def _calculate_effect_sizes(self, df: pd.DataFrame) -> Dict:
        """Calcula eta-squared y omega-squared"""
        effect_sizes = {}
        grand_mean = df['goals'].mean()
        ss_total = np.sum((df['goals'] - grand_mean)**2)
        
        for param in df['params'].unique():
            groups = df.groupby(param)['goals']
            ss_between = np.sum([len(g)*(g.mean()-grand_mean)**2 for _,g in groups])
            ss_within = np.sum([np.sum((g - g.mean())**2) for _,g in groups])
            
            # Eta-squared
            eta_sq = ss_between / ss_total
            
            # Omega-squared (más preciso)
            omega_sq = (ss_between - (len(groups)-1)*ss_within/(len(df)-len(groups))) / ss_total
            
            effect_sizes[param] = {
                'eta_squared': eta_sq,
                'omega_squared': max(0, omega_sq)  
            }
        
        return effect_sizes
    
