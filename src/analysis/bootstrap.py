import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy import stats
from typing import List, Dict, Tuple, Optional

from src.simulator.team import Team
from src.analysis.translator import _translate_metric

class BootstrapAnalyzer:
    def __init__(
        self, 
        original_simulations: List[Dict],
        original_stats: Dict,  
        home_team: Team,  
        away_team: Team,  
        n_bootstrap: int = 10000,
        random_seed: Optional[int] = None,
    ):
        self.original_simulations = original_simulations
        self.original_stats = original_stats[1]
        self.n_bootstrap = n_bootstrap
        self.bootstrap_samples = []
        self.bootstrap_metrics = {
            'home': {'goals': [], 'shots': [], 'fouls': [], 'possession': []},
            'away': {'goals': [], 'shots': [], 'fouls': [], 'possession': []}
        }
        self.home_team = home_team
        self.away_team = away_team
        
        if random_seed:
            np.random.seed(random_seed)
            
        self._generate_bootstrap_samples()
        self._process_bootstrap_samples() 
    
    def _generate_bootstrap_samples(self):
        """Genera muestras bootstrap con reemplazo"""
        n = len(self.original_simulations)
        indices = np.arange(n)
        
        for _ in tqdm(range(self.n_bootstrap), desc="Generando muestras bootstrap"):
            sample_indices = np.random.choice(indices, size=n, replace=True)
            self.bootstrap_samples.append([self.original_simulations[i] for i in sample_indices])
    
    def _process_bootstrap_samples(self):
        """Procesa muestras bootstrap para extraer métricas"""
        for sample in tqdm(self.bootstrap_samples, desc="Analizando muestras"):
            for team in ['home', 'away']:
                for metric in ['goals', 'shots', 'fouls', 'possession']:
                    if metric == 'possession':
                        values = [
                            (sim['possession_time'][team] / sim['clock'] * 100 
                            if sim['clock'] > 0 else 0)
                            for sim in sample
                        ]
                    else:
                        values = [sim[metric][team] for sim in sample]
                    
                    self.bootstrap_metrics[team][metric].append({
                        'mean': np.nanmean(values),
                        'median': np.nanmedian(values),
                        'std': np.nanstd(values)
                    })
    
    # Análisis
    def calculate_bias(self) -> Dict:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        """Calcula sesgo bootstrap vs estadísticas originales"""
        bias_results = {}
        
        for team in ['home', 'away']:
            team_bias = {}
            for metric in ['goals', 'shots', 'fouls', 'possession']:
                try:
                    original_item = next(
                        item for item in self.original_stats['descriptive_stats']
                        if item['team'] == team.upper() and item['metric'] == metric  
                    )
                    original_value = original_item['mean']
                except StopIteration:
                    raise ValueError(f"Estadística no encontrada para {team}.{metric}") from None
                
                # Manejo de muestras vacías
                bootstrap_means = [x['mean'] for x in self.bootstrap_metrics[team][metric] if not np.isnan(x['mean'])]
                if not bootstrap_means:
                    raise ValueError(f"No hay datos bootstrap para {team}.{metric}")
                
                bias = np.mean(bootstrap_means) - original_value
                
                team_bias[metric] = {
                    'original': original_value,
                    'bootstrap_mean': np.mean(bootstrap_means),
                    'bias_absolute': bias,
                    'bias_relative': bias / original_value if original_value != 0 else 0
                }
            bias_results[team] = team_bias
        
        # Generar gráficos
        img_paths = []
        for team in ['home', 'away']:
            plt.figure(figsize=(10, 6))
            metrics = list(bias_results[team].keys())
            biases = [v['bias_relative'] for v in bias_results[team].values()]
            
            plt.barh(metrics, biases, color=['#3498db' if b < 0 else '#e74c3c' for b in biases])
            plt.title(f"Sesgo Relativo - {team.upper()}")
            plt.xlabel("Sesgo (% respecto al valor original)")
            
            img_path = f"report/imgs/{team}_bias_analysis_{timestamp}.png"
            plt.savefig(img_path, dpi=150)
            plt.close()
            img_paths.append(img_path)
        
        explanation = (
            "Análisis de Sesgo:\n"
            "Compara medias bootstrap vs originales para detectar desviaciones sistemáticas.\n\n"
            "Interpretación Estadística:\n"
            "- Sesgo positivo: Las simulaciones sobrestiman el valor real\n"
            "- Sesgo negativo: Subestimación en las simulaciones\n\n"
            "Estructura del Resultado (result):\n"
            "- Dict con keys 'home' y 'away'\n"
            "- Cada key contiene dict de métricas:\n"
            "  - 'original': Valor original (float)\n"
            "  - 'bootstrap_mean': Media bootstrap (float)\n"
            "  - 'bias_absolute': Diferencia absoluta (float)\n"
            "  - 'bias_relative': Diferencia porcentual (float)\n\n"
            "Imágenes Generadas (img_paths):\n"
            f"- {img_paths[0]}: Gráfico de barras horizontales (HOME)\n"
            "  - Eje Y: Métricas (goals, shots, etc.)\n"
            "  - Eje X: Sesgo relativo (%)\n"
            "  - Barras: Valores de 'bias_relative' por métrica\n"
            f"- {img_paths[1]}: Análogo para AWAY"
        )
        return explanation, bias_results, img_paths
            
    def distribution_hypothesis_testing(self) -> Dict:
        """Pruebas de hipótesis basadas en distribución bootstrap"""
        test_results = {}
        
        for metric in ['goals', 'shots', 'fouls', 'possession']:
            home_means = [x['mean'] for x in self.bootstrap_metrics['home'][metric]]
            away_means = [x['mean'] for x in self.bootstrap_metrics['away'][metric]]
            
            # Diferencia de medias
            differences = [h - a for h, a in zip(home_means, away_means)]
            
            # Intervalo de confianza para la diferencia
            ci_low, ci_high = np.percentile(differences, [2.5, 97.5])
            
            # Valor p empírico
            p_value = np.mean(np.array(differences) <= 0) if np.mean(differences) > 0 else np.mean(np.array(differences) >= 0)
            
            test_results[metric] = {
                'mean_difference': np.mean(differences),
                'ci_95_difference': [ci_low, ci_high],
                'p_value': p_value,
                'probability_superiority': np.mean(np.array(home_means) > np.array(away_means))
            }
        
        return test_results
    
    def analyze_metric(self, metric: str, statistic: str) -> Tuple[str, Dict, List[str]]:
        """
        Análisis completo para una métrica y estadística específica
        """
        analyses = {}
        img_paths = []
        
        # Intervalos de confianza
        _, ci_results, ci_imgs = self._calculate_confidence_intervals(statistic)
        analyses['confidence_intervals'] = ci_results
        img_paths.extend(ci_imgs)
        
        # Prueba H0: valor original vs bootstrap
        original_value = next(
            item for item in self.original_stats['descriptive_stats'] 
            if item['metric'] == metric  
        )[statistic]  
        
        for team in ['home', 'away']:
            _, ht_results, ht_imgs = self._stats_hypothesis_test(
                metric, team, statistic, original_value*0.9  # H0: 10% menos
            )
            analyses[f'hypothesis_test_{team}'] = ht_results
            img_paths.extend(ht_imgs)
        
        # Análisis de estabilidad
        stability = self._stability_analysis()
        analyses['stability'] = stability
        
        explanation = (
            f"Análisis Integral de {metric} ({statistic}):\n"
            "Combina múltiples técnicas bootstrap para evaluación exhaustiva\n\n"
            "Contenido del Análisis:\n"
            "1. Intervalos de Confianza: Precisión estimación\n"
            "2. Pruebas de Hipótesis: Validación contra valor teórico\n"
            "3. Estabilidad: Consistencia entre muestras bootstrap\n\n"
            "Interpretación Clave:\n"
            "- IC estrecho + alta estabilidad = Resultado confiable\n"
            "- p-value <0.05 en prueba = Diferencias significativas\n\n"
            "Estructura del Resultado (result):\n"
            "- 'confidence_intervals': Dict con ICs por equipo\n"
            "- 'hypothesis_test_home'/'hypothesis_test_away': Resultados pruebas\n"
            "- 'stability': Coeficientes variación por métrica\n\n"
            "Imágenes Generadas (img_paths):\n"
            "- Gráficos de IC: Histogramas con líneas percentiles\n"
            "- Gráficos de prueba hipótesis: Distribuciones H0 vs observado\n"
            "- Total imágenes: 2 (IC) + 2 (pruebas) = 4"
        )
        return explanation, analyses, img_paths

    def performance_comparison(self, benchmark: Dict) -> Tuple[str, Dict, List[str]]:
        """
        Compara resultados contra benchmarks externos
        Ejemplo benchmark: {'goals': 2.1, 'shots': 12.5, ...}
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison = {}
        img_paths = []
        
        for team in ['home', 'away']:
            plt.figure(figsize=(10,6))
            diffs = []
            metrics = []
            for metric in ['goals', 'shots']:
                bootstrap_mean = np.mean([x['mean'] for x in self.bootstrap_metrics[team][metric]])
                diff = bootstrap_mean - benchmark[metric]
                
                comparison.setdefault(team, {})[metric] = {
                    'benchmark': benchmark[metric],
                    'bootstrap_mean': bootstrap_mean,
                    'difference': diff,
                    'percent_difference': diff/benchmark[metric]*100
                }
                
                diffs.append(diff)
                metrics.append(metric)
            
            plt.bar(metrics, diffs)
            plt.title(f"Desviación vs Benchmark - {team.upper()}")
            plt.ylabel("Diferencia Absoluta")
            img_path = f"report/imgs/{team}_benchmark_comparison_{timestamp}.png"
            plt.savefig(img_path, dpi=150)
            plt.close()
            img_paths.append(img_path)
        
        explanation = (
            "Comparación con Benchmark Externo:\n"
            "Evalúa desempeño relativo respecto a valores de referencia\n\n"
            "Interpretación Estadística:\n"
            "- Diferencia positiva: Supera benchmark\n"
            "- Diferencia negativa: Por debajo del estándar\n\n"
            "Estructura del Resultado (result):\n"
            "- Dict por equipo con:\n"
            "  - 'benchmark': Valor referencia (float)\n"
            "  - 'bootstrap_mean': Media simulaciones (float)\n"
            "  - 'difference': Diferencia absoluta (float)\n"
            "  - 'percent_difference': Diferencia porcentual (float)\n\n"
            "Imágenes Generadas (img_paths):\n"
            f"- {img_paths[0]}: Barras verticales HOME\n"
            "  - Eje X: Métricas (goles, tiros)\n"
            "  - Eje Y: Diferencia vs benchmark\n"
            "  - Datos: 'difference' del resultado\n"
            f"- {img_paths[1]}: Análogo para AWAY"
        )
        return explanation, comparison, img_paths
    
    def time_decay_analysis(self) -> Tuple[str, Dict, List[str]]:
        """
        Analiza cómo cambian las métricas en función del tiempo de posesión
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        decay_results = {}
        img_paths = []
        
        for team in ['home', 'away']:
            # Calcular posesión real desde los datos originales
            possession = [
                (x['possession_time'][team] / x['clock'] * 100 
                if x['clock'] > 0 else 0)
                for x in self.original_simulations
            ]
            
            # Agrupar las muestras bootstrap por cuartil de posesión original
            bins = pd.qcut(possession, 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
            
            plt.figure(figsize=(12,6))
            for metric in ['goals', 'shots']:
                metric_means = []
                for q in ['Q1', 'Q2', 'Q3', 'Q4']:
                    # Filtrar muestras bootstrap que pertenecen a este cuartil
                    filtered_samples = [
                        sample 
                        for sample, b in zip(self.bootstrap_samples, bins)
                        if b == q
                    ]
                    
                    if not filtered_samples:
                        continue  # Saltar cuartiles sin datos
                    
                    # Calcular media de medias para el cuartil
                    means = [
                        np.mean([x[metric][team] for x in sample])
                        for sample in filtered_samples
                    ]
                    metric_means.append(np.mean(means))
                
                if len(metric_means) == 4:  # Solo graficar si hay 4 cuartiles válidos
                    plt.plot(['Q1', 'Q2', 'Q3', 'Q4'], metric_means, label=metric)
            
            plt.title(f"Rendimiento por Cuartil de Posesión - {team.upper()}")
            plt.xlabel("Cuartil de Posesión")
            plt.ylabel("Valor Promedio")
            img_path = f"report/imgs/{team}_possession_decay_{timestamp}.png"
            plt.savefig(img_path, dpi=150)
            plt.close()
            img_paths.append(img_path)
        
        explanation = (
            "Análisis de Decaimiento por Posesión:\n"
            "Evalúa relación entre tiempo de posesión y rendimiento\n\n"
            "Interpretación Estadística:\n"
            "- Tendencia ascendente: Mejor rendimiento con más posesión\n"
            "- Tendencia plana/descendente: Posesión no determina resultado\n\n"
            "Estructura del Resultado (result):\n"
            "- Imágenes de tendencias (no datos estructurados)\n\n"
            "Imágenes Generadas (img_paths):\n"
            f"- {img_paths[0]}: Gráfico lineal HOME\n"
            "  - Eje X: Cuartiles posesión (Q1-Q4)\n"
            "  - Eje Y: Valor promedio métricas\n"
            "  - Líneas: goals y shots\n"
            f"- {img_paths[1]}: Análogo para AWAY"
        )
        return explanation, {}, img_paths
  
    def distribution_analysis(self) -> Dict:
        """Análisis completo de distribuciones bootstrap"""
        analysis = {}
        
        for team in ['home', 'away']:
            team_analysis = {}
            for metric in ['goals', 'shots', 'fouls', 'possession']:
                data = [x['mean'] for x in self.bootstrap_metrics[team][metric]]                            
                try:
                    mode_value = stats.mode(data).mode.item()  # Convertir a escalar
                except IndexError:
                    mode_value = np.nan                
                # Estadísticas de distribución
                kde = stats.gaussian_kde(data)
                cdf = lambda x: kde.integrate_box_1d(-np.inf, x)
                
                ks_stat, ks_p = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
                
                team_analysis[metric] = {
                    'distribution_params': {
                        'skewness': stats.skew(data),
                        'kurtosis': stats.kurtosis(data),
                        'mode': mode_value
                    },
                    'ks_test': {
                        'statistic': ks_stat,
                        'p_value': ks_p
                    },
                    'cdf': cdf
                }
            analysis[team] = team_analysis
        explanation = (
            "Análisis de Normalidad de Medias Bootstrap:\n"
            "Evalúa si las medias siguen distribución normal (TLC)\n\n"
            "KS-test aplicado sobre distribución bootstrap vs normal teórica"
            "Interpretación Estadística:\n"
            "- Sesgo: Asimetría distribución\n"
            "- Curtosis: Medida colas/picos\n"
            "- KS-test: Normalidad distribución\n\n"
            "Estructura del Resultado (result):\n"
            "- Dict por equipo y métrica con:\n"
            "  - 'distribution_params': skewness, kurtosis, mode\n"
            "  - 'ks_test': Resultados prueba normalidad que contiene statistic y p_value\n"
            "  - 'cdf': Función distribución acumulativa\n\n"
            "Imágenes Generadas (img_paths):\n"
            "None (Análisis numérico sin visualización)"
        )
        return explanation, analysis, []
    
    def metric_correlation_analysis(self) -> Dict:
        """Análisis de correlación entre métricas usando bootstrap"""
        correlation_results = {}
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for team in ['home', 'away']:
            metrics_data = {
                'goals': [],
                'shots': [],
                'fouls': [],
                'possession': []
            }
            
            # Colectar datos de todas las muestras
            for sample in self.bootstrap_metrics[team]['goals']:
                for metric in metrics_data:
                    metrics_data[metric].append(sample['mean'])
            
            # Matriz de correlación
            df = pd.DataFrame(metrics_data)
            corr_matrix = df.corr()
            
            # Gráfico
            plt.figure(figsize=(10, 6))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
            plt.title(f"Correlación entre Métricas - {team.upper()}")
            img_path = f"report/imgs/{team}_metric_correlation_{timestamp}.png"
            plt.savefig(img_path, dpi=150)
            plt.close()
            
            correlation_results[team] = {
                'correlation_matrix': corr_matrix.to_dict(),
                'image_path': img_path
            }
        explanation = (
            "Matriz de Correlación entre Métricas:\n"
            "Identifica relaciones lineales usando medias bootstrap\n\n"
            "Interpretación Estadística:\n"
            "- Valor ±1: Correlación perfecta\n"
            "- Valor 0: Independencia lineal\n\n"
            "Estructura del Resultado (result):\n"
            "- 'correlation_matrix': Coeficientes en dict\n"
            "- 'image_path': Ruta matriz visual\n\n"
            "Imágenes Generadas (img_paths):\n"
            f"- {img_path}: Mapa de calor correlaciones\n"
            "  - Ejes X/Y: Métricas analizadas\n"
            "  - Celdas: Valor correlación\n"
            "  - Datos: 'correlation_matrix' del resultado"
        )
        return explanation, correlation_results, [img_path]    
    
    def error_distribution_analysis(self) -> Tuple[str, Dict, List[str]]:
        """Analiza distribución de errores con gráficos Q-Q"""
        img_paths = []
        error_data = {}
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Crear estructura de búsqueda normalizada
        original_stats = {}
        for entry in self.original_stats['descriptive_stats']:
            team = entry['team'].lower()
            metric = entry['metric']
            
            if team not in original_stats:
                original_stats[team] = {}
            original_stats[team][metric] = entry
        
        for team in ['home', 'away']:
            team_errors = {}
            for metric in ['goals', 'shots', 'fouls', 'possession']:
                try:
                    # Obtener estadísticas originales
                    original = original_stats[team][metric]['mean']
                    errors = [x['mean'] - original for x in self.bootstrap_metrics[team][metric]]
                    
                    # Configurar prueba de normalidad según tamaño muestral
                    n = len(errors)
                    if n >= 20:
                        norm_test = stats.normaltest(errors)  # D'Agostino-Pearson (n >= 20)
                        test_name = "D'Agostino-Pearson"
                    else:
                        norm_test = stats.anderson(errors, dist='norm')  # Anderson-Darling (n < 20)
                        test_name = "Anderson-Darling"
                    
                    # Construir resultados
                    test_result = {
                        'test': test_name,
                        'statistic': norm_test.statistic,
                        'critical_values': norm_test.critical_values.tolist() if hasattr(norm_test, 'critical_values') else None,
                        'significance_level': norm_test.significance_level.tolist() if hasattr(norm_test, 'significance_level') else None,
                        'pvalue': getattr(norm_test, 'pvalue', None)
                    }
                    
                except KeyError as e:
                    available = list(original_stats.get(team, {}).keys())
                    raise ValueError(f"Estadísticas no encontradas para {team}/{metric}. Disponibles: {available}") from e
                
                # Calcular errores
                errors = [x['mean'] - original for x in self.bootstrap_metrics[team][metric]]
                
                # Gráfico Q-Q
                plt.figure(figsize=(10, 6))
                stats.probplot(errors, plot=plt)
                plt.title(f"Distribución de Errores - {team.upper()} {_translate_metric(metric)}")
                
                img_path = f"report/imgs/{team}_{metric}_qqplot_{timestamp}.png"
                plt.savefig(img_path, dpi=150)
                plt.close()
                img_paths.append(img_path)
                
                # Prueba de normalidad apropiada
                if len(errors) >= 20:  # Mínimo para normaltest
                    norm_test = stats.normaltest(errors)
                    test_result = {
                        'statistic': norm_test.statistic,
                        'pvalue': norm_test.pvalue
                    }
                else:
                    test_result = {'statistic': None, 'pvalue': None}
                
                team_errors[metric] = {
                    'mean_error': np.mean(errors),
                    'error_std': np.std(errors),
                    'normality_test': test_result
                }
            error_data[team] = team_errors
        
        explanation = (
            "Análisis de Normalidad de Errores:\n"
            "Evalúa distribución de errores mediante:\n"
            "- Gráficos Q-Q\n"
            "- D'Agostino-Pearson (n ≥ 20) o Anderson-Darling (n < 20)\n\n"
            "Interpretación Estadística:\n"
            "- Puntos alineados con línea roja: Errores normales\n"
            "- Desviaciones sistemáticas: No normalidad\n"
            "- p-value >0.05: No se rechaza normalidad\n\n"
            "Estructura del Resultado (result):\n"
            "- Dict por equipo y métrica con:\n"
            "  - 'mean_error': Error promedio\n"
            "  - 'error_std': Desviación estándar\n"
            "  - 'normality_test': Resultados prueba (estadístico, p-value)\n\n"
            "Imágenes Generadas (img_paths):\n"
            f"- {img_paths}: Gráficos Q-Q por equipo y métrica"
        )
        return explanation, error_data, img_paths

    def trend_analysis(self) -> Tuple[str, Dict, List[str]]:
        """Analiza tendencias en secuencia bootstrap"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        img_paths = []
        trend_results = {}
        
        for team in ['home', 'away']:
            plt.figure(figsize=(12, 6))
            for metric in ['goals', 'shots']:
                means = [x['mean'] for x in self.bootstrap_metrics[team][metric]]
                
                # Suavizado con media móvil
                window_size = 100
                moving_avg = np.convolve(means, np.ones(window_size)/window_size, mode='valid')
                
                plt.plot(moving_avg, label=metric.capitalize())
            
            plt.title(f"Tendencias Bootstrap - {team.upper()}")
            plt.xlabel("Iteración Bootstrap")
            plt.ylabel("Valor Promedio")
            
            img_path = f"report/imgs/{team}_trend_analysis_{timestamp}.png"
            plt.savefig(img_path, dpi=150)
            plt.close()
            img_paths.append(img_path)
            
            trend_results[team] = {
                'correlation_iteration': {
                    metric: stats.pearsonr(range(len(means)), means)[0] 
                    for metric in ['goals', 'shots', 'fouls', 'possession']
                }
            }
        
        explanation = (
            "Análisis de Tendencias en Iteraciones:\n"
            "Evalúa estabilidad de estimaciones durante proceso bootstrap\n\n"
            "Interpretación Estadística:\n"
            "- Tendencia ascendente/descendente: Inestabilidad en simulaciones\n"
            "- Línea plana: Convergencia estable\n"
            "- Correlación significativa: Dependencia del orden de iteración\n\n"
            "Estructura del Resultado (result):\n"
            "- Dict por equipo con:\n"
            "  - 'correlation_iteration': Coeficiente Pearson entre iteración y valor (float)\n\n"
            "Imágenes Generadas (img_paths):\n"
            f"- {img_paths}: Gráficos de líneas con media móvil\n"
            "  - Eje X: Número de iteración (progresión bootstrap)\n"
            "  - Eje Y: Valor promedio métrica (suavizado)\n"
            "  - Líneas: Medias móviles de goles y tiros\n"
            "  - Datos: 'bootstrap_metrics' para cada iteración"
        )
        return explanation, trend_results, img_paths
    
    def comparative_distributions(self) -> Tuple[str, Dict, List[str]]:
        """Comparación visual distribuciones vs original"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        img_paths = []
        
        for team in ['home', 'away']:
            plt.figure(figsize=(12, 6))
            for metric in ['goals', 'shots']:
                # Datos bootstrap
                data = [x['mean'] for x in self.bootstrap_metrics[team][metric]]
                sns.kdeplot(data, label=f"Bootstrap {metric}")
                
                # Valor original - Búsqueda en lista
                original_item = next(
                    item for item in self.original_stats['descriptive_stats']
                    if item['team'] == team.upper() and item['metric'] == metric  # Claves en inglés
                )
                original = original_item['mean']
                
                plt.axvline(original, color='red', linestyle='--', label=f"Original {metric}")
            
            plt.title(f"Distribuciones Comparativas - {team.upper()}")
            plt.xlabel("Valor")
            plt.ylabel("Densidad")
            
            img_path = f"report/imgs/{team}_distribution_comparison_{timestamp}.png"
            plt.savefig(img_path, dpi=150)
            plt.close()
            img_paths.append(img_path)
        
        explanation = (
            "Comparación de Distribuciones:\n"
            "Superpone distribuciones bootstrap con valor original para detectar sesgos\n\n"
            "Interpretación Estadística:\n"
            "- Solapamiento >90%: Estimaciones precisas\n"
            "- Desplazamiento notable: Posible sesgo en simulaciones\n"
            "- Formas diferentes: Distribución no representativa\n\n"
            "Estructura del Resultado (result):\n"
            "- Dict vacío (énfasis en imágenes comparativas)\n\n"
            "Imágenes Generadas (img_paths):\n"
            f"- {img_paths}: Gráficos de densidad KDE\n"
            "  - Curva azul: Distribución bootstrap\n"
            "  - Línea roja: Valor original\n"
            "  - Eje X: Valores posibles de la métrica\n"
            "  - Eje Y: Densidad de probabilidad\n"
            "  - Datos: 'bootstrap_metrics' vs 'original_stats'"
        )
        return explanation, {}, img_paths
    
    # Auxiliares
    def _calculate_confidence_intervals(self, statistic: str = 'mean', alpha: float = 0.05) -> Tuple[str, Dict, List[str]]:
        """
        Calcula intervalos de confianza usando método percentil bootstrap
        Args:
            statistic: 'mean', 'median', 'mode', 'std'
            alpha: Nivel de significancia
        """
        ci_results = {}
        img_paths = []
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Validar estadística solicitada
        valid_stats = ['mean', 'median', 'mode', 'std']
        if statistic not in valid_stats:
            raise ValueError(f"Estadística inválida: {statistic}. Opciones válidas: {valid_stats}")
        
        for team in ['home', 'away']:
            team_ci = {}
            for metric in ['goals', 'shots', 'fouls', 'possession']:
                # Extraer estadística bootstrap con manejo de errores
                try:
                    stats_data = [x[statistic] for x in self.bootstrap_metrics[team][metric]]
                except KeyError as e:
                    available_stats = list(self.bootstrap_metrics[team][metric][0].keys())
                    raise KeyError(
                        f"Estadística '{statistic}' no disponible para {team}.{metric}. "
                        f"Estadísticas calculadas: {available_stats}"
                    ) from e
                
                # Calcular percentiles
                lower = np.percentile(stats_data, 100*(alpha/2))
                upper = np.percentile(stats_data, 100*(1-alpha/2))
                
                # Gráfico de distribución
                plt.figure(figsize=(10,6))
                sns.histplot(stats_data, kde=True)
                plt.axvline(lower, color='red', linestyle='--')
                plt.axvline(upper, color='red', linestyle='--')
                plt.title(f"IC {100*(1-alpha)}% {statistic.capitalize()} - {team.upper()} {metric.capitalize()}")
                img_path = f"report/imgs/{team}_{metric}_ci_{statistic}_{timestamp}.png"
                plt.savefig(img_path, dpi=150)
                plt.close()
                img_paths.append(img_path)
                
                team_ci[metric] = {
                    'statistic': statistic,
                    'ci': [float(lower), float(upper)],
                    'point_estimate': np.mean(stats_data),
                    'original_value': next(
                        item for item in self.original_stats['descriptive_stats'] 
                        if item['team'] == team.upper() and item['metric'] == metric
                    )[statistic]
                }
            ci_results[team] = team_ci
        
        explanation = (
            f"Intervalos de Confianza del {100*(1-alpha)}% ({statistic}):\n"
            "Método: Percentil bootstrap\n"
            "Interpretación: Rango donde se espera encontrar el verdadero valor poblacional\n"
            "Estructura 'result':\n"
            "- Dict por equipo y métrica con:\n"
            "   - statistic: str\n"
            "   - ci: [lower, upper]\n"
            "   - point_estimate: float\n"
            "   - original_value: float\n"
            f"Imágenes:\n" + "\n".join([f"- {path}: Histograma con IC ({statistic})" for path in img_paths])
        )
        return explanation, ci_results, img_paths

    def _stats_hypothesis_test(self, metric: str, team: str, statistic: str, null_value: float) -> Tuple[str, Dict, List[str]]:
        """
        Prueba de hipótesis bootstrap para una estadística específica
        Args:
            metric: 'goals', 'shots', 'fouls', 'possession'
            team: 'home'/'away'
            statistic: 'mean', 'median', 'mode', 'std'
            null_value: Valor bajo hipótesis nula
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        original = next(
            item for item in self.original_stats['descriptive_stats'] 
            if item['team'] == team.upper() and item['metric'] == metric
        )[statistic]
        
        bootstrap_stats = [x[statistic] for x in self.bootstrap_metrics[team][metric]]
        
        # Centrar datos en H0
        shifted_stats = [s - (original - null_value) for s in bootstrap_stats]
        
        # Calcular p-value
        if original > null_value:
            p_value = np.mean(np.array(shifted_stats) >= original)
        else:
            p_value = np.mean(np.array(shifted_stats) <= original)
        
        # Gráfico
        plt.figure(figsize=(10,6))
        sns.histplot(shifted_stats, kde=True)
        plt.axvline(original, color='red', linestyle='--')
        plt.title(f"Distribución H0: {statistic}={null_value}\n vs Observado={original:.2f}")
        img_path = f"report/imgs/{team}_{metric}_hypothesis_{statistic}_{timestamp}.png"
        plt.savefig(img_path, dpi=150)
        plt.close()
        
        result = {
            'null_hypothesis': f"{statistic} = {null_value}",
            'observed_value': original,
            'p_value': p_value,
            'conclusion': "Rechazar H0" if p_value < 0.05 else "No rechazar H0"
        }
        
        explanation = (
            f"Prueba de Hipótesis Bootstrap:\n"
            f"H0: {statistic} {metric} {team} = {null_value}\n"
            f"Ha: ≠ {null_value}\n"
            f"Resultado: p = {p_value:.4f} → {result['conclusion']}\n"
            "Metodología:\n1. Shift bootstrap para alinear con H0\n"
            "2. Calcular proporción de valores extremos\n"
            "Estructura 'result':\n"
            "- null_hypothesis: str\n- observed_value: float\n"
            "- p_value: float\n- conclusion: str\n"
            f"Imagen: {img_path} (Distribución H0 vs valor observado)"
        )
        return explanation, result, [img_path]

    def _stability_analysis(self, threshold: float = 0.1) -> Dict:
        """Analiza estabilidad de las estimaciones"""
        stability = {}
        
        for team in ['home', 'away']:
            team_stability = {}
            for metric in ['goals', 'shots', 'fouls', 'possession']:
                values = [x['mean'] for x in self.bootstrap_metrics[team][metric]]
                cv = stats.variation(values)
                
                team_stability[metric] = {
                    'coef_variation': cv,
                    'stable': cv < threshold,
                    'stability_category': 'Alta' if cv < 0.05 else 'Moderada' if cv < 0.15 else 'Baja'
                }
            stability[team] = team_stability
        
        return stability
    

