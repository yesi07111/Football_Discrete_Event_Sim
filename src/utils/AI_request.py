class AI_Request:
    @property
    def STATISTICAL_PROMPT(self) -> str:
        return """
        Generar un reporte técnico en LaTeX basado en el siguiente análisis estadísticos clave sobre unas simulaciones. En la Explicación viene texto descriptivo sobre el método estadístico usado, breve interpretación, nombre y estructura del resultado y explicacion de que es la imagen y que datos contiene, mientras que en Resultado está un objeto estructurado (dict/list) llevado a string con valores numéricos clave y en Ruta a las imágenes estan las rutas a las imagenes generadas \(graficos o tablas\). 

        **Datos del Modelo de Simulación:**
        - **Motor DES**: 
        - Eventos (tiros/faltas) programados con distribución exponencial usando tasas ajustadas por stamina
        - Cambios de posesión automáticos después de cada evento
        - Decaimiento de stamina: 0.1% por minuto de posesión (mín 40%)
        - Probabilidad de gol: Función logística basada en fuerza de ataque/defensa

        - **Equipos**:
        - Parámetros configurables: attack_rate, foul_rate, attack_strength, defense_strength
        - Comportamiento emergente: Mayor posesión = más desgaste físico

        **Datos Clave:**
        \begin{itemize}
        \item Tiempos entre eventos: $t \sim \text{Exp}(\lambda=\text{attack\_rate} \times \text{stamina})$
        \item Probabilidad de gol: $\sigma(a - d) = \frac{1}{1 + e^{-k(a-d)}}$, $k=6$
        \item Decaimiento stamina: $\Delta s = -\alpha t$, $\alpha=0.1\%/\text{min}$
        \end{itemize}

        **Instrucciones:**
        Para cada análisis proporcionado:
        1. **Explicación Estadística Detallada**:
        - Presentar el análisis actual por su nombre técnico
        - Describe metodología con términos técnicos (ej: "Intervalo de Wilson", "KDE")
        - Incluir figura LaTeX usando la ruta proporcionada (ej: \includegraphics)
        - Presenta y analiza valores clave (p-valores, intervalos confianza, R²)
        - Menciona supuestos estadísticos y limitaciones
        - Métodos matemáticos con fórmulas en LaTeX (ej: $p < 0.05$, $\lambda = 1/\text{rate}$)
        - Interpretación técnica de resultados
        - Incluye justificación de porque se escogió dicho análisis sobre los parámetros escogidos
    

        2. **Explicación en Contexto Futbolístico**:
        - Traduce resultados a implicaciones tácticas detalladas
        - Compara con benchmarks de fútbol real (ej: UCL, Premier League)
        - Analiza impacto en estrategia de equipos
        - Considera dinámicas de partido (ej: gestión de stamina, presión alta)
        - Relacionar gráficos con situaciones reales de partido

        El formato de salida debe ser un string con código LaTeX válido, usando escapes adecuados (ej: \\ para \). 

        **Estructura Requerida:**
        \begin{analysis}[<NombreAnálisis>]
        \section*{Explicación Estadística}
        <Contenido detallado con fórmulas>
        \section*{Contexto Futbolístico}
        <Contenido con referencias>
        \end{analysis}

        **Ejemplo muy corto y simplificado de Salida:**
        r'''
        \begin{analysis}[Distribución de Goles]
        \section*{Explicación Estadística}
        La distribución de Poisson ajustada muestra $\lambda=2.3$ goles/partido:
        \[
        P(k) = \frac{e^{-\lambda}\lambda^k}{k!}
        \]
        IC 95\% [2.1, 2.5] (Método bootstrap). Resultado clave: 63\% de probabilidad >2 goles.
        
        \begin{figure}[H]
            \centering
            \includegraphics[width=0.6\textwidth]{img/goal_dist.png}
            \caption{Distribución comparativa vs Premier League 2023 (línea punteada)}
        \end{figure}

        \section*{Contexto Futbolístico}
        Un 63\% de probabilidad >2 goles supera el promedio UEFA (58\%). Recomendación: Fortalecer defensa en últimos 15 mins (ver pico derecho en figura).
        \end{analysis}
        '''

        **Instrucciones Adicionales:**
        - Usar paquetes LaTeX: amsmath, amssymb
        - Evitar comandos complejos (no usar tikz)
        - Citar referencias por análisis futbolístico

        **Restricciones de Formato:**
        - Prohibido usar \maketitle, \begin{document} o comandos de preámbulo
        - Solo usar estos paquetes: babel, graphicx, booktabs, siunitx

        **Análisis:** 
        """
    property
    def BOOTSTRAP_PROMPT(self) -> str:
        return """
        **Contexto para Análisis Bootstrap:**
        Generar un reporte técnico en LaTeX basado en la siguiente evaluación de robustez estadística mediante remuestreo sobre unas simulaciones. En la Explicación viene texto descriptivo sobre el método estadístico usado, breve interpretación, nombre y estructura del resultado y explicacion de que es la imagen y que datos contiene, mientras que en Resultado está un objeto estructurado (dict/list) llevado a string con valores numéricos clave y en Ruta a las imágenes estan las rutas a las imagenes generadas \(graficos o tablas\). 
        **Datos del Modelo de Simulación:**
        - **Motor DES**: 
        - Eventos (tiros/faltas) programados con distribución exponencial usando tasas ajustadas por stamina
        - Cambios de posesión automáticos después de cada evento
        - Decaimiento de stamina: 0.1% por minuto de posesión (mín 40%)
        - Probabilidad de gol: Función logística basada en fuerza de ataque/defensa

        - **Equipos**:
        - Parámetros configurables: attack_rate, foul_rate, attack_strength, defense_strength
        - Comportamiento emergente: Mayor posesión = más desgaste físico

        **Datos Clave:**
        \begin{itemize}
        \item Tiempos entre eventos: $t \sim \text{Exp}(\lambda=\text{attack\_rate} \times \text{stamina})$
        \item Probabilidad de gol: $\sigma(a - d) = \frac{1}{1 + e^{-k(a-d)}}$, $k=6$
        \item Decaimiento stamina: $\Delta s = -\alpha t$, $\alpha=0.1\%/\text{min}$
        \end{itemize}

        **Instrucciones Específicas:**
        1. **Explicación Técnica**:
        - Describir método bootstrap usado (Ej: percentil/BCa)
        - Analizar estabilidad de estimaciones (SE, sesgo)
        - Incluir figura LaTeX usando la ruta proporcionada (ej: \includegraphics)
        - Explicar gráficos de distribución usando figuras proporcionadas
        - Discutir solapamiento de intervalos de confianza
        - Incluye justificación de porque se escogió dicho análisis sobre los parámetros escogidos


        2. **Contexto Deportivo**:
        - Explicar en contexto de fútbol que significa los resultados obtenidos
        - Interpretar estabilidad de métricas clave
        - Analizar variabilidad en contexto de partido real
        - Proponer estrategias basadas en consistencia estadística

        **Ejemplo muy corto y simplificado:**
        r'''
        \begin{analysis}[Estabilidad de Posesión]
        \section*{Explicación Técnica}
        IC 95\% via percentil bootstrap (30,000 muestras):
        \[
        \mu_{poses} = 52.3\% [50.1\%, 54.5\%]
        \]
        Sesgo: -0.3\% (Error estándar: 1.2\%)
        
        \begin{figure}[H]
            \centering
            \includegraphics[width=0.7\textwidth]{img/bootstrap_poss.png}
            \caption{Distribución bootstrap con referencia a t-student (línea roja)}
        \end{figure}

        \section*{Contexto Deportivo}
        Un IC estrecho (4.4\%) sugiere posesión consistente. Comparar con Bayern Múnich (IC 3.8\% en UCL). Estrategia: Mantener presión alta en primeros 30 mins.
        \end{analysis}
        '''

        El formato de salida debe ser un string con código LaTeX válido, usando escapes adecuados (ej: \\ para \). 
        
        **Instrucciones Adicionales:**
        - Usar paquetes LaTeX: amsmath, amssymb
        - Evitar comandos complejos (no usar tikz)
        - Citar referencias por análisis futbolístico

        **Restricciones de Formato:**
        - Prohibido usar \maketitle, \begin{document} o comandos de preámbulo
        - Solo usar estos paquetes: babel, graphicx, booktabs, siunitx

        **Análisis:** 
        """

    @property
    def SENSITIVITY_PROMPT(self) -> str:
        return """
        **Contexto para Análisis de Sensibilidad:**
        Generar un reporte técnico en LaTeX basado en la siguiente evaluación de cómo parámetros clave afectan resultados de una simulación. En la Explicación viene texto descriptivo sobre el método estadístico usado, breve interpretación, nombre y estructura del resultado y explicacion de que es la imagen y que datos contiene, mientras que en Resultado está un objeto estructurado (dict/list) llevado a string con valores numéricos clave y en Ruta a las imágenes estan las rutas a las imagenes generadas \(graficos o tablas\). 

        **Datos del Modelo de Simulación:**
        - **Motor DES**: 
        - Eventos (tiros/faltas) programados con distribución exponencial usando tasas ajustadas por stamina
        - Cambios de posesión automáticos después de cada evento
        - Decaimiento de stamina: 0.1% por minuto de posesión (mín 40%)
        - Probabilidad de gol: Función logística basada en fuerza de ataque/defensa

        - **Equipos**:
        - Parámetros configurables: attack_rate, foul_rate, attack_strength, defense_strength
        - Comportamiento emergente: Mayor posesión = más desgaste físico

        **Datos Clave:**
        \begin{itemize}
        \item Tiempos entre eventos: $t \sim \text{Exp}(\lambda=\text{attack\_rate} \times \text{stamina})$
        \item Probabilidad de gol: $\sigma(a - d) = \frac{1}{1 + e^{-k(a-d)}}$, $k=6$
        \item Decaimiento stamina: $\Delta s = -\alpha t$, $\alpha=0.1\%/\text{min}$
        \end{itemize}

        **Instrucciones Específicas:**
        1. **Explicación Técnica**:
        - Describir metodología (ANOVA, Sobol, etc.)
        - Cuantificar impacto con índices de sensibilidad
        - Explicar gráficos usando figuras adjuntas
        - Analizar interacciones entre variables
        - Incluye justificación de porque se escogió dicho análisis sobre los parámetros escogidos

        2. **Contexto Táctico**:
        - Explicar en contexto de fútbol que significa los resultados obtenidos
        - Identificar parámetros críticos para optimizar
        - Proponer ajustes tácticos basados en sensibilidad
        - Comparar con enfoques de equipos profesionales

        **Ejemplo muy corto y simplificado:**
        r'''
        \begin{analysis}[Sensibilidad Ataque-Defensa]
        \section*{Explicación Técnica}
        Índice Sobol 1er orden:
        \[
        S_{ataque} = 0.62,\ S_{defensa} = 0.33
        \]
        Interacción cruzada: 0.15 (18\% de varianza total)
        
        \begin{figure}[H]
            \centering
            \includegraphics[width=0.75\textwidth]{img/tornado.png}
            \caption{Gráfico Tornado: Efecto parámetros en diferencia goles}
        \end{figure}

        \section*{Contexto Táctico}
        La tasa de atque explica 62\% de variabilidad (vs 45\% en Liverpool FC). Recomendación: Incrementar un 10\% attack_rate priorizando stamina.
        \end{analysis}
        '''

        El formato de salida debe ser un string con código LaTeX válido, usando escapes adecuados (ej: \\ para \). 

        **Instrucciones Adicionales:**
        - Usar paquetes LaTeX: amsmath, amssymb
        - Evitar comandos complejos (no usar tikz)
        - Citar referencias por análisis futbolístico

        **Restricciones de Formato:**
        - Prohibido usar \maketitle, \begin{document} o comandos de preámbulo
        - Solo usar estos paquetes: babel, graphicx, booktabs, siunitx

        **Análisis:** 
        """
    
    @property
    def CONCLUSIONS_PROMPT(self) -> str:
        return r"""
        **Contexto para Conclusiones:**
        Generar un resumen ejecutivo técnico en LaTeX que integre todos los análisis previos. Debe incluir:
        - Síntesis de hallazgos clave estadísticos, de bootstrap y sensibilidad
        - Recomendaciones tácticas basadas en evidencia estadística
        - Comparaciones con datos reales de fútbol profesional
        - Justificar análisis escogidos
        - Limitaciones del estudio y sugerencias para análisis futuros

        **Instrucciones Específicas:**
        1. **Estructura Requerida:**
        \section*{Resumen Ejecutivo}
        <Síntesis técnica integrando múltiples análisis>
        
        \section*{Recomendaciones Estratégicas}
        <Acciones tácticas priorizadas con base en datos>
        
        \section*{Benchmarks Profesionales}
        <Comparativas cuantitativas con ligas reales>
        
        \section*{Limitaciones y Futuros Estudios}
        <Análisis crítico de metodología y sugerencias>

        2. **Restricciones de Formato:**
        - Prohibido usar \maketitle, \begin{document} o comandos de preámbulo
        - Solo usar estos paquetes: babel, graphicx, booktabs, siunitx
    
        """