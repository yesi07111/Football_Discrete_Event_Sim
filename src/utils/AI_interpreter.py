from collections import defaultdict
import os
import re
import json
import time
import locale

from datetime import datetime
from pathlib import Path
from typing import List, Literal
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError
from fireworks.client import Fireworks
from src.utils.AI_request import AI_Request

locale.setlocale(locale.LC_ALL, 'es_ES.UTF-8')

class Message(BaseModel):
    role: Literal["user", "system", "assistant"]
    content: str
    
class StatsInterpreter:
    def __init__(self, model: str = "accounts/fireworks/models/llama-v3p3-70b-instruct"):
        env_path = Path(__file__).resolve().parent.parent.parent / '.env' 
        load_dotenv(env_path)
        
        api_key = os.getenv("FIREWORKS_API_KEY")
        if not api_key:
            raise ValueError("FIREWORKS_API_KEY no encontrada en .env")
        self.client = Fireworks(api_key="fw_3ZPpdQ2ADzFPE6pjM1kSTMKj")
        self.model = model
        self.max_retries = 1000
        self.timeout = 30  # segundos
        self.conclusions = None
        self.temperature = 0.7 
        self.messages: List[Message] = []

    def _validate_latex(self, latex_str: str) -> bool:
        """Valida caracteres problemáticos en LaTeX"""
        invalid_patterns = [
            r"(?<!\\)(?:\\{2})*[&%$#_{}]",
            r"\\[^a-zA-Z]",
            r"\bbegin\{|\\end\{"
        ]
        return not any(re.search(pattern, latex_str) for pattern in invalid_patterns)

    def _format_single_analysis(self, analysis: tuple) -> str:
        """Formatea un único análisis para el prompt"""
        if not self._is_valid_analysis_data(analysis):
            return ""
        
        explanation, result, img = analysis
        
        # Función de conversión mejorada
        def extended_serializer(obj):
            if isinstance(obj, (np.generic, np.ndarray)):
                return str(obj)
            elif isinstance(obj, (datetime, pd.Timestamp)):
                return obj.isoformat()
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            return str(obj)
        
        return (
            f"Explicación: {explanation}\n"
            f"Resultado: {json.dumps(result, default=extended_serializer, indent=2, ensure_ascii=False)}\n"
            f"Ruta a las imágenes: {img or 'Sin imágenes'}"
        )

    def _get_prompt(self, analysis_type: str) -> str:
        """Selecciona el prompt adecuado"""
        ai_request = AI_Request()
        return {
            'statistical': ai_request.STATISTICAL_PROMPT,
            'bootstrap': ai_request.BOOTSTRAP_PROMPT,
            'sensitivity': ai_request.SENSITIVITY_PROMPT
        }[analysis_type]

    def _add_message(self, role: Literal["user", "assistant"], content: str):
        """Añade mensajes validados al historial"""
        try:
            self.messages.append(
                Message(role=role, content=content)
            )
        except ValidationError as e:
            print(f"Mensaje inválido: {str(e)}")
            
    # def generate_interpretation(self, analysis_dict: dict) -> str:
    #     """Procesa cada tipo de análisis con su prompt correspondiente"""
    #     full_report = []
        
    #     for analysis_type, analyses in analysis_dict.items():
    #         for analysis in analyses:
    #             formatted_data = self._format_single_analysis(analysis)
    #             if formatted_data != "":
    #                 prompt = self._get_prompt(analysis_type) + formatted_data
    #                 self._add_message("user", prompt)

    #                 for attempt in range(1, self.max_retries):
    #                     try:
    #                         response = self.client.chat.completions.create(
    #                             model=self.model,
    #                             # messages=[msg.model_dump() for msg in self.messages],
    #                             messages = [{'role': 'user', 'content': prompt}]
    #                         )
    #                         latex_output = response.choices[0].message.content
    #                         full_report.append(latex_output)

    #                     except Exception as e:
    #                         print(f"Intento {attempt}: Error de conexión - {str(e)}")
    #                         if attempt == self.max_retries:
    #                             raise RuntimeError(f"Timeout después de {self.max_retries} intentos")
    #                     except Exception as e:
    #                         raise RuntimeError(f"Error crítico: {str(e)}")
    #     self.generate_conclusions(full_report)
    #     self.toTXT(full_report)
    #     return full_report

    def generate_conclusions(self, full_report: list) -> None:
        """Genera conclusiones resumen desde el reporte completo"""
        combined_analysis = "\n".join(full_report)
        prompt = AI_Request().CONCLUSIONS_PROMPT + f"Reporte Completo:\n{combined_analysis}"
        
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    request_timeout=self.timeout
                )
                latex_output = response.choices[0].message.content
                self.conclusions = latex_output
            
            except Exception as e:
                print(f"Intento {attempt}: Error de conexión - {str(e)}")
                if attempt == self.max_retries:
                    raise RuntimeError(f"Timeout después de {self.max_retries} intentos")
            except Exception as e:
                raise RuntimeError(f"Error crítico: {str(e)}")

    def get_conclusions(self) -> str:
        """Devuelve las conclusiones generadas"""
        return self.conclusions if self.conclusions else ""
    
    def toTXT(self, full_report: str):
        """Guarda el reporte en dos versiones: LaTeX crudo y texto limpio"""
        try:
            # Generar timestamp único
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 1. Guardar LaTeX crudo
            raw_filename = f"full_report_{timestamp}.txt"
            with open(raw_filename, 'w', encoding='utf-8') as f:
                f.write(full_report)
            
            # 2. Procesar y guardar texto limpio
            clean_filename = f"clean_report_{timestamp}.txt"
            
            # Eliminar comandos LaTeX
            cleaned = re.sub(r'\\(?:begin|end|section|subsection|item|textbf|textit)\{.*?\}|\\[a-zA-Z]+', '', full_report)
            
            # Eliminar símbolos especiales y espacios múltiples
            cleaned = re.sub(r'[{}&%$#_]', '', cleaned)
            cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)  # Compactar saltos de línea
            cleaned = cleaned.strip()
            
            with open(clean_filename, 'w', encoding='utf-8') as f:
                f.write(cleaned)

            print(f"Reportes generados: {raw_filename} y {clean_filename}")
            
        except Exception as e:
            print(f"Error guardando reportes: {str(e)}")

    def generate_interpretation(self, analysis_dict: dict) -> str:
        """Procesa análisis con manejo robusto de datos y progreso"""
        try:
            # Intentar cargar progreso existente
            status = self._load_progress()
            full_report = self._load_existing_report()
        except:
            # Inicializar desde cero si hay error
            status = {
                'start_time': datetime.now().isoformat(),
                'completed': defaultdict(list),
                'failures': []
            }
            full_report = []

        try:
            for analysis_type in analysis_dict.keys():
                # Saltar análisis ya completados
                if analysis_type in status['completed']:
                    completed_idx = status['completed'][analysis_type]
                else:
                    completed_idx = []
                
                for idx, analysis in enumerate(analysis_dict[analysis_type]):
                    if idx in completed_idx:
                        continue
                    
                    # Procesar solo si los datos son válidos
                    if self._is_valid_analysis_data(analysis):
                        success = self._process_analysis(
                            analysis_type, 
                            analysis, 
                            idx,
                            full_report,
                            status
                        )
                        if not success:
                            status['failures'].append(f"{analysis_type}-{idx}")
                    else:
                        self._log_validation_error(analysis_type, idx, "Datos inválidos detectados")
                        status['failures'].append(f"{analysis_type}-{idx}")

            self._finalize_process(full_report)
            return full_report
            
        except Exception as e:
            self._save_failure_status(status, str(e))
            raise
    def _process_analysis(self, analysis_type, analysis, idx, full_report, status):
        """Intenta procesar un análisis individual con manejo inteligente de errores"""
        # Validación temprana y definitiva
        if not self._is_valid_analysis_data(analysis):
            self._log_validation_error(analysis_type, idx, "Estructura inválida - No es tupla de 3 elementos")
            return False
        
        for attempt in range(3):  # Solo 3 reintentos para errores transitorios
            try:
                result = self._execute_analysis(analysis_type, analysis)
                full_report.append(result)
                status['completed'][analysis_type].append(idx)
                self._save_progress(status, full_report)
                return True
                
            except ValueError as e:
                if "Datos de análisis no válidos" in str(e):
                    self._log_validation_error(analysis_type, idx, str(e))
                    return False  # Error de datos, no reintentar
                
                self._handle_attempt_error(analysis_type, idx, attempt, e)
                time.sleep(1)
                
            except Exception as e:
                self._handle_attempt_error(analysis_type, idx, attempt, e)
                time.sleep(min(2 ** attempt, 5))  # Backoff exponencial máximo 5 seg
        
        self._log_failure(analysis_type, idx)
        return False
    
    def _is_valid_analysis_data(self, analysis):
        """Valida la estructura básica de los datos de análisis como tupla"""
        if not isinstance(analysis, tuple) or len(analysis) != 3:
            return False
        
        explanation, result, img = analysis
        
        # Validar tipos básicos
        if not isinstance(explanation, str) or not explanation.strip():
            return False
        
        # Img puede ser None o string con ruta
        if img is not None and not isinstance(img, str):
            return False
        
        return True

    def _log_validation_error(self, analysis_type, idx, error):
        """Registra errores de validación de datos"""
        error_msg = f"""
        ERROR DE VALIDACIÓN:
        - Tipo: {analysis_type}
        - Índice: {idx}
        - Hora: {datetime.now().isoformat()}
        - Error: {error}
        """
        print(error_msg)
        
        with open('validation_errors.log', 'a', encoding='utf-8') as f:
            f.write(error_msg)

    def _execute_analysis(self, analysis_type, analysis):
        """Ejecuta un análisis individual"""
        formatted_data = self._format_single_analysis(analysis)
        if not formatted_data:
            raise ValueError("Datos de análisis no válidos")
            
        prompt = self._get_prompt(analysis_type) + formatted_data
        self._add_message("user", prompt)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=self.temperature
        )
        
        return response.choices[0].message.content

    def _load_progress(self):
        """Carga el progreso guardado"""
        progress_file = 'analysis_progress.json'
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                return json.load(f)
        return {
            'start_time': datetime.now().isoformat(),
            'completed': {},
            'failures': []
        }

    def _save_progress(self, status, report):
        """Guarda el progreso actual"""
        with open('analysis_progress.json', 'w', encoding='utf-8') as f:  
            json.dump(status, f, ensure_ascii=False)  
                
        with open('partial_report.txt', 'w', encoding='utf-8') as f:  
            f.write('\n'.join(report))

    def _save_failure_status(self, status, error):
        """Guarda el estado de fallo"""
        status['last_error'] = {
            'timestamp': datetime.now().isoformat(),
            'message': error
        }
        with open('analysis_progress.json', 'w', encoding='utf-8') as f:  
            json.dump(status, f, ensure_ascii=False)  

    def _finalize_process(self, full_report):
        """Finaliza el proceso exitoso"""
        self.generate_conclusions(full_report)
        self.toTXT(full_report)
        self._cleanup_temp_files()

    def _cleanup_temp_files(self):
        """Limpia archivos temporales"""
        for f in ['analysis_progress.json', 'partial_report.txt']:
            if os.path.exists(f):
                os.remove(f)

    def _handle_attempt_error(self, analysis_type, idx, attempt, error):
        """Maneja errores de intento individual"""
        error_msg = f"""
        Error procesando análisis:
        - Tipo: {analysis_type}
        - Índice: {idx}
        - Intento: {attempt + 1}/100
        - Error: {str(error)}
        """
        print(error_msg)
        if attempt >= 10:
            sleep_time = min(2 ** attempt, 300)
            time.sleep(sleep_time)

    def _log_failure(self, analysis_type, idx):
        """Registra fallo permanente"""
        with open('analysis_failures.log', 'a') as f:
            f.write(f"""
            FALLO PERMANENTE:
            - Tipo: {analysis_type}
            - Índice: {idx}
            - Hora: {datetime.now().isoformat()}
            """)

    def _load_existing_report(self):
        """Carga reporte parcial existente"""
        if os.path.exists('partial_report.txt'):
            with open('partial_report.txt', 'r', encoding='utf-8') as f:  
                return f.read().splitlines()
        return []



