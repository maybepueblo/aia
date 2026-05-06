import time
import numpy as np
from typing import Tuple, Dict
from deepface import DeepFace
from sklearn.ensemble import IsolationForest

class CyberSecurityManager:
    def __init__(self):
        # Isolation Forest para Detección de Anomalías (Enfoque D)
        # Entrenado para detectar comportamientos extraños basados en [distancia_embedding, fallos_recientes, hora_del_dia]
        self.iso_forest = IsolationForest(contamination=0.05, random_state=42)
        self.is_fitted = False
        self.history_data = []  # En producción, esto se carga de la DB (audit_log)
        
        # Rate Limiting: user_id -> [timestamps de fallos]
        self.failed_attempts: Dict[int, list] = {}

    def extract_embedding(self, img_path_or_bytes) -> np.ndarray:
        """Enfoque A: Extrae el embedding E con DeepFace."""
        result = DeepFace.represent(img_path=img_path_or_bytes, model_name="Facenet", enforce_detection=False)
        emb = np.array(result[0]["embedding"], dtype=np.float32)
        
        # Facenet genera un vector de 128-D, pero database.py exige 256-D.
        # Rellenamos con ceros hasta 256. Matemáticamente, la distancia Euclidiana
        # de los ceros añadidos es 0, por lo que la precisión se mantiene intacta.
        padded_emb = np.zeros(256, dtype=np.float32)
        padded_emb[:128] = emb
        
        return padded_emb

    def check_rate_limit(self, user_id: int) -> bool:
        """Bloquea si hay >5 intentos fallidos en los últimos 10 minutos."""
        now = time.time()
        if user_id not in self.failed_attempts:
            return True
        
        # Limpiar intentos viejos (> 600 segundos = 10 mins)
        self.failed_attempts[user_id] = [t for t in self.failed_attempts[user_id] if now - t < 600]
        
        if len(self.failed_attempts[user_id]) >= 5:
            return False # Bloqueado
        return True

    def record_failed_attempt(self, user_id: int):
        if user_id not in self.failed_attempts:
            self.failed_attempts[user_id] = []
        self.failed_attempts[user_id].append(time.time())

    def get_anomaly_score(self, distance: float, failed_count: int) -> float:
        """Calcula el score de anomalía usando Isolation Forest."""
        current_hour = time.localtime().tm_hour
        features = [distance, failed_count, current_hour]
        
        self.history_data.append(features)
        
        # Entrenar de forma incremental si tenemos suficientes datos
        if len(self.history_data) > 15:
            self.iso_forest.fit(self.history_data)
            self.is_fitted = True
            
        if not self.is_fitted:
            return 0.05 # Score base si no hay suficientes datos
            
        # El score de scikit-learn va de negativo (anomalía) a positivo (normal).
        # Lo invertimos y normalizamos entre 0 y 1.
        raw_score = self.iso_forest.score_samples([features])[0]
        anomaly_score = max(0.0, min(1.0, (raw_score * -1 + 0.5) / 1.0))
        return round(anomaly_score, 4)

cyber_manager = CyberSecurityManager()