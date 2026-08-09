-- BetBrain — esquema de Supabase
-- Ejecutar en: Supabase Dashboard → SQL Editor → New Query
--
-- Replica exactamente las columnas de src/services/tracking.py (COLUMNS y
-- bankroll_config.json) para que calcular_metricas() funcione idéntico
-- ya sea que las apuestas vivan en CSV local o en Supabase.

CREATE TABLE IF NOT EXISTS apuestas (
  id                 BIGSERIAL PRIMARY KEY,
  fecha_registro     TEXT,
  fecha_partido      TEXT,
  liga               TEXT,
  local              TEXT,
  visitante          TEXT,
  pick_tipo          TEXT,
  pick_descripcion   TEXT,
  cuota              NUMERIC(6, 3),
  stake              NUMERIC(10, 2),
  prob_predicha      NUMERIC(6, 4),
  ev_predicho        NUMERIC(6, 4),
  confianza_score    NUMERIC(6, 4),
  confianza_badge    TEXT,
  resultado          TEXT DEFAULT 'pendiente'
                     CHECK (resultado IN ('ganada', 'perdida', 'void', 'cashout', 'pendiente')),
  ganancia_neta      NUMERIC(10, 2),
  bankroll_antes     NUMERIC(10, 2),
  bankroll_despues   NUMERIC(10, 2),
  fixture_id_api     TEXT,
  notas              TEXT,
  created_at         TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_apuestas_resultado   ON apuestas(resultado);
CREATE INDEX IF NOT EXISTS idx_apuestas_created_at  ON apuestas(created_at DESC);

-- Tabla de una sola fila (id fijo = 1) — reemplaza data/bankroll_config.json
CREATE TABLE IF NOT EXISTS bankroll_config (
  id                     INT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
  bankroll_inicial       NUMERIC(10, 2) DEFAULT 800,
  bankroll_actual        NUMERIC(10, 2) DEFAULT 800,
  stake_modo             TEXT DEFAULT 'conservador',
  stake_fijo             NUMERIC(10, 2) DEFAULT 20,
  max_stake_pct          NUMERIC(5, 4) DEFAULT 0.03,
  racha_negativa_alerta  INT DEFAULT 5,
  fecha_inicio           DATE DEFAULT CURRENT_DATE
);

-- Log de CADA análisis de tenis corrido (con o sin pick de value), para medir
-- precisión real del modelo — independiente de `apuestas` (que solo guarda lo
-- que el usuario decidió apostar). Ver src/services/tennis_predictions.py.
-- std_dev no se guarda: es constante por formato (best_of_3/best_of_5), se
-- deriva de STD_GAMES al cargar el resultado, no depende del partido.
CREATE TABLE IF NOT EXISTS predicciones_tenis (
  id                 BIGSERIAL PRIMARY KEY,
  fecha_registro     TIMESTAMPTZ DEFAULT NOW(),
  fecha_partido      TEXT,
  jugador1           TEXT,
  jugador2           TEXT,
  favorito           TEXT,
  prob_favorito      NUMERIC(5, 4),
  superficie         TEXT,
  formato            TEXT,
  total_esp          NUMERIC(5, 2),
  tuvo_pick          BOOLEAN DEFAULT FALSE,
  tipo_pick          TEXT,
  ganador_real       TEXT,
  total_games_real   INTEGER,
  acerto_ganador     BOOLEAN,
  acerto_total       BOOLEAN,
  resultado_cargado  TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_predicciones_fecha ON predicciones_tenis(fecha_partido);

-- Row Level Security: la app accede solo con la service_role key (bypassa RLS),
-- así que estas tablas quedan cerradas para las claves públicas (anon/publishable).
ALTER TABLE apuestas ENABLE ROW LEVEL SECURITY;
ALTER TABLE bankroll_config ENABLE ROW LEVEL SECURITY;
ALTER TABLE predicciones_tenis ENABLE ROW LEVEL SECURITY;
