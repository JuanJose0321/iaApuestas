"""
Cálculo dinámico de Elo para tenistas.

Sistema de rating basado en resultados de partidos.
Usa K-factor adaptativo según importancia del torneo.
"""
import logging
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import math

_log = logging.getLogger("betbrain.tennis_elo")

# K-factors por nivel de torneo
K_FACTORS = {
    "Grand Slam": 32,
    "Masters 1000": 24,
    "ATP 500": 20,
    "ATP 250": 16,
    "Challenger": 12,
    "ITF": 8,
}

INITIAL_ELO = 1500.0


@dataclass
class PlayerElo:
    """Almacena Elo actual de un jugador."""
    name: str
    elo: float
    games_played: int
    last_update: str  # fecha


class TennisEloCalculator:
    """Calcula y actualiza Elo de tenistas."""

    def __init__(self):
        self.players: Dict[str, PlayerElo] = {}

    def get_elo(self, player_name: str) -> float:
        """Obtiene Elo actual del jugador."""
        if player_name not in self.players:
            return INITIAL_ELO
        return self.players[player_name].elo

    def get_k_factor(self, tournament_level: str) -> int:
        """Obtiene K-factor según nivel de torneo."""
        return K_FACTORS.get(tournament_level, 16)

    def expected_score(self, elo_a: float, elo_b: float) -> float:
        """
        Calcula probabilidad esperada de victoria (Elo).
        Usa fórmula estándar de ajedrez.
        """
        diff = elo_b - elo_a
        return 1.0 / (1.0 + 10.0 ** (diff / 400.0))

    def update_elo(self, winner_name: str, loser_name: str,
                   tournament_level: str = "ATP 250",
                   winner_sets: int = 2, loser_sets: int = 0) -> Tuple[float, float]:
        """
        Actualiza Elo después de un partido.

        Args:
            winner_name: Nombre del ganador
            loser_name: Nombre del perdedor
            tournament_level: Nivel del torneo
            winner_sets: Sets ganados por el ganador
            loser_sets: Sets ganados por el perdedor

        Returns:
            (nuevo_elo_ganador, nuevo_elo_perdedor)
        """
        winner_elo = self.get_elo(winner_name)
        loser_elo = self.get_elo(loser_name)

        # K-factor base
        k = self.get_k_factor(tournament_level)

        # Ajuste por competitividad (sets jugados)
        total_sets = winner_sets + loser_sets
        set_factor = 1.0 + (total_sets - 1) * 0.1  # 2 sets = 1.1x, 3 sets = 1.2x
        k = k * set_factor

        # Expectativa
        expected_winner = self.expected_score(winner_elo, loser_elo)
        expected_loser = 1.0 - expected_winner

        # Cambio de Elo
        delta_winner = k * (1.0 - expected_winner)
        delta_loser = -k * expected_loser

        new_winner_elo = round(winner_elo + delta_winner, 1)
        new_loser_elo = round(loser_elo + delta_loser, 1)

        # Registrar
        if winner_name not in self.players:
            self.players[winner_name] = PlayerElo(winner_name, INITIAL_ELO, 0, "")
        if loser_name not in self.players:
            self.players[loser_name] = PlayerElo(loser_name, INITIAL_ELO, 0, "")

        self.players[winner_name].elo = new_winner_elo
        self.players[winner_name].games_played += 1
        self.players[loser_name].elo = new_loser_elo
        self.players[loser_name].games_played += 1

        _log.info(f"[ELO] {winner_name} {new_winner_elo} (Δ+{delta_winner:.1f}) vs "
                  f"{loser_name} {new_loser_elo} (Δ{delta_loser:.1f})")

        return new_winner_elo, new_loser_elo

    def batch_update(self, matches: list[dict]):
        """
        Actualiza Elo desde lista de partidos.

        Cada partido debe tener:
        {
            "winner": "Jannik Sinner",
            "loser": "Carlos Alcaraz",
            "tournament_level": "Grand Slam",
            "winner_sets": 3,
            "loser_sets": 1,
            "date": "2026-05-08"
        }
        """
        for match in matches:
            self.update_elo(
                match.get("winner", "Unknown"),
                match.get("loser", "Unknown"),
                match.get("tournament_level", "ATP 250"),
                match.get("winner_sets", 2),
                match.get("loser_sets", 0),
            )

    def get_ranking(self, top_n: int = 100) -> list[Tuple[str, float]]:
        """Devuelve ranking top N por Elo."""
        ranked = sorted(
            self.players.items(),
            key=lambda x: x[1].elo,
            reverse=True
        )
        return [(name, player.elo) for name, player in ranked[:top_n]]
