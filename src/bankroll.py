"""
Gestión de bankroll: Kelly fraccional, stake fijo, flat % y registro de apuestas.
"""
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import ROOT_DIR, BANKROLL_INICIAL, KELLY_FRACTION


LEDGER_PATH = ROOT_DIR / "data" / "ledger.json"


def kelly(p: float, cuota: float, fraction: float = KELLY_FRACTION) -> float:
    """Fracción del bankroll a apostar (Kelly fraccional). 0 si EV <= 0."""
    b = cuota - 1
    if b <= 0:
        return 0.0
    f_full = (p * (b + 1) - 1) / b
    return max(0.0, f_full * fraction)


def stake_recomendado(bankroll: float, p: float, cuota: float,
                      metodo: str = "kelly_frac") -> float:
    """
    Calcula el stake en unidades monetarias según el método:
      - 'kelly_frac' : Kelly fraccional (KELLY_FRACTION de Kelly completo)
      - 'flat_1pct'  : 1% del bankroll
      - 'flat_2pct'  : 2% del bankroll
    """
    if metodo == "kelly_frac":
        return round(bankroll * kelly(p, cuota), 2)
    if metodo == "flat_1pct":
        return round(bankroll * 0.01, 2)
    if metodo == "flat_2pct":
        return round(bankroll * 0.02, 2)
    raise ValueError(f"Método desconocido: {metodo}")


@dataclass
class Apuesta:
    fecha: str
    partido: str
    mercado: str
    seleccion: str
    cuota: float
    stake: float
    prob_modelo: float
    ev: float
    resultado: str = "pendiente"   # pendiente | ganada | perdida | push
    pnl: float = 0.0


class Ledger:
    def __init__(self, bankroll_inicial: float = BANKROLL_INICIAL,
                 path: Path = LEDGER_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.bankroll_inicial = bankroll_inicial
        self.apuestas: list[Apuesta] = []
        self._cargar()

    def _cargar(self):
        if self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.bankroll_inicial = raw.get("bankroll_inicial", self.bankroll_inicial)
            self.apuestas = [Apuesta(**a) for a in raw.get("apuestas", [])]

    def _guardar(self):
        self.path.write_text(json.dumps({
            "bankroll_inicial": self.bankroll_inicial,
            "apuestas": [asdict(a) for a in self.apuestas],
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- Operaciones ---
    def registrar(self, partido: str, mercado: str, seleccion: str,
                  cuota: float, stake: float, prob_modelo: float, ev: float):
        a = Apuesta(
            fecha=datetime.now().isoformat(timespec="seconds"),
            partido=partido, mercado=mercado, seleccion=seleccion,
            cuota=cuota, stake=stake, prob_modelo=prob_modelo, ev=ev,
        )
        self.apuestas.append(a)
        self._guardar()
        return a

    def liquidar(self, indice: int, resultado: str):
        a = self.apuestas[indice]
        a.resultado = resultado
        if resultado == "ganada":
            a.pnl = round(a.stake * (a.cuota - 1), 2)
        elif resultado == "perdida":
            a.pnl = -a.stake
        elif resultado == "push":
            a.pnl = 0.0
        self._guardar()
        return a

    # --- Métricas ---
    def bankroll_actual(self) -> float:
        pnl = sum(a.pnl for a in self.apuestas if a.resultado != "pendiente")
        return round(self.bankroll_inicial + pnl, 2)

    def resumen(self) -> dict:
        cerradas = [a for a in self.apuestas if a.resultado in ("ganada", "perdida")]
        if not cerradas:
            return {
                "bankroll_inicial": self.bankroll_inicial,
                "bankroll_actual": self.bankroll_inicial,
                "apuestas": 0,
                "roi": 0.0, "yield": 0.0, "hit_rate": 0.0,
            }
        total_staked = sum(a.stake for a in cerradas)
        pnl = sum(a.pnl for a in cerradas)
        ganadas = sum(1 for a in cerradas if a.resultado == "ganada")
        return {
            "bankroll_inicial": self.bankroll_inicial,
            "bankroll_actual": self.bankroll_actual(),
            "apuestas": len(cerradas),
            "pendientes": sum(1 for a in self.apuestas if a.resultado == "pendiente"),
            "total_staked": round(total_staked, 2),
            "pnl": round(pnl, 2),
            "roi": round(pnl / self.bankroll_inicial, 4),
            "yield": round(pnl / total_staked, 4),
            "hit_rate": round(ganadas / len(cerradas), 4),
        }


if __name__ == "__main__":
    l = Ledger()
    print(json.dumps(l.resumen(), indent=2, ensure_ascii=False))
