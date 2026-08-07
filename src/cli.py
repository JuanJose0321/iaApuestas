"""
CLI unificado de BetBrain.

Uso:
    python src/cli.py analyze "Real Madrid" "Barcelona"
    python src/cli.py backtest
    python src/cli.py clear-cache
    python src/cli.py pending-bets
    python src/cli.py diagnose
    python src/cli.py setup-data
    python src/cli.py train
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))


# ── analyze ──────────────────────────────────────────────────────────────────

def cmd_analyze(args):
    from src.engines.football import BettingEngine

    eng = BettingEngine()
    print(f"\n--- Análisis: {args.home} vs {args.away} ---")

    cuotas = {"1X2": {"1": args.c1, "X": args.cx, "2": args.c2}}
    if args.over and args.under:
        cuotas["OU_2.5"] = {"Over": args.over, "Under": args.under}
    if args.btts_yes and args.btts_no:
        cuotas["BTTS"] = {"Yes": args.btts_yes, "No": args.btts_no}

    result = eng.pick_multileg(args.home, args.away, cuotas)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


# ── backtest ─────────────────────────────────────────────────────────────────

def cmd_backtest(args):
    from src.core.backtest import backtest

    print("🔬 Corriendo backtest...")
    for metodo in ["kelly_frac", "flat_1pct", "flat_2pct"]:
        r = backtest(metodo=metodo)
        print(f"\n--- {metodo.upper()} ---")
        print(f"  Apuestas      : {r['apuestas']}")
        print(f"  Hit rate      : {r['hit_rate'] * 100:.2f}%")
        print(f"  Total staked  : {r['total_staked']:.2f}")
        print(f"  PnL           : {r['pnl']:+.2f}")
        print(f"  ROI           : {r['roi'] * 100:+.2f}%")
        print(f"  Yield         : {r['yield'] * 100:+.2f}%")
        print(f"  Bankroll final: {r['bankroll_final']:.2f}")


# ── clear-cache ───────────────────────────────────────────────────────────────

def cmd_clear_cache(args):
    from src.providers.league_manager import get_league_manager

    print("\n🧹 Limpiando caché de módulos...")
    for key in [k for k in sys.modules if k.startswith("src.")]:
        del sys.modules[key]

    manager = get_league_manager()
    print(f"✅ League manager recargado: {manager}")

    # Limpiar caché de disco si se pide
    if args.disk:
        from config import CACHE_DIR
        removed = 0
        for f in CACHE_DIR.glob("*.json"):
            f.unlink()
            removed += 1
        print(f"✅ Eliminados {removed} archivos de caché de API")


# ── pending-bets ──────────────────────────────────────────────────────────────

def cmd_pending_bets(args):
    from src.services.tracking import leer_historial

    historial = leer_historial()
    pendientes = [b for b in historial if b.get("resultado") == "pendiente"]

    if not pendientes:
        print("✅ No hay apuestas pendientes.")
        return

    print(f"\n📋 {len(pendientes)} apuestas pendientes:\n")
    for b in pendientes:
        print(f"  [{b.get('id', '?')}] {b.get('local')} vs {b.get('visitante')}")
        print(f"       {b.get('pick_descripcion')} @ {b.get('cuota')} — stake: {b.get('stake')}")
        print(f"       Fecha partido: {b.get('fecha_partido', 'N/A')}\n")


# ── diagnose ──────────────────────────────────────────────────────────────────

def cmd_diagnose(args):
    checks = []

    print("\n🔍 DIAGNÓSTICO DEL SISTEMA - BetBrain\n" + "=" * 50)

    # Python version
    v = f"{sys.version_info.major}.{sys.version_info.minor}"
    ok = sys.version_info >= (3, 8)
    print(f"{'✅' if ok else '❌'} Python {v}")
    checks.append(ok)

    # Dependencias
    for pkg in ["joblib", "numpy", "pandas", "sklearn", "xgboost", "scipy", "flask", "requests"]:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "?")
            print(f"✅ {pkg} {ver}")
            checks.append(True)
        except ImportError:
            print(f"❌ {pkg} NO instalado — pip install {pkg}")
            checks.append(False)

    # Modelos entrenados
    from config import MODEL_PATH, CALIBRATOR_PATH
    print(f"{'✅' if MODEL_PATH.exists() else '❌'} {MODEL_PATH.name}")
    print(f"{'✅' if CALIBRATOR_PATH.exists() else '❌'} {CALIBRATOR_PATH.name}")
    checks.append(MODEL_PATH.exists())
    checks.append(CALIBRATOR_PATH.exists())

    # Datos
    from config import RAW_DATA_DIR
    csvs = list(RAW_DATA_DIR.glob("*.csv"))
    ok = len(csvs) > 0
    print(f"{'✅' if ok else '❌'} {len(csvs)} CSVs en data/raw/")
    checks.append(ok)

    print("\n" + "=" * 50)
    if all(checks):
        print("✅ Sistema listo.")
    else:
        fails = checks.count(False)
        print(f"❌ {fails} problema(s) encontrado(s). Revisa los puntos marcados con ❌.")


# ── setup-data ────────────────────────────────────────────────────────────────

def cmd_setup_data(args):
    from src.providers.loader import descargar_todo, cargar_todo

    print("📥 Descargando datos históricos...")
    descargar_todo()
    df = cargar_todo()
    print(f"\n✅ {len(df)} partidos cargados.")
    print(f"   Rango: {df['Date'].min().date()} → {df['Date'].max().date()}")


# ── train ─────────────────────────────────────────────────────────────────────

def cmd_train(args):
    from src.core.model import entrenar
    entrenar()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="betbrain",
        description="BetBrain CLI — herramienta de análisis de apuestas deportivas",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # analyze
    p_analyze = sub.add_parser("analyze", help="Analiza un partido con cuotas dadas")
    p_analyze.add_argument("home", help="Equipo local")
    p_analyze.add_argument("away", help="Equipo visitante")
    p_analyze.add_argument("--c1", type=float, required=True, help="Cuota local (1X2)")
    p_analyze.add_argument("--cx", type=float, required=True, help="Cuota empate (1X2)")
    p_analyze.add_argument("--c2", type=float, required=True, help="Cuota visitante (1X2)")
    p_analyze.add_argument("--over", type=float, help="Cuota Over 2.5")
    p_analyze.add_argument("--under", type=float, help="Cuota Under 2.5")
    p_analyze.add_argument("--btts-yes", type=float, dest="btts_yes", help="Cuota BTTS Sí")
    p_analyze.add_argument("--btts-no", type=float, dest="btts_no", help="Cuota BTTS No")
    p_analyze.set_defaults(func=cmd_analyze)

    # backtest
    p_bt = sub.add_parser("backtest", help="Corre backtest histórico")
    p_bt.set_defaults(func=cmd_backtest)

    # clear-cache
    p_cc = sub.add_parser("clear-cache", help="Limpia caché de módulos y/o disco")
    p_cc.add_argument("--disk", action="store_true", help="También borra caché de API en disco")
    p_cc.set_defaults(func=cmd_clear_cache)

    # pending-bets
    p_pb = sub.add_parser("pending-bets", help="Lista apuestas pendientes de resultado")
    p_pb.set_defaults(func=cmd_pending_bets)

    # diagnose
    p_dx = sub.add_parser("diagnose", help="Verifica que el sistema está listo")
    p_dx.set_defaults(func=cmd_diagnose)

    # setup-data
    p_sd = sub.add_parser("setup-data", help="Descarga datos históricos de football-data.co.uk")
    p_sd.set_defaults(func=cmd_setup_data)

    # train
    p_tr = sub.add_parser("train", help="Entrena el modelo XGBoost")
    p_tr.set_defaults(func=cmd_train)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
