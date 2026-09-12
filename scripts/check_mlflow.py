import mlflow
from pathlib import Path

print("=" * 60)
print("MLFLOW DIAGNOSTIC REPORT")
print("=" * 60)

# 1. Default tracking URI
default_uri = mlflow.get_tracking_uri()
print(f"1. Default Tracking URI: {default_uri}")
try:
    exps = mlflow.search_experiments()
    print("   Experiments in Default URI:")
    for e in exps:
        print(f"     * ID: {e.experiment_id} | Name: '{e.name}' | Location: {e.artifact_location}")
        try:
            runs = mlflow.search_runs(experiment_ids=[e.experiment_id])
            print(f"       Runs found: {len(runs)}")
        except Exception as r_err:
            print(f"       Could not query runs: {r_err}")
except Exception as err:
    print(f"   Error querying default: {err}")

# 2. Check ./mlruns explicitly
mlruns_path = Path("mlruns").resolve()
mlruns_uri = f"file:///{mlruns_path.as_posix()}"
print(f"\n2. Checking Local File Store URI: {mlruns_uri}")
try:
    mlflow.set_tracking_uri(mlruns_uri)
    exps = mlflow.search_experiments()
    print("   Experiments in ./mlruns:")
    for e in exps:
        print(f"     * ID: {e.experiment_id} | Name: '{e.name}' | Location: {e.artifact_location}")
        try:
            runs = mlflow.search_runs(experiment_ids=[e.experiment_id])
            print(f"       Runs found: {len(runs)}")
        except Exception as r_err:
            print(f"       Could not query runs: {r_err}")
except Exception as err:
    print(f"   Error querying ./mlruns: {err}")

# 3. Check sqlite:///mlflow.db explicitly
sqlite_db = Path("mlflow.db").resolve()
sqlite_uri = f"sqlite:///{sqlite_db.as_posix()}"
print(f"\n3. Checking SQLite DB URI: {sqlite_uri}")
try:
    mlflow.set_tracking_uri(sqlite_uri)
    exps = mlflow.search_experiments()
    print("   Experiments in SQLite DB:")
    for e in exps:
        print(f"     * ID: {e.experiment_id} | Name: '{e.name}' | Location: {e.artifact_location}")
        try:
            runs = mlflow.search_runs(experiment_ids=[e.experiment_id])
            print(f"       Runs found: {len(runs)}")
        except Exception as r_err:
            print(f"       Could not query runs: {r_err}")
except Exception as err:
    print(f"   Error querying SQLite: {err}")

print("=" * 60)
