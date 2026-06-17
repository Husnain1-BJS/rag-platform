"""Full ingestion test - run directly."""
import time
import sys
sys.path.insert(0, "apps")

from ingestion.pipeline import run_nvd_pipeline_batched, run_mitre_pipeline

start = time.time()

print("=" * 60)
print("Phase 1: NVD 2025 - 44,482 CVEs (batched)")
print("=" * 60)

nvd_result = run_nvd_pipeline_batched(limit=None, batch_size=500, resume=False)
nvd_time = time.time() - start

print()
print("=" * 60)
print("Phase 2: MITRE ATT&CK - 697 techniques")
print("=" * 60)

mitre_start = time.time()
mitre_result = run_mitre_pipeline(limit=None, deduplicate=True)
mitre_time = time.time() - mitre_start

total_time = time.time() - start

print()
print("=" * 60)
print("INGESTION COMPLETE")
print("=" * 60)
print("NVD:  ", nvd_result)
print("MITRE:", mitre_result)
print("NVD time:  %.0fs (%.1fmin)" % (nvd_time, nvd_time/60))
print("MITRE time: %.0fs" % mitre_time)
print("Total time: %.0fs (%.1fmin)" % (total_time, total_time/60))
total_parsed = nvd_result.get("parsed", 0) + mitre_result.get("parsed", 0)
total_upserted = nvd_result.get("upserted", 0) + mitre_result.get("upserted", 0)
print("Total: %d parsed, %d upserted" % (total_parsed, total_upserted))
if total_time > 0:
    print("Speed: %.0f items/sec" % (total_parsed/total_time))
