"""Mean score of a run, 0 if the run has no result."""
import json, sys
try:
    print(json.load(open(sys.argv[1]))['mean_cand'])
except OSError:
    print(0)
