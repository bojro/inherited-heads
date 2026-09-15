# Run scripts

The scripts that launched each batch of runs, kept as the exact record of which
conditions ran and in what order. They are a record, not a supported entry
point: each changes into `experiments/` and many wait for the previous batch's
done marker in a log file under the home directory.

Serial dependency: `run_e2_mass` -> `run_e2_sets` -> `run_e1_raw` -> `run_text`
-> `run_confirm_gated` -> `run_confirm` -> `run_overnight2`.

Every step is resumable: it skips if its `report.json` / `steer_report.json`
already exists, so a reboot costs at most the run in flight.
