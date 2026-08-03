"""Optional teacher mechanisms for the DiCode curriculum loop.

Every teacher in this package is OPT-IN via the Hydra ``teacher`` group
(``teacher=<name>``). With no teacher selected the legacy ``GenManager`` path is
used and behavior is byte-identical to the baseline.
"""
