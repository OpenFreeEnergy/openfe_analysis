**Fixed:**

* ``gather_rms_data`` now correctly identifies the ligand when cofactors
  share the same residue name ``UNK``. The ligand is auto-detected using
  hybrid topology tempfactors (b-factors), while custom ``ligand_selection``
  strings are still respected.
