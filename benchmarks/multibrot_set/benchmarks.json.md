1 Command './benchmarks/multibrot_set/multibrot_mojo_parallelize'
  runs:         10
  mean:      0.007 s
  stddev:    0.001 s
  median:    0.007 s
  min:       0.006 s
  max:       0.008 s

  percentiles:
     P_05 .. P_95:    0.006 s .. 0.008 s
     P_25 .. P_75:    0.007 s .. 0.007 s  (IQR = 0.001 s)

2 Command './benchmarks/multibrot_set/multibrot_codon'
  runs:         10
  mean:      0.044 s
  stddev:    0.001 s
  median:    0.044 s
  min:       0.043 s
  max:       0.046 s

  percentiles:
     P_05 .. P_95:    0.043 s .. 0.046 s
     P_25 .. P_75:    0.043 s .. 0.045 s  (IQR = 0.001 s)

3 Command './benchmarks/multibrot_set/multibrot'
  runs:         10
  mean:      0.136 s
  stddev:    0.001 s
  median:    0.136 s
  min:       0.135 s
  max:       0.138 s

  percentiles:
     P_05 .. P_95:    0.135 s .. 0.138 s
     P_25 .. P_75:    0.135 s .. 0.137 s  (IQR = 0.002 s)

4 Command 'python3 benchmarks/multibrot_set/__pycache__/multibrot.cpython-311.pyc'
  runs:         10
  mean:      5.444 s
  stddev:    0.023 s
  median:    5.445 s
  min:       5.408 s
  max:       5.491 s

  percentiles:
     P_05 .. P_95:    5.414 s .. 5.477 s
     P_25 .. P_75:    5.429 s .. 5.455 s  (IQR = 0.026 s)
