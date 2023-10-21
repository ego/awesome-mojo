1 Command './benchmarks/fibonacci_sequence/mojo_iteration'
  runs:        100
  mean:      0.001 s
  stddev:    0.000 s
  median:    0.001 s
  min:       0.001 s
  max:       0.005 s

  percentiles:
     P_05 .. P_95:    0.001 s .. 0.002 s
     P_25 .. P_75:    0.001 s .. 0.001 s  (IQR = 0.000 s)

2 Command './benchmarks/fibonacci_sequence/codon_iteration'
  runs:        100
  mean:      0.003 s
  stddev:    0.001 s
  median:    0.003 s
  min:       0.002 s
  max:       0.013 s

  percentiles:
     P_05 .. P_95:    0.002 s .. 0.004 s
     P_25 .. P_75:    0.002 s .. 0.003 s  (IQR = 0.000 s)

3 Command 'python3 benchmarks/fibonacci_sequence/python_iteration.py'
  runs:        100
  mean:      0.016 s
  stddev:    0.001 s
  median:    0.016 s
  min:       0.015 s
  max:       0.023 s

  percentiles:
     P_05 .. P_95:    0.016 s .. 0.017 s
     P_25 .. P_75:    0.016 s .. 0.017 s  (IQR = 0.001 s)

4 Command 'python3 benchmarks/fibonacci_sequence/__pycache__/python_iteration.cpython-311.pyc'
  runs:        100
  mean:      0.017 s
  stddev:    0.001 s
  median:    0.016 s
  min:       0.016 s
  max:       0.021 s

  percentiles:
     P_05 .. P_95:    0.016 s .. 0.018 s
     P_25 .. P_75:    0.016 s .. 0.017 s  (IQR = 0.001 s)

5 Command 'mojo run benchmarks/fibonacci_sequence/mojo_iteration.mojo'
  runs:        100
  mean:      0.044 s
  stddev:    0.001 s
  median:    0.044 s
  min:       0.042 s
  max:       0.049 s

  percentiles:
     P_05 .. P_95:    0.042 s .. 0.047 s
     P_25 .. P_75:    0.043 s .. 0.044 s  (IQR = 0.001 s)

6 Command 'codon run -release benchmarks/fibonacci_sequence/codon_iteration.codon'
  runs:        100
  mean:      0.628 s
  stddev:    0.010 s
  median:    0.628 s
  min:       0.613 s
  max:       0.663 s

  percentiles:
     P_05 .. P_95:    0.615 s .. 0.646 s
     P_25 .. P_75:    0.620 s .. 0.633 s  (IQR = 0.013 s)
