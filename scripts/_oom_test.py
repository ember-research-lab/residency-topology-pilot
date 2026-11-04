"""Cap self-test: allocate and touch GB chunks until killed. Must be OOM-killed inside the cgroup, not freeze the VM."""
import sys, numpy as np
a = []
for i in range(int(sys.argv[1])):
    a.append(np.ones(2**27, dtype=np.float64)); print("allocated", i + 1, "GB", flush=True)   # 1 GB each, touched
