
import os

# options:
#   -h, --help            show this help message and exit
#   --problem {eq17,eq18}
#   --algos [ALGOS ...]
#   --topologies [TOPOLOGIES ...]
#   --compressor {no,rand,top,gossip,quant}
#   --compress-rates [COMPRESS_RATES ...]
#   --quant-levels QUANT_LEVELS
#                         Used when compressor=quant.
#   --edge-num EDGE_NUM   Number of edges for random topology (optional).
#   --seed SEED
#   --T T
#   --m M
#   --rho RHO
#   --proj {l1,l2}
#   --adpds-proj-regret {l1,l2}
#                         Override ADPDS projection for regret curves (Eq.17 only).
#   --adpds-proj-violation {l1,l2}
#                         Override ADPDS projection for violation curves (Eq.17 only).
#   --R R
#   --c C
#   --c18 C18
#   --adpds-Ceta ADPDS_CETA
#   --adpds-Cbeta ADPDS_CBETA
#   --adpds-Cgamma ADPDS_CGAMMA
#   --eq17-tol EQ17_TOL
#   --viol-metric {post_sum_clip,pre_sum_clip,post_perstep,pre_perstep}
#   --dc-Cbeta DC_CBETA
#   --dc-Ceta DC_CETA
#   --dc-Calpha DC_CALPHA
#   --dc-gamma DC_GAMMA
#   --dc-Cgamma-dual DC_CGAMMA_DUAL
#                         Coefficient for dual step size (multiplis t^{-1/2}).
#   --dc-omega DC_OMEGA
#   --dc-nu DC_NU         Exploration radius ν for DC-DOPDGD zero-order estimator.
#   --dc-nu-exp DC_NU_EXP
#                         Decay exponent p for ν_t = ν / t^p.
#   --dc-batch DC_BATCH   Number of directions per step for bandit averaging.
#   --dc-common-random    Use common random direction across nodes each step.
#   --no-dc-common-random
#                         Disable common random.
#   --dc-beta-exp DC_BETA_EXP
#   --dc-eta-exp DC_ETA_EXP
#   --dc-alpha-exp DC_ALPHA_EXP
#   --output-dir OUTPUT_DIR
#   --combine-rates       Plot all compressor rates on shared regret/bits axes (Eq.17).
#   --combine-viol        Plot average violation curves across rates on shared axes (Eq.17).
#   --compare-compressors [COMPARE_COMPRESSORS ...]
#                         Compare specified compressor types for DC-DOPDGD (Eq.17 only).
#   --compare-topologies  Compare DC-DOPDGD across the provided topologies (Eq.17 only).

import subprocess
import shlex
import os

exe = r"C:\Users\bhjia\miniconda3\envs\sci4\python.exe"
py_file = r"c:\Users\bhjia\Desktop\SCI四区\exp\codework-main\codework-main\CC-DOCO\compare_algos.py"

# output_dir = "exp_dc-dopdgd"
# output_dir = "exp_dc-dopdgd_c_0.2"
# output_dir = "exp_dc-dopdgd_final_c_0.1"
# output_dir = "exp_1"
# output_dir = "exp_2"
# output_dir = "exp_4"
output_dir = "exp_7"

fixed_params = [
    "--problem", "eq17",
    "--algos", "dc-dopdgd",
    "--topologies", 'geom', 'ring', 'star', 'full',
    # "--topologies", 'geom', 
    "--seed", "42",
    "--output-dir", output_dir,
]


dcdopdgd_params = [
    "--c", "0.1",
    "--T", "800",
    "--m", "18",
    "--rho", "1.5",
    # "--compressor", 'rand',
    "--compress-rates", "0.10", "0.30",
    "--combine-rates",
    "--combine-viol",
    "--compare-compressors", 'rand', 'top', 'gossip', 'quant',
    "--compare-topologies",
    "--dc-batch", "256",
    "--dc-common-random",
    "--viol-metric", 'post_sum_clip',
    # 'post_sum_clip', 'pre_sum_clip', 'post_perstep', 'pre_perstep'
    "--dc-Cbeta", "0.26",
]

cmd = [exe, py_file] + fixed_params + dcdopdgd_params

print("Running:", shlex.join(cmd))

try:
    subprocess.run(cmd, check=True)
except subprocess.CalledProcessError as e:
    print("命令执行失败，返回码：", e.returncode)
